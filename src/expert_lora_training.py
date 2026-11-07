"""Colab helper: explicit GPU check, then separately invoked resumable LoRA training."""
import hashlib
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path


def assistant_labels(prompt_ids, full_ids, eos):
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError('Chat template did not preserve the exact generation prefix')
    stop = full_ids.index(eos, len(prompt_ids)) + 1
    ids = full_ids[:stop]
    return ids, [-100] * len(prompt_ids) + ids[len(prompt_ids):]


class ExpertTraining:
    def __init__(self, model, tokenizer, package, output):
        import torch
        from peft import LoraConfig, get_peft_model
        self.torch, self.tokenizer = torch, tokenizer
        self.package, self.output = Path(package), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((self.package / 'package-manifest.json').read_text())
        for name, expected in manifest['files'].items():
            assert hashlib.sha256((self.package / name).read_bytes()).hexdigest() == expected, name
        self.config = c = json.loads((self.package / 'training-config.json').read_text())
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        assert tokenizer.eos_token_id == c['eos_token_id']
        assert getattr(model.config, '_commit_hash', None) == c['revision'], 'Loaded model revision differs.'
        assert getattr(model, 'peft_config', None) is None, 'Use an unadapted base model; reuse the existing training object instead of initializing twice.'
        random.seed(c['seed']); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
        self.examples = []
        for line in (self.package / 'expert-train.jsonl').read_text().splitlines():
            record = json.loads(line)
            assert record['partition'] == 'train'
            messages = record['messages']
            prompt = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
            ids, labels = assistant_labels(prompt, full, tokenizer.eos_token_id)
            assert len(ids) <= min(c['max_sequence_length'], model.config.max_position_embeddings), record['case_id']
            assert any(v != -100 for v in labels)
            self.examples.append({'case_id': record['case_id'], 'input_ids': ids, 'labels': labels})
        assert len(self.examples) == len({e['case_id'] for e in self.examples}) == c['training_examples']
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and name.rsplit('.', 1)[-1] in c['target_modules']]
        assert targets and {name.rsplit('.', 1)[-1] for name in targets} == set(c['target_modules'])
        model.config.use_cache = False
        self.net = get_peft_model(model, LoraConfig(
            task_type='CAUSAL_LM', r=c['rank'], lora_alpha=c['alpha'], lora_dropout=c['dropout'],
            target_modules=targets, bias='none',
        ))
        self.net.enable_input_require_grads()
        self.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        trainable = [(n, p) for n, p in self.net.named_parameters() if p.requires_grad]
        assert trainable and all('lora_' in n for n, _ in trainable)
        self.params = [p for _, p in trainable]
        self.optimizer = torch.optim.AdamW(self.params, lr=c['learning_rate'], weight_decay=0)
        self.updates_per_epoch = math.ceil(len(self.examples) / c['gradient_accumulation'])
        self.total_updates = self.updates_per_epoch * c['epochs']
        self.next_group, self.elapsed_training = 0, 0.0
        identity = {
            'config': c, 'package_manifest': manifest, 'adapter_modules': targets,
            'software': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'peft', 'mamba-ssm', 'causal-conv1d']},
        }
        self.identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.run_path = self.output / 'run-manifest.json'
        if self.run_path.exists():
            assert json.loads(self.run_path.read_text())['identity_hash'] == self.identity_hash, 'Training configuration or environment changed.'
        else:
            self.run_path.write_text(json.dumps({**identity, 'identity_hash': self.identity_hash,
                'gpu': torch.cuda.get_device_name(0), 'trainable_parameters': sum(p.numel() for p in self.params),
                'total_parameters': sum(p.numel() for p in self.net.parameters()),
                'max_observed_sequence_length': max(len(e['input_ids']) for e in self.examples),
                'total_optimizer_updates': self.total_updates, 'cloud_cost': None}, indent=2))
        state_path = self.output / 'training-state.pt'
        if state_path.exists():
            from peft import set_peft_model_state_dict
            state = torch.load(state_path, map_location='cpu', weights_only=False)
            assert state['identity_hash'] == self.identity_hash
            set_peft_model_state_dict(self.net, state['adapter'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.next_group = state['next_group']; self.elapsed_training = state['elapsed_training_seconds']
            random.setstate(state['random_state']); torch.set_rng_state(state['torch_state'])
            torch.cuda.set_rng_state_all(state['cuda_state'])
            del state
        self.net.print_trainable_parameters()
        print(f'Prepared {len(self.examples)} training examples; {self.total_updates} optimizer updates; resume at {self.next_group}.', flush=True)
        self.checked = False

    def _loss(self, example):
        torch = self.torch
        ids = torch.tensor([example['input_ids']], device='cuda')
        labels = torch.tensor([example['labels']], device='cuda')
        return self.net(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                        use_cache=False, return_dict=True).loss

    def check(self):
        """Longest-case forward/backward; no optimizer step or weight updates."""
        torch = self.torch
        case = max(self.examples, key=lambda e: len(e['input_ids']))
        self.net.train(); self.optimizer.zero_grad(set_to_none=True)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        start = time.perf_counter()
        loss = self._loss(case)
        assert bool(torch.isfinite(loss)), 'Non-finite pilot loss'
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        assert bool(torch.isfinite(norm)) and float(norm) > 0, 'Missing or invalid adapter gradients'
        assert all(p.grad is None for n, p in self.net.named_parameters() if 'lora_' not in n)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        result = {'case_id': case['case_id'], 'tokens': len(case['input_ids']), 'loss': float(loss.detach()),
                  'adapter_gradient_norm': float(norm), 'forward_backward_seconds': seconds,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                  'rough_remaining_minutes_from_longest_case': seconds * len(self.examples) * self.config['epochs'] / 60,
                  'weight_updates_performed_by_check': 0,
                  'note': 'Time estimate excludes optimizer/checkpoint overhead; no monetary cost inferred.'}
        self.optimizer.zero_grad(set_to_none=True); del loss
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
        torch.cuda.empty_cache()
        (self.output / 'gpu-check.json').write_text(json.dumps(result, indent=2))
        self.checked = True
        print(json.dumps(result, indent=2), flush=True)
        return result

    def _save(self):
        from peft import get_peft_model_state_dict
        torch = self.torch
        state = {'identity_hash': self.identity_hash, 'adapter': get_peft_model_state_dict(self.net),
                 'optimizer': self.optimizer.state_dict(), 'next_group': self.next_group,
                 'elapsed_training_seconds': self.elapsed_training,
                 'random_state': random.getstate(), 'torch_state': torch.get_rng_state(),
                 'cuda_state': torch.cuda.get_rng_state_all()}
        temp = self.output / 'training-state.tmp'
        torch.save(state, temp); temp.replace(self.output / 'training-state.pt')

    def train(self):
        assert self.checked, 'Run training.check() before full training.'
        torch = self.torch
        started = time.perf_counter()
        groups = []
        for epoch in range(self.config['epochs']):
            order = list(range(len(self.examples)))
            random.Random(self.config['seed'] + epoch).shuffle(order)
            size = self.config['gradient_accumulation']
            groups.extend((epoch, order[start:start+size]) for start in range(0, len(order), size))
        self.net.train()
        for group_index in range(self.next_group, len(groups)):
            epoch, indexes = groups[group_index]
            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            step_start = time.perf_counter()
            for example_index in indexes:
                loss = self._loss(self.examples[example_index])
                assert bool(torch.isfinite(loss)), 'Non-finite training loss; last checkpoint preserved.'
                losses.append(float(loss.detach()))
                (loss / len(indexes)).backward()
                del loss
            norm = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            assert bool(torch.isfinite(norm)), 'Non-finite gradients; last checkpoint preserved.'
            # Linear warmup for the first two optimizer updates, constant thereafter.
            lr = self.config['learning_rate'] * min(1.0, (group_index + 1) / 2)
            for group in self.optimizer.param_groups:
                group['lr'] = lr
            self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            self.elapsed_training += time.perf_counter() - step_start
            self.next_group = group_index + 1
            self._save()
            log = {'update': self.next_group, 'total_updates': len(groups), 'epoch': epoch + 1,
                   'mean_loss': sum(losses) / len(losses), 'learning_rate': lr,
                   'elapsed_training_seconds': self.elapsed_training}
            with (self.output / 'training-log.jsonl').open('a') as f:
                f.write(json.dumps(log) + '\n')
            print(json.dumps(log), flush=True)
            if time.perf_counter() - started >= self.config['training_wall_time_limit_minutes'] * 60:
                print('Time cap reached; checkpoint saved. Call train() again to resume.', flush=True)
                break
        complete = self.next_group == len(groups)
        self.net.save_pretrained(self.output / 'adapter')
        self.tokenizer.save_pretrained(self.output / 'adapter')
        result = {'status': 'complete' if complete else 'partial', 'optimizer_updates': self.next_group,
                  'expected_updates': len(groups), 'training_compute_seconds': self.elapsed_training,
                  'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                  'adapter_folder': str(self.output / 'adapter'), 'cloud_cost': None}
        (self.output / 'training-status.json').write_text(json.dumps(result, indent=2))
        self.net.eval()
        print(json.dumps(result, indent=2), flush=True)
        return result
