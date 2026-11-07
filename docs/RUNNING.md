# Running the study

Run local commands from the repository root. `src/` holds runtime implementations rather than notebook-authoring scripts. Python resolves sibling modules when a source file is run directly; use `PYTHONPATH=src:.` when importing them from another entry point.

## Inputs and environments

Obtain the original SCPCP000004 matrix/metadata and independently reviewed annotation ledgers separately. Set `SCPCA_H5AD` and `SCPCA_METADATA` in the shell before starting local notebooks. `.env.example` is a template and is not loaded automatically. Keep all private inputs under ignored `data/` or `outputs/` paths.

`requirements/analysis.txt` specifies the biological-analysis stack. `requirements/test.txt` includes the CPU model and mocked judge dependencies. Training used an NVIDIA A100-SXM4-80GB, BF16, PyTorch 2.11.0+cu130, Transformers 4.53.0 and PEFT 0.17.1. Install a compatible CUDA PyTorch wheel before `requirements/training.txt`. Install causal-conv1d 1.7.0 before mamba-ssm 2.3.2.post1 with `--no-build-isolation`. The training notebook checks the actual GPU/kernel environment.

## Data and expert targets

1. Run sections 1–3 of `notebooks/00_Data_Preparation.ipynb` locally. These perform metadata-only donor splitting, then sample-specific training/validation clustering and evidence extraction. Technical libraries are combined within each biological sample. Counts are normalized to 10,000 per cell, log-transformed, and processed with HVG/PCA/neighbors/Leiden and Wilcoxon markers. The train/validation generator rejects test cells.
2. Inspect the generated technical QC plots, then run section 4 to record the reviewed recipe. Run section 5 to generate evidence from reserved donors using those same configuration/source/environment checks. These checks are scientific preprocessing controls, not code-generation tools.
3. Obtain independent expert annotations. Reviewer workbook construction and HTML review-site authoring are not distributed in this runtime package. To use the original ingestion functions, supply completed workbooks with the field layouts defined in `src/ingest_independent_bio_review.py` and `src/ingest_test_expert_review.py`, plus the generated case manifests. The training workbook path is `outputs/sample-baseline-review-20261101/reviewer-package/01-independent-bio-review-expert.xlsx`; the test workbook is `outputs/test-review-67-cases/01-independent-test-review-expert-feedback.xlsx`.
4. The target preparation preserves original expert wording, uncertainty and evidence provenance. It expects the preserved baseline prompts in `outputs/sample-baseline-review-20261101/baseline-records-preserved.jsonl`, a private run artifact that is not included. Only the 101 training cases enter the adapter package.

```bash
python src/ingest_independent_bio_review.py
python src/prepare_expert_training.py
python src/ingest_test_expert_review.py
```

The counts and input locations reproduce the original study's contracts; this is not a generic arbitrary-cohort command-line service.

## Training and evaluation

1. Run `SCPCP000004_Clean_Training.ipynb` in Colab using the prepared training package. The trainer uses rank-16 LoRA, assistant-target loss masking, three epochs and 39 updates, and saves compatible checkpoint/resume state. Training and evaluation inputs remain separated.
2. Run `SCPCP000004_Clean_Matched_4B_Evaluation.ipynb` with the trained V1 adapter. Place the returned validation archive at `outputs/nemotron_4b_matched_development_v1.zip`; score it with `python src/score_matched_development.py`. The scorer needs the independent validation ledger and original run manifests.
3. After test references are available, return to section 6 of `00_Data_Preparation.ipynb`. It fixes the scoring map before inference and prepares `outputs/frozen-test-evaluation-v1/frozen-test-inference-v1.zip`. This is runtime data packaging; it does not create a notebook or author Python source. Copy the printed SHA-256 into `EXPECTED_PACKAGE_SHA256` in the existing frozen-test notebook. Retain the local scoring protocol; never upload expert answers with inference inputs.
4. Run `SCPCP000004_Frozen_Test_Evaluation.ipynb` with the same pinned V1 adapter. The inference package contains expression evidence and runtime modules. It runs both identity and separate review passes and retains terminal failures. Returned output belongs at `outputs/nemotron_4b_frozen_test_v1.zip`.
5. Run `python src/score_frozen_test.py`. Optionally run `python src/verify_frozen_test_results.py --tokenizer-local /path/to/pinned-tokenizer-snapshot` to independently reproduce macro-F1 and audit saved tokens. A local snapshot at revision `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f` is required for token auditing; no model weights are loaded.

Moving source files changes certain hashes. A new run must use its own generated manifests, protocol and input archives. Do not disable integrity checks to combine this reorganized source with historical execution manifests. The included aggregate results are historical observations, not newly rerun GPU results.

The whole-cohort analysis notebook is exploratory context, not independent test evidence. Prior investigator exposure to that exploration remains a study limitation.

## DeepEval

Prepare the private input package from the preserved test answers and independent expert ledger:

```bash
python -m pip install -r requirements/deepeval.txt
python evaluation/prepare_data.py
```

It creates data/configuration files and copies the existing runner into a private execution ZIP; it does not author source or generate a notebook. Use the included live-progress DeepEval notebook and a key supplied through Colab Secrets as `NVIDIA_API_KEY`. There are 134 saved identity answers × 3 criteria = 402 logical judgments. Only expert-reference fidelity receives expert text. All requests go to NVIDIA's hosted API. Do not commit generated inputs or returned requests/responses.

Audit a returned archive against its original private source package:

```bash
python evaluation/audit_results.py --archive /path/to/nemotron120b_deepeval_v1.zip --package outputs/deepeval-nemotron120b-v1 --predictions outputs/final-test-results/predictions-preserved.jsonl --output outputs/deepeval-audit
```

The audit is intentionally specific to the original complete 67-case, three-criterion study. The fixed rubric and historical protocol snapshot are under `evaluation/`; a new preparation records its own source hashes. Preserve failures and do not tune the rubric after viewing scores.
