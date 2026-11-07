# Study record and interpretation

The experiment asks whether a compact model adapted with limited expert examples can improve broad single-cell annotation on reserved donors. It used public neuroblastoma data and one anonymous expert reference. The objective of lab-controlled deployment motivated the study; private/offline deployment was not demonstrated.

## Primary evidence

The frozen V1 adapter trained on 101 cluster examples from 12 donors. Four validation donors contributed 21 cases, used for development/model selection. The final four test donors contributed 67 cases across six samples. Identity agreement uses only the 57 expert-resolved test cases. Ten uncertain cases are retained for review evaluation, not forced into hard identity scoring.

V1 proposed identities matched the expert on 39/57 cases (68.4%), versus 29/57 (50.9%) untuned. Macro-F1 is 0.465 versus 0.301 across nine broad reference groups, with each cluster counting equally. The locked exact alias map counts unlisted wording as disagreement. Non-agreement is not always an independently confirmed biological error.

V1 originally requested review for none of the 67 cases, missing every uncertain case. A separate review gate flagged one uncertain case and two resolved mismatches, accepting 64 cases: 39 supported matches, 16 resolved mismatches and nine uncertain cases. The 70.9% agreement among accepted resolved cases (39/55) must not be called accuracy across all 64 accepted cases (39/64 supported matches,60.9%).

All 134 identity responses completed. Across both model arms and both stages, 267/268 attempts completed; one timeout remained counted. Gene-citation traceability improved from 90.3% to 99.2%, but citation presence does not establish correct biological interpretation.

## Supplementary automated assessment

A fixed post hoc DeepEval rubric rated 134 saved identity answers on three criteria. All 402 scores completed. Mean ratings increased for evidence consistency (0.499 to0.561), restraint of unsupported claims (0.522 to0.588), and expert-reference fidelity (0.349 to0.576). These are normalized rubric ratings, not accuracy percentages. Fidelity on the ten uncertain cases was unchanged on average (0.280 in both arms).

The judge came from the Nemotron family and may have related-family/style biases. There was no independent human validation of the judge ratings. Three unchanged candidate answers echoed arbitrary case keys; no additional arm/donor/sample metadata was given to the judge. Only the fidelity criterion received expert text. One spot-checked restraint rating criticized an explicitly tentative malignancy alternative, illustrating potential judge imperfections.

## Limits

One expert, one training seed, four test donors, uneven groups, an analyst-defined alias map and earlier whole-cohort exploratory exposure limit conclusions. This is not an external-cohort or prospectively preregistered study. Broad lineage agreement does not validate malignancy or subtype/state. Expert-time savings, total dollar cost, private deployment, and superiority over CASSIA/RAG were not measured or established.

## Sources

- OpenScPCA: https://openscpca.readthedocs.io/
- ScPCA processing: https://scpca.readthedocs.io/en/stable/processing_information.html
- NVIDIA base model: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
- LoRA: https://arxiv.org/abs/2106.09685
- CASSIA: https://doi.org/10.1038/s41467-025-67084-x
- DeepEval G-Eval: https://deepeval.com/docs/metrics-llm-evals
- NVIDIA judge API: https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-super-vl-120b-a12b-infer
