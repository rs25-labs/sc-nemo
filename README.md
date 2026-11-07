# Nemotron single-cell annotation

Expert-guided adaptation of NVIDIA Nemotron 3 Nano 4B for single-cell annotation in pediatric neuroblastoma, with donor-separated evaluation and supplementary DeepEval assessment.

The study uses public OpenScPCA project **SCPCP000004**. On 57 expert-resolved test clusters, broad cell-type agreement increased from **29/57 (50.9%)** untuned to **39/57 (68.4%)** after adaptation; macro-F1 increased from **0.301 to 0.465**. Uncertainty recognition remained a limitation. See [the study summary](docs/STUDY.md) and [aggregate results](results/aggregate.json).

## Repository

| Folder | Contents |
|---|---|
| `notebooks/` | Runnable data preparation, exploratory analysis, training, matched validation, frozen test and DeepEval notebooks |
| `src/` | Clustering/evidence extraction, expert-target preparation, LoRA trainer, inference and scoring implementations |
| `evaluation/` | DeepEval data preparation, NVIDIA judge runner, rubric and saved-response audit |
| `figures/` | Result, training-loss and source-coordinate UMAP plotting |
| `results/` | Aggregate metrics and recorded training loss |
| `tests/` | Offline CPU tests with synthetic data and mocked API responses |

There are no standalone notebook generators, dependency-repair utilities, poster-authoring scripts, or historical V2/V3 handoff tools in this repository.

## Setup and tests

Use Python 3.12 from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/test.txt
PYTHONPATH=src:. DEEPEVAL_TELEMETRY_OPT_OUT=1 python -m unittest discover -s tests -v
```

Tests do not require study data, a GPU, model checkpoint downloads or API credentials. For notebooks, install Jupyter separately in the analysis environment, or use the supplied Colab notebooks as appropriate. GPU training requires compatible PyTorch/CUDA and Mamba kernels; see [running the study](docs/RUNNING.md).

## Run the study

Start with [00_Data_Preparation.ipynb](notebooks/00_Data_Preparation.ipynb). The local notebook performs metadata-only donor splitting and records the reviewed preprocessing recipe directly. Its last section prepares expression-only test inputs after independent expert review and matched validation.

The training, matched-validation and frozen-test notebooks run in Colab with an A100 runtime and externally supplied training/inference packages. The DeepEval notebook runs on saved answers using NVIDIA's hosted API; expert references enter only the fidelity criterion. See [required inputs and execution order](docs/RUNNING.md).

## Plot results

```bash
python figures/plot_results.py
python figures/plot_training_loss.py
# Requires local saved coordinates and the expert display ledger:
python figures/plot_umap.py
```

Plots go to ignored `outputs/figures/`. The logged training loss covers 39 updates; validation loss was not recorded.

## Data and limitations

Raw expression matrices, reviewer workbooks, expert prose, full model responses, model weights and API keys are external inputs and are not included. The source-coordinate UMAP renderer also requires local coordinates; aggregate plots require only the included results.

This is one small study with one expert, one training seed, four test donors and no validated subtype/malignancy endpoint. DeepEval scores are exploratory rubric ratings, not biological accuracy percentages. Private deployment, researcher-time savings and superiority over other annotation tools were not demonstrated. See [study limitations](docs/STUDY.md).

## License

Original code and documentation are [MIT licensed](LICENSE). Data, model weights and dependencies retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

Validation: 25 offline CPU tests passed after cleanup; aggregate and training-loss plots exported successfully. Full GPU training/inference and hosted judge calls were not rerun for this source reorganization.
