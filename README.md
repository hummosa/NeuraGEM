# NeuraGEM

NeuraGEM (Neural Gradient-based Expectation-Maximization) is a computational framework for learning latent task structure with fast latent updates and slower synaptic weight updates. This repository contains the simulation, analysis, and figure-generation code for the NeuraGEM paper.


## Quick Start

The easiest way to see something happening and run code is:

```bash
python run_training_behavior.py
```

This runs several behavioral conditions (NeuraGEM and RNN baselines) and exports compact figures.

For a flexible sandbox script, use:

```bash
python run_example.py
```

In `run_example.py` you can:

- Load either `ContextualSwitchingTaskConfig` (contextual switching task) or `seq_learnConfig` (sequence learning task).
- Run NeuraGEM (default) or an RNN baseline by setting `config.LU_lr = 0`.
- Run a long-horizon RNN baseline by setting `config.LU_lr = 0` and `config.seq_len = 50`.

## Code Organization

Files were renamed so experiment scripts are grouped by prefix:

- `seq_learn*`: human sequence learning task
- `cst*`: contextual switching task
- `em*`: comparison between EM and NeuraGEM
- `time_scales*`: multiple time scales learning
- `neural*`: neural dissection and analysis

General/shared components are in files such as `configs.py`, `models.py`, `datasets.py`, `train_and_infer_functions.py`, and `functions_and_utils*.py`.

## Paper Figures to Code (rough mapping)

- Figures 2-3 (contextual switching generalization)
- Run experiments: `cst_run_generalization.py`
- Plot/analyze: `cst_analyze_gen_tests.py`

- Figure 4 (EM vs NeuraGEM)
- Run/demo: `em_3_clusters_em_demo.ipynb`
- Run script: `em_neuragem_3_clusters.py`

- Figure 5 (multiple time scales)
- Run experiments: `time_scales_nested_run.py`
- Plot/analyze: `time_scales_nested_analyze.py`

- Figure 6 (sequence learning)
- Run experiments: `seq_learn_run.py`
- Plot/analyze: `seq_learn_analyze.py`

- Figure 6 supplementary
- Run experiments: `seq_learn_supplementary_run.py`
- Plot/analyze: `seq_learn_supplementary_analyze.py`

- Figure 7 (neural dissection)
- Run experiments/logging: `adapt_run_array_input_z_sweeps.py`
- Plot/analyze: `neural_ccgp_modules.py`, `neural_analysis_behavior_dynamics.py`, `neural_internal_dynamics_analysis.py`

## Running Large Experiments (Cluster or Local)

Most experiment scripts are intended for cluster use because reproducing paper-scale results often requires hundreds to thousands of runs (seeds x parameter sweeps). The same scripts can also be run on a single machine:

- On a cluster: use Slurm array jobs (recommended for large sweeps).
- On a single machine: run the script directly; many scripts will run sequentially when `SLURM_ARRAY_TASK_ID` is not set.

### Slurm helper script

Use `submit_adapt_job.sh` to submit array jobs:

```bash
sh ./submit_adapt_job.sh <MAX_TASK_ID> <EXPERIMENT_NAME>
```

Examples:

```bash
sh ./submit_adapt_job.sh 299 generalization_tests
sh ./submit_adapt_job.sh 119 seq_learn
```

Notes:

- `<MAX_TASK_ID>` is the highest array index (the script submits `0..MAX_TASK_ID`).
- The script currently supports:
  - `generalization_tests` -> `cst_run_generalization.py`
  - `input_z_sweeps` -> `adapt_run_array_input_z_sweeps.py`
  - `seq_learn_supp` -> `seq_learn_supplementary_run.py`
  - `seq_learn_interleaved_phase` -> `seq_learn_varying_interleaved_phase_run.py`
  - `seq_learn` -> `seq_learn_run.py`
  - `time_scales` -> `time_scales_nested_run.py`
- The helper script is just a convenience wrapper. If you add/rename experiment files, update the mapping in `submit_adapt_job.sh`.
- The script writes Slurm logs to `./slurm/`.

## Disclaimer

Code was refactored and organized as a last step using an LLM agent. Some minor discrepancies are possible. 

## Citation

If you build on this codebase, please cite the accompanying manuscript:

```bibtex
@article{hummos2026neuragem,
  title={A neural architecture for rapid learning of latent task states},
  author={Hummos, Ali and Wang, Mien Brabeeba and Lu, Qihong and Norman, Kenneth A. and Jazayeri, Mehrdad},
  year={2026},
  note={Preprint}
}
```

## License

MIT License. See `LICENSE`.
