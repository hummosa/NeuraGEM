# NeuraGEM: Neural Gating and Expectation Maximization

## Project Architecture

This is a PyTorch-based research project implementing **Neural Gating with meta-learning** for contextual switching tasks. The core architecture separates **weight updates (WU)** and **latent updates (LU)** using dual optimizers operating on different timescales.

### Core Model: `RNN_with_latent`

The central model (`models.py`) implements recurrent networks with inference-time latent variables (`Z`) that gate dynamics:

- **Dual optimization**: `W_optimizer` (weights) and `Z_optimizer` (latent variables) 
- **Gating modes**: Multiplicative (`use_mul_gating`) vs Additive (`use_add_gating`)
- **Latent management**: `init_Z()`, `update_Z()`, `adjust_Z_grads()` with various aggregation operations
- **Legacy aliases**: `WU_optimizer`/`LU_optimizer`, `update_latent()`/`update_Z()`

Key pattern: Models maintain separate state for weights and latents with independent learning rates (`WU_lr`, `LU_lr`).

## Configuration System

All experiments use hierarchical configs (`configs.py`) with task-specific subclasses:

```python
# Base config with extensive latent/gating parameters
Config() 
├── ContextualSwitchingTaskConfig()  # Main contextual switching
├── HierarchicalContextualSwitchingTaskConfig()  # Hierarchical patterns
├── SeqLearnConfig()  # Context-dependent sequence learning  
├── CoinConfig()  # Binary prediction task
├── TafazoliTaskConfig()  # RL classification task
└── HierarchicalReasoningConfig()  # Perceptual decision making
```

**Key config patterns:**
- `experiment_to_run`: Controls task variant ('figure', 'tweaking', 'rnn', 'few_long_blocks')
- **Phase control**: `add_passive_learning_phase`, `add_interleaved_phase`, `add_blocked_phase`
- **Latent config**: `latent_dims`, `latent_chunks`, `latent_aggregation_op`, `latent_activation`
- **Training phases**: `passive_phase_length`, `interleaved_phase_length`, `blocked_phase_length`

## Training Workflow

### Multi-Phase Training (`train_and_infer_functions.py`)

1. **Passive phase**: `_allow_latent_updates = False` - pure predictive learning
2. **Active phase**: Latent optimization enabled with `update_Z()`
3. **Testing phase**: `reconfigure_for_prediction()` - inference only

### Dataset Integration (`datasets.py`)

Each task implements PyTorch `Dataset` with:
- **Contextual switching**: Block-structured mean changes with hierarchical latents
- **CSW task**: 6-step sequences with context-dependent transitions  
- **COIN task**: Binary sequence prediction with spontaneous recovery variants
- **Batch format**: `(inputs, low_level_latents, high_level_latents)`

## Experiment Management

### Array Job System

- **Runners**: `*_run_array.py` scripts handle parameter sweeps
- **Analysis**: `*_analyze_*.py` scripts process results across parameter combinations  
- **SLURM integration**: `submit_adapt_job.sh` for HPC job submission
- **Storage**: Results saved as pickled loggers in `exports/` directory structure

### Parameter Sweeps

Use `generate_param_combinations()` pattern for systematic exploration:
```python
param_grid = {
    'WU_lr': [1e-3, 1e-4],
    'l2_loss': [0.0008, 0.001], 
    'seed': list(range(20))
}
```

## Critical Development Patterns

### Latent Optimization Mechanics

- **Gradient aggregation**: `latent_aggregation_op` controls temporal gradients ('average', 'exponential_increase', 'per_latent_chunk')
- **Activation functions**: `latent_activation` ('softmax', 'softmax_chunked', 'sigmoid', 'none')
- **Exponential filtering**: `exponential_increase_steepness` and `exponential_increase_multipliers` for temporal weighting

### Logging Infrastructure (`functions_and_utils.py`)

Comprehensive `Logger` class tracks:
- Training/testing batches and losses
- Latent values, gradients, and optimization steps  
- Model predictions and hidden states
- Phase transitions with `log_phase()`

**Plotting**: `plot_logger_panels()` creates multi-panel figures with configurable panel orders.

### Model States and Phases

- **Phase tracking**: Logger records training phases for analysis
- **State management**: Models handle batch/sequence dimensions dynamically
- **Curriculum learning**: Interleaved vs blocked training with `shuffle_or_interleave`

## Running Experiments

### Single Experiments
```bash
python run_main_fig.py  # Main figures
python run_main_additive_behavioral_dynamics.py  # Additive gating variants
```

### Array Jobs
```bash 
./submit_adapt_job.sh 100 generalization_tests  # Parameter sweeps
./submit_adapt_job.sh 50 csw  # CSW task variants
```

### Key Flags
- Set `baseline_rnn = True` and `LU_lr = 0.0` to disable latent updates
- Use `config.env_seed` for reproducible random number generation
- Enable `config.log_hidden_states = True` for neural trajectory analysis

## Analysis Workflow

1. **Data collection**: Array jobs save `logger_train.npy` and `logger_test.npy`
2. **Aggregation**: `get_matching_loggers()` loads results by parameter values
3. **Metrics**: `get_accuracy()`, `get_accuracies_averaged_across_time()` for performance analysis
4. **Visualization**: Panel-based plotting with phase annotations and switch detection

This architecture enables systematic exploration of neural gating mechanisms across multiple cognitive tasks with rigorous experimental control.

# General instructions. 
Do not worry about lint errors. Try to aim for good code practice and typing but no need to correct all typing issues.# Use modern python where possible (e.g. f-strings, pathlib, etc)
