#!/bin/bash

# Usage: ./submit_adapt_job.sh <MAX_TASK_ID> <EXPERIMENT_NAME>
# EXPERIMENT_NAME: generalization_tests or input_z_sweeps


if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <MAX_TASK_ID> <EXPERIMENT_NAME>"
    echo "EXPERIMENT_NAME: generalization_tests | input_z_sweeps | seq_learn_supp | seq_learn_interleaved_phase | seq_learn | time_scales"
    exit 1
fi

MAX_TASK_ID=$1
EXPERIMENT_NAME=$2
MAX_PARALLEL=100  # Change if you want a different % max concurrency

if [ "$EXPERIMENT_NAME" = "generalization_tests" ]; then
    PYTHON_FILE="cst_run_generalization.py"
elif [ "$EXPERIMENT_NAME" = "input_z_sweeps" ]; then
    PYTHON_FILE="adapt_run_array_input_z_sweeps.py"
elif [ "$EXPERIMENT_NAME" = "seq_learn_supp" ]; then
    PYTHON_FILE="seq_learn_supplementary_run.py"
elif [ "$EXPERIMENT_NAME" = "seq_learn_interleaved_phase" ]; then
    PYTHON_FILE="seq_learn_varying_interleaved_phase_run.py"
elif [ "$EXPERIMENT_NAME" = "seq_learn" ]; then
    PYTHON_FILE="seq_learn_run.py"
elif [ "$EXPERIMENT_NAME" = "time_scales" ]; then
    PYTHON_FILE="time_scales_nested_run.py"
else
    echo "Invalid experiment name: $EXPERIMENT_NAME"
    echo "Valid options: generalization_tests, input_z_sweeps, seq_learn_supp, seq_learn_interleaved_phase, seq_learn, time_scales"
    exit 1
fi

mkdir -p ./slurm

sbatch --array=0-$MAX_TASK_ID%$MAX_PARALLEL <<EOF
#!/bin/bash
#SBATCH --job-name=adapt
#SBATCH -n 1
#SBATCH --output=./slurm/slurm-%A_%a.out
#SBATCH --error=./slurm/slurm-%A_%a.err
#SBATCH --time=0-00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hummosa@live.com

# Activate env and run
# conda activate neo
python $PYTHON_FILE
EOF
echo "Submitted array jobs 0..$MAX_TASK_ID for '$EXPERIMENT_NAME' with max parallelism $MAX_PARALLEL."
