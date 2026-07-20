#!/bin/bash
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=h2-flat
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu

# Usage (from repo root on jubail):
#   mkdir -p logs/slurm
#   Smoke test: sbatch --time=00:30:00 --export=ALL,NUM_ENVS=256,MAX_ITERS=30,LOGGER=tensorboard scripts/slurm/train_h2_flat.sbatch
#   Full run:   sbatch scripts/slurm/train_h2_flat.sbatch

set -euo pipefail

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERS=${MAX_ITERS:-10001}
LOGGER=${LOGGER:-wandb}

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

WANDB_ENV=/scratch/zl4487/zhewen/.secrets/wandb.env
if [[ "${LOGGER}" == "wandb" ]]; then
  source "${WANDB_ENV}"
fi

echo "=== node: $(hostname) ==="
nvidia-smi

python scripts/train.py Unitree-H2-Flat \
  --env.scene.num-envs="${NUM_ENVS}" \
  --agent.max-iterations="${MAX_ITERS}" \
  --agent.logger="${LOGGER}" \
  --agent.wandb-project=unitree_rl_mjlab
