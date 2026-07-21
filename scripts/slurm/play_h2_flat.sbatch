#!/bin/bash
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=h2-play
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu

# Headless play + video recording for a trained H2-Flat policy.
# The viser viewer auto-plays without a client; VideoRecorder saves the clip
# after VIDEO_LENGTH steps, then `timeout` reaps the never-exiting viewer.
#
# Usage (from repo root on jubail):
#   sbatch --export=ALL,CHECKPOINT=logs/rsl_rl/h2_velocity/<run>/model_10000.pt scripts/slurm/play_h2_flat.sbatch
# Video lands in: <run dir>/videos/play/

set -uo pipefail

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT=path/to/model_xxx.pt}
VIDEO_LENGTH=${VIDEO_LENGTH:-500}
NUM_ENVS=${NUM_ENVS:-1}

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export MUJOCO_GL=egl

echo "=== node: $(hostname), checkpoint: ${CHECKPOINT} ==="

timeout 20m python scripts/play.py Unitree-H2-Flat \
  --checkpoint_file="${CHECKPOINT}" \
  --num_envs="${NUM_ENVS}" \
  --video=True \
  --video_length="${VIDEO_LENGTH}" \
  --viewer=viser
rc=$?

echo "=== play exited with ${rc} (124 = reaped by timeout, expected) ==="
ls -la "$(dirname "${CHECKPOINT}")/videos/play/" || true
