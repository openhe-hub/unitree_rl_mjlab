#!/bin/bash
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=h2-track-play
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu

# Headless play + video recording for a trained H2 tracking policy.
# The viser viewer auto-plays without a client; VideoRecorder saves the clip
# after VIDEO_LENGTH steps, then `timeout` reaps the never-exiting viewer.
#
# Usage (from repo root on jubail):
#   sbatch --export=ALL,CHECKPOINT=logs/rsl_rl/h2_tracking/<run>/model_30000.pt,MOTION_FILE=src/assets/motions/h2/dance1_subject2.npz scripts/slurm/play_h2_tracking.sbatch
# Video lands in: <run dir>/videos/play/

set -uo pipefail

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT=path/to/model_xxx.pt}
MOTION_FILE=${MOTION_FILE:?set MOTION_FILE=path/to/motion.npz}
TASK=${TASK:-Unitree-H2-Tracking-No-State-Estimation}
VIDEO_LENGTH=${VIDEO_LENGTH:-500}
NUM_ENVS=${NUM_ENVS:-1}
VIDEO_HEIGHT=${VIDEO_HEIGHT:-1080}
VIDEO_WIDTH=${VIDEO_WIDTH:-1920}

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export MUJOCO_GL=egl

echo "=== node: $(hostname), checkpoint: ${CHECKPOINT} ==="

timeout 20m python scripts/play.py "${TASK}" \
  --checkpoint_file="${CHECKPOINT}" \
  --motion_file="${MOTION_FILE}" \
  --num_envs="${NUM_ENVS}" \
  --video=True \
  --video_length="${VIDEO_LENGTH}" \
  --video_height="${VIDEO_HEIGHT}" \
  --video_width="${VIDEO_WIDTH}" \
  --viewer=viser
rc=$?

echo "=== play exited with ${rc} (124 = reaped by timeout, expected) ==="
ls -la "$(dirname "${CHECKPOINT}")/videos/play/" || true
