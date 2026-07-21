#!/bin/bash
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=h2-csv2npz
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu

# Convert a motion CSV to npz by replaying it through the H2 model.
# With RENDER=True also writes <npz>_replay.mp4 next to the npz for QC.
#
# Usage (from repo root on jubail):
#   Smoke test (borrow the G1 csv, lift root by 0.25 m):
#     sbatch --export=ALL,INPUT=src/assets/motions/g1/dance1_subject2.csv,OUTPUT=dance1_subject2_smoke,JOINT_ORDER=g1,ROOT_Z_OFFSET=0.25,RENDER=True scripts/slurm/csv_to_npz_h2.sbatch
#   Real H2 csv from GMR:
#     sbatch --export=ALL,INPUT=src/assets/motions/h2/dance1_subject2.csv,OUTPUT=dance1_subject2,RENDER=True scripts/slurm/csv_to_npz_h2.sbatch
# npz lands in src/assets/motions/h2/<OUTPUT>.npz

set -euo pipefail

INPUT=${INPUT:?set INPUT=path/to/motion.csv}
OUTPUT=${OUTPUT:?set OUTPUT=name_without_ext}
JOINT_ORDER=${JOINT_ORDER:-}
ROOT_Z_OFFSET=${ROOT_Z_OFFSET:-0.0}
RENDER=${RENDER:-False}

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export MUJOCO_GL=egl

EXTRA_ARGS=()
if [[ -n "${JOINT_ORDER}" ]]; then
  EXTRA_ARGS+=(--joint-order="${JOINT_ORDER}")
fi

echo "=== node: $(hostname), input: ${INPUT} ==="

python scripts/csv_to_npz.py \
  --robot=h2 \
  --input-file="${INPUT}" \
  --output-name="${OUTPUT}" \
  --root-z-offset="${ROOT_Z_OFFSET}" \
  --render="${RENDER}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

ls -la src/assets/motions/h2/
