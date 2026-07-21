#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=gmr-h2
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu

# LAFAN1 BVH -> H2 pkl -> BeyondMimic-compatible CSV via GMR (headless, CPU).
#
# Prereq (once): bash scripts/gmr_h2/setup_gmr_h2.sh   # clone GMR + wire in H2
#   then create the GMR venv, see the header of that script.
#
# Usage (from unitree_rl_mjlab repo root on jubail):
#   sbatch --export=ALL,BVH=/scratch/zl4487/zhewen/datasets/lafan1/dance1_subject2.bvh scripts/slurm/gmr_retarget_h2.sbatch
# Output: ${OUT_DIR}/<name>.pkl and ${OUT_DIR}/csv/<name>.csv
# Then copy the csv into src/assets/motions/h2/ and commit.

set -euo pipefail

GMR_DIR=${GMR_DIR:-/scratch/zl4487/zhewen/GMR}
BVH=${BVH:?set BVH=path/to/motion.bvh}
OUT_DIR=${OUT_DIR:-/scratch/zl4487/zhewen/gmr_out/h2}
NAME=$(basename "${BVH}" .bvh)

cd "${SLURM_SUBMIT_DIR}"
source "${GMR_DIR}/.venv/bin/activate"
mkdir -p "${OUT_DIR}"

echo "=== node: $(hostname), bvh: ${BVH} ==="

python scripts/gmr_h2/retarget_bvh_headless.py \
  --bvh_file "${BVH}" \
  --robot unitree_h2 \
  --save_path "${OUT_DIR}/${NAME}.pkl"

python "${GMR_DIR}/scripts/batch_gmr_pkl_to_csv.py" --folder "${OUT_DIR}"

echo "=== done ==="
ls -la "${OUT_DIR}" "${OUT_DIR}/csv"
