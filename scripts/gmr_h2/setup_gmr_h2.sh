#!/bin/bash
# Wire Unitree H2 into a GMR checkout (idempotent).
#
# Clones GMR (pinned commit), copies the H2 robot MJCF + mocap scene + IK
# config into it, symlinks the meshes, and registers "unitree_h2" in
# general_motion_retargeting/params.py.
#
# Usage (from unitree_rl_mjlab repo root on jubail):
#   bash scripts/gmr_h2/setup_gmr_h2.sh
# Env overrides: GMR_DIR (default /scratch/zl4487/zhewen/GMR), GMR_COMMIT.
#
# Afterwards create the GMR venv (separate from the main repo's .venv):
#   cd "${GMR_DIR}" && uv venv --python 3.10 .venv && source .venv/bin/activate
#   uv pip install -e . daqp torch --index-url https://download.pytorch.org/whl/cpu

set -euo pipefail

GMR_DIR=${GMR_DIR:-/scratch/zl4487/zhewen/GMR}
GMR_COMMIT=${GMR_COMMIT:-bb1bbe40774794fceb2a7c579a3464a28e68c844}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
H2_XML_DIR="${REPO_ROOT}/src/assets/robots/unitree_h2/xmls"

if [[ ! -d "${GMR_DIR}" ]]; then
  git clone https://github.com/YanjieZe/GMR "${GMR_DIR}"
fi
git -C "${GMR_DIR}" checkout "${GMR_COMMIT}"

# Robot assets: robot MJCF (copy), mocap scene wrapper, meshes (symlink).
mkdir -p "${GMR_DIR}/assets/unitree_h2"
cp "${H2_XML_DIR}/h2.xml" "${GMR_DIR}/assets/unitree_h2/h2.xml"
cp "${SCRIPT_DIR}/h2_mocap.xml" "${GMR_DIR}/assets/unitree_h2/h2_mocap.xml"
ln -sfn "${H2_XML_DIR}/assets" "${GMR_DIR}/assets/unitree_h2/assets"

# IK config.
cp "${SCRIPT_DIR}/bvh_lafan1_to_h2.json" \
  "${GMR_DIR}/general_motion_retargeting/ik_configs/bvh_lafan1_to_h2.json"

# Register the robot in params.py.
python3 - "${GMR_DIR}" <<'EOF'
import pathlib
import sys

params = pathlib.Path(sys.argv[1]) / "general_motion_retargeting" / "params.py"
text = params.read_text()
if "unitree_h2" in text:
    print("params.py already references unitree_h2; skipping")
    sys.exit()
text = text.replace(
    'ROBOT_XML_DICT = {\n',
    'ROBOT_XML_DICT = {\n'
    '    "unitree_h2": ASSET_ROOT / "unitree_h2" / "h2_mocap.xml",\n', 1)
text = text.replace(
    '"bvh_lafan1":{\n',
    '"bvh_lafan1":{\n'
    '        "unitree_h2": IK_CONFIG_ROOT / "bvh_lafan1_to_h2.json",\n', 1)
text = text.replace(
    'ROBOT_BASE_DICT = {\n',
    'ROBOT_BASE_DICT = {\n    "unitree_h2": "pelvis",\n', 1)
text = text.replace(
    'VIEWER_CAM_DISTANCE_DICT = {\n',
    'VIEWER_CAM_DISTANCE_DICT = {\n    "unitree_h2": 3.0,\n', 1)
expected = (
    '"unitree_h2": ASSET_ROOT',
    '"unitree_h2": IK_CONFIG_ROOT',
    '"unitree_h2": "pelvis"',
    '"unitree_h2": 3.0',
)
missing = [e for e in expected if e not in text]
if missing:
    sys.exit(f"params.py insertion failed for {missing} (upstream layout changed?)")
params.write_text(text)
print("params.py patched")
EOF

echo "GMR H2 setup complete at ${GMR_DIR} (commit ${GMR_COMMIT})"
