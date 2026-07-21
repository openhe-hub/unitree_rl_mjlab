# Reproducing the Unitree H2 Flat-Terrain Locomotion Policy

This documents the exact steps used to train and verify the first `Unitree-H2-Flat`
velocity-tracking policy (2026-07-21, on the NYUAD Jubail HPC cluster). Follow it to
reproduce the result on any Linux machine with an NVIDIA GPU, or on a Slurm cluster.

## 1. Result summary

| Item | Value |
|---|---|
| Task | `Unitree-H2-Flat` (velocity tracking, plane terrain) |
| Hardware | 1× NVIDIA A100 (40 GB), 16 CPU cores, 100 GB RAM |
| Scale | 4096 envs, 10001 iterations, ~55k FPS |
| Wall time | 4 h 58 m |
| Final metrics | mean reward ≈ 25, episode length ≈ 1000 (cap), fell-over ≈ 0.04/episode |
| Artifacts | `model_10000.pt` (checkpoints every 100 iters) + `policy.onnx` (deploy-ready) |

Note on the reward curve: mean reward peaks around ~37 near iteration 5000 and then
settles to ~25 while episode length stays at the cap. This is not a regression — the
action std anneals (0.98 → 0.46) and the reward mix rebalances as the policy shifts
from exploration to exploitation.

## 2. Environment setup

Python 3.11 virtualenv (uv shown; conda works the same way):

```bash
cd unitree_rl_mjlab
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .
```

`setup.py` pins the versions that matter. These pins exist because the upstream
packages leave them unbounded and the latest versions break:

| Pin | Why |
|---|---|
| `mujoco>=3.5,<3.6` | mujoco ≥3.6 removes `mjENBL_MULTICCD`, which mujoco-warp 3.5.0 still references |
| `warp-lang==1.12.1` | warp ≥1.13 hides `wp.context`, which mjlab 1.2.0 `sim.py` accesses |
| `scipy` | used by `mjlab.terrains` but not declared by mjlab |

Sanity check (CPU is fine, no GPU needed):

```bash
python -c "import src.tasks; from mjlab.tasks.registry import list_tasks; \
print([t for t in list_tasks() if 'H2' in t])"
# expect: ['Unitree-H2-Flat', 'Unitree-H2-Rough']
```

No dataset/asset download is required: H2 meshes are committed under
`src/assets/robots/unitree_h2/xmls/assets/`, and velocity tasks use no motion files.

### wandb (optional but recommended)

Training defaults to `--agent.logger=wandb`. Provide the key via environment
variable (do not commit it): `export WANDB_API_KEY=...`. Use
`--agent.logger=tensorboard` to run without wandb.

## 3. Training

### 3.1 Direct (any machine with an NVIDIA GPU)

```bash
# Smoke test first (~3 min): validates CUDA/EGL/warp end-to-end
python scripts/train.py Unitree-H2-Flat \
  --env.scene.num-envs=256 --agent.max-iterations=30 --agent.logger=tensorboard

# Full run (~5 h on A100)
python scripts/train.py Unitree-H2-Flat \
  --env.scene.num-envs=4096 --agent.wandb-project=unitree_rl_mjlab
```

Notes:
- `train.py` forces `MUJOCO_GL=egl` (headless GPU rendering); a working NVIDIA
  driver is all you need, no display.
- Boolean flags need explicit values with this repo's tyro config: `--video=True`,
  not `--video`.

### 3.2 Slurm (as used on Jubail)

`scripts/slurm/train_h2_flat.sbatch` wraps the above (partition `nvidia`,
`gpu:a100:1`, mail notifications). From the repo root on the cluster:

```bash
mkdir -p logs/slurm
# smoke test
sbatch --time=00:30:00 --export=ALL,NUM_ENVS=256,MAX_ITERS=30,LOGGER=tensorboard \
  scripts/slurm/train_h2_flat.sbatch
# full run
sbatch scripts/slurm/train_h2_flat.sbatch
```

The script sources `WANDB_API_KEY` from a private env file (see the script header)
when `LOGGER=wandb`. Queue tip: short jobs often start much earlier than Slurm's
estimate via backfill; if the a100 queue is crowded, `--gres=gpu:h100:1` frequently
starts immediately.

Outputs land in `logs/rsl_rl/h2_velocity/<timestamp>/`: `model_<iter>.pt` every
100 iterations, `policy.onnx` (re-exported at each save), tensorboard events.

## 4. Verification (headless play + video)

`scripts/play.py` records video but then blocks in an interactive viewer, so the
Slurm wrapper `scripts/slurm/play_h2_flat.sbatch` lets it run under `timeout` and
collects the clip (the viser viewer auto-plays without a client):

```bash
sbatch --gres=gpu:h100:1 \
  --export=ALL,CHECKPOINT=logs/rsl_rl/h2_velocity/<timestamp>/model_10000.pt \
  scripts/slurm/play_h2_flat.sbatch
```

Optional overrides: `VIDEO_LENGTH` (default 500 steps ≈ 10 s), `VIDEO_HEIGHT`/
`VIDEO_WIDTH` (default 1920×1080 — the mjlab default of 320×240 is too blurry to
judge gait), `NUM_ENVS` (default 1).

The clip appears in `logs/rsl_rl/h2_velocity/<timestamp>/videos/play/`. In play
mode commands sample from lin_vel_x ∈ (-0.5, 1.0), lin_vel_y ∈ (±0.5),
ang_vel_z ∈ (±0.5); expect steady omnidirectional walking with no falls.

## 5. Known issues

- **wandb sync can stall mid-run** (observed once around iteration 5000): training
  and local tensorboard logging are unaffected; curves just stop updating on the
  web. Verify real progress via checkpoint mtimes in the log dir.
- **Slurm stdout is block-buffered**: the `.out` file can lag many minutes behind;
  it is not a sign the job hung.

---

# Reproducing the Unitree H2 Motion-Tracking Policy (BeyondMimic)

Follow-up to the locomotion policy above (2026-07-21/22). Adds H2 support to the
`src/tasks/tracking/` module (BeyondMimic re-implementation, previously G1-only)
and produces H2 reference motions by retargeting LAFAN1 with
[GMR](https://github.com/YanjieZe/GMR), which does not ship H2 support.

## 6. Pipeline overview

```
LAFAN1 bvh ──GMR (jubail, CPU)──> pkl ──batch_gmr_pkl_to_csv──> csv (in git)
    csv ──scripts/csv_to_npz.py --robot h2 (GPU)──> npz ──train.py──> policy
```

- Task registration: `src/tasks/tracking/config/h2/` (auto-discovered; foot end
  body is `*_ankle_pitch_link` — H2's ankle chain is roll→pitch, reversed vs G1).
- GMR integration: `scripts/gmr_h2/setup_gmr_h2.sh` clones GMR at a pinned
  commit, installs the H2 mocap scene + IK config, and patches `params.py`.
  `retarget_bvh_headless.py` replaces GMR's viewer-bound script for cluster use.
- Slurm wrappers: `scripts/slurm/{gmr_retarget_h2,csv_to_npz_h2,train_h2_tracking,play_h2_tracking}.sbatch`.

## 7. IK config tuning history (bvh_lafan1_to_h2.json)

QC method: replay the csv through the H2 model (`csv_to_npz --render`), compare
root/waist pitch numerically against Unitree's official G1 retarget of the same
motion. Iterations (each ~2 min CPU on the `compute` partition):

| ver | change | result |
|-----|--------|--------|
| v1 | copy G1 config, scale legs 1.1 / arms 0.95, foot=ankle_pitch_link | pelvis sagged 11° fwd, waist +15° perm. bend, 7.4% frames >15° off |
| v2 | pelvis rot weight 10→50 (stage1), 5→30 (stage2) | bias −11→−3°, extremes remain |
| v3 | stage2 torso rot 10→50, shoulders 100→50 | pelvis tracks exactly; waist still +17° |
| v4 | torso offset ⊗ pitch(+18°) | wrong sign: waist saturation 72% |
| v5 | torso offset ⊗ pitch(−18°) | waist mean +0.1°, 0% frames >15° off — shipped |

Root causes: (a) pelvis rotation tracked too weakly for H2's mass layout;
(b) H2's `torso_link` target needs a constant −18° pitch offset relative to the
G1 convention. The v4/v5 sign flip is the cheap way to resolve the offset
direction empirically.

## 8. Training & verification

```bash
# full run (h100, ~30k iters; smoke-test variant: NUM_ENVS=256 MAX_ITERS=30 LOGGER=tensorboard)
sbatch --export=ALL,MOTION_FILE=src/assets/motions/h2/dance1_subject2.npz \
  scripts/slurm/train_h2_tracking.sbatch
# play + video
sbatch --export=ALL,CHECKPOINT=logs/rsl_rl/h2_tracking/<run>/model_<it>.pt,MOTION_FILE=src/assets/motions/h2/dance1_subject2.npz \
  scripts/slurm/play_h2_tracking.sbatch
```

## 9. Known issues (tracking-specific)

- **nefc overflow / SIGABRT**: H2 poses generate more simultaneous contacts than
  the G1-tuned defaults allow. Symptoms: `nefc overflow - please increase njmax`
  in training logs (constraints silently dropped), or `csv_to_npz --render`
  dying with SIGABRT. Fixed by raising `nconmax`/`njmax` in the H2 tracking cfg
  (60/400) and in `csv_to_npz.py` (100/500). If H2 assets change, expect to
  revisit these.
- **GMR quirks** (pinned commit bb1bbe4): `bvh_to_robot.py` hard-requires a
  display; `bvh_to_robot_dataset.py` has three stale-API bugs; the default
  `daqp` solver is not installed by `pip install -e .` — install `daqp`
  explicitly; `torch` is an undeclared import dependency (install CPU build).
- **CSV conventions**: root quaternion is **xyzw** (matches `csv_to_npz.py`);
  dof columns follow the robot's MJCF tree order — for H2 the ankle columns are
  roll-then-pitch, so use `--joint-order g1` only when replaying a G1 csv.
- **tyro `--line-range`**: pass as `--line-range <start> <end>` fails under the
  project's TYRO_FLAGS; segment rendering via sbatch `--wrap` needs care (or
  just render the full clip).
