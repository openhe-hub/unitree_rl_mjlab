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
