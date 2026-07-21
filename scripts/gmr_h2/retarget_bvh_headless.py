"""Headless LAFAN1 BVH -> robot motion retargeting via GMR.

GMR's own scripts/bvh_to_robot.py unconditionally opens a MuJoCo viewer, which
fails on display-less cluster nodes; this driver runs the same pipeline without
a viewer and writes the pkl consumed by GMR's scripts/batch_gmr_pkl_to_csv.py.

Run inside the GMR virtualenv (the general_motion_retargeting package must be
installed); this script does not depend on unitree_rl_mjlab.
"""

import argparse
import os
import pickle

import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", required=True, type=str)
    parser.add_argument("--robot", default="unitree_h2", type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    parser.add_argument("--motion_fps", default=30, type=int)
    args = parser.parse_args()

    frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    qpos_list = [
        retargeter.retarget(frame) for frame in tqdm(frames, desc="Retargeting")
    ]
    qpos = np.stack(qpos_list)

    motion_data = {
        "fps": args.motion_fps,
        "root_pos": qpos[:, :3],
        # wxyz -> xyzw, matching GMR's bvh_to_robot.py pkl convention.
        "root_rot": qpos[:, [4, 5, 6, 3]],
        "dof_pos": qpos[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"Saved {qpos.shape[0]} frames to {args.save_path}")


if __name__ == "__main__":
    main()
