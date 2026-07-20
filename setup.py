"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
    # mjlab/mujoco-warp leave these unbounded; newer versions break:
    # mujoco>=3.6 removes mjENBL_MULTICCD (mujoco-warp 3.5.0 needs it),
    # warp-lang>=1.13 hides wp.context (mjlab 1.2.0 sim.py uses it),
    # scipy is used by mjlab.terrains but not declared.
    "mujoco>=3.5,<3.6",
    "warp-lang==1.12.1",
    "scipy",
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
