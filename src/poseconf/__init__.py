"""poseconf: split-conformal pose and keypoint sets on top of the frozen P1 pipeline.

`poseconf.p1_adapter` is the only module that imports `speedpose`. `poseconf.conformal` is pure
numpy/scipy (no torch) so it ports to the ship-deck project.
"""

__version__ = "0.1.0"
