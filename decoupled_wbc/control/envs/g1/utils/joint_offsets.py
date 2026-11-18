"""Joint position offsets for the G1 body.

A joint's encoder can read differently from where the joint really is, because of a zero
calibration error or because a linkage or a sole gives under load. JOINT_OFFSETS in the WBC
config holds one offset per joint, in radians and in joint order: the joint is at its encoder
reading plus its offset.

The state processor adds the offsets to the measured joint positions and the command sender
subtracts them from the commanded ones, so the rest of the control stack works with where the
joints really are. Without JOINT_OFFSETS in the config there are no offsets.
"""

from typing import Any, Dict

import numpy as np


def get_joint_offsets(config: Dict[str, Any]) -> np.ndarray:
    """Read the joint position offsets from the WBC config.

    Args:
        config: The WBC configuration dictionary

    Returns:
        One offset per joint in radians, in joint order (all zeros if none are configured)

    Raises:
        ValueError: If JOINT_OFFSETS does not hold one finite number per joint
    """
    num_joints = config["NUM_JOINTS"]
    offsets = np.asarray(config.get("JOINT_OFFSETS", np.zeros(num_joints)), dtype=float)
    if offsets.shape != (num_joints,):
        raise ValueError(
            f"JOINT_OFFSETS needs {num_joints} values, one per joint, but has shape {offsets.shape}"
        )
    if not np.all(np.isfinite(offsets)):
        raise ValueError("JOINT_OFFSETS must hold finite numbers")
    return offsets
