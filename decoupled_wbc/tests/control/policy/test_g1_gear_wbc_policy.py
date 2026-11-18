import numpy as np

from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy


def _policy(cmd, use_teleop_policy_cmd):
    """A policy with only the fields the teleop switch touches (no model files needed)."""
    policy = G1GearWbcPolicy.__new__(G1GearWbcPolicy)
    policy.config = {"cmd_init": np.zeros(3)}
    policy.cmd = np.array(cmd, dtype=float)
    policy.use_teleop_policy_cmd = use_teleop_policy_cmd
    return policy


def test_walk_stops_when_teleop_commands_stop():
    """The safety timeout disables teleop: the last teleop velocity must not stay in force."""
    policy = _policy([0.4, 0.0, 0.1], use_teleop_policy_cmd=True)
    policy.set_use_teleop_policy_cmd(False)
    assert not policy.use_teleop_policy_cmd
    np.testing.assert_array_equal(policy.cmd, np.zeros(3))


def test_keyboard_command_is_kept_when_teleop_was_not_in_use():
    """Goals without teleop commands arrive in keyboard mode too: they must not stop the walk."""
    policy = _policy([0.2, 0.0, 0.0], use_teleop_policy_cmd=False)
    policy.set_use_teleop_policy_cmd(False)
    np.testing.assert_array_equal(policy.cmd, [0.2, 0.0, 0.0])


def test_command_is_kept_when_teleop_starts():
    policy = _policy([0.2, 0.0, 0.0], use_teleop_policy_cmd=False)
    policy.set_use_teleop_policy_cmd(True)
    assert policy.use_teleop_policy_cmd
    np.testing.assert_array_equal(policy.cmd, [0.2, 0.0, 0.0])
