import numpy as np
import pytest

from decoupled_wbc.control.envs.g1.utils.joint_safety import JointSafetyMonitor
from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

ARM_JOINT = "left_shoulder_pitch_joint"
LEG_JOINT = "left_knee_joint"


@pytest.fixture(scope="module")
def robot_model():
    return instantiate_g1_robot_model()


def make_obs(robot_model, arm_position=0.3, arm_velocity=0.0):
    """An observation with every arm joint at one position and one arm joint moving."""
    q = np.zeros(robot_model.num_dofs)
    q[robot_model.get_joint_group_indices("arms")] = arm_position
    dq = np.zeros(robot_model.num_dofs)
    dq[robot_model.dof_index(ARM_JOINT)] = arm_velocity
    return {"q": q, "dq": dq}


def make_action(robot_model, arm_target=1.0, leg_target=0.2):
    q = np.zeros(robot_model.num_dofs)
    q[robot_model.get_joint_group_indices("arms")] = arm_target
    q[robot_model.dof_index(LEG_JOINT)] = leg_target
    return {"q": q}


def step(monitor, obs, action):
    """handle_violations() changes the action in place, so give it a copy."""
    return monitor.handle_violations(obs, {"q": action["q"].copy()})


def finish_startup_ramp(monitor, obs, action):
    for _ in range(monitor.ramp_duration_steps + 1):
        step(monitor, obs, action)
    assert monitor.startup_complete


def test_kill_mode_is_the_default_and_shuts_down_on_the_real_robot(robot_model):
    monitor = JointSafetyMonitor(robot_model, env_type="real")
    assert monitor.safety_mode == "kill"
    too_fast = make_obs(robot_model, arm_velocity=JointSafetyMonitor.ARM_VELOCITY_LIMIT + 1.0)
    with pytest.raises(SystemExit):
        step(monitor, too_fast, make_action(robot_model))


def test_kill_mode_only_reports_in_simulation(robot_model):
    monitor = JointSafetyMonitor(robot_model, env_type="sim")
    too_fast = make_obs(robot_model, arm_velocity=JointSafetyMonitor.ARM_VELOCITY_LIMIT + 1.0)
    result = step(monitor, too_fast, make_action(robot_model))
    assert result["shutdown_required"] and not result["safe_to_continue"]


def test_invalid_safety_mode_is_refused(robot_model):
    with pytest.raises(ValueError, match="safety_mode"):
        JointSafetyMonitor(robot_model, safety_mode="freeze")


@pytest.mark.parametrize("env_type", ["sim", "real"])
def test_hold_mode_holds_the_arms_and_leaves_the_rest_alone(robot_model, env_type):
    monitor = JointSafetyMonitor(robot_model, env_type=env_type, safety_mode="hold")
    arms = robot_model.get_joint_group_indices("arms")
    leg = robot_model.dof_index(LEG_JOINT)
    action = make_action(robot_model, arm_target=1.0, leg_target=0.2)
    finish_startup_ramp(monitor, make_obs(robot_model), action)

    too_fast = make_obs(robot_model, arm_position=0.3, arm_velocity=7.0)
    result = step(monitor, too_fast, action)  # no SystemExit, on either env type

    assert not result["shutdown_required"]
    assert not result["safe_to_continue"]
    np.testing.assert_allclose(result["action"]["q"][arms], 0.3)  # held where they are
    assert result["action"]["q"][leg] == pytest.approx(0.2)  # the lower body is untouched


def test_hold_lasts_for_the_hold_duration_and_then_ramps_back(robot_model):
    monitor = JointSafetyMonitor(robot_model, safety_mode="hold")
    arms = robot_model.get_joint_group_indices("arms")
    action = make_action(robot_model, arm_target=1.0)
    finish_startup_ramp(monitor, make_obs(robot_model), action)
    step(monitor, make_obs(robot_model, arm_position=0.3, arm_velocity=7.0), action)

    # The arms have moved on, but the targets stay at the positions the hold started from
    calm = make_obs(robot_model, arm_position=0.35)
    for _ in range(monitor.hold_duration_steps - 1):
        result = step(monitor, calm, action)
        assert not result["safe_to_continue"]
        np.testing.assert_allclose(result["action"]["q"][arms], 0.3)

    # Then they are ramped back to the commanded pose, starting from the held positions
    targets = []
    for _ in range(monitor.ramp_duration_steps + 1):
        result = step(monitor, calm, action)
        assert result["safe_to_continue"]
        targets.append(result["action"]["q"][arms[0]])
    assert targets[0] == pytest.approx(0.3)
    assert all(later >= earlier for earlier, later in zip(targets, targets[1:]))
    assert max(np.diff(targets)) <= (1.0 - 0.3) / monitor.ramp_duration_steps + 1e-9
    assert targets[-1] == pytest.approx(1.0)


def test_hold_is_extended_while_the_violation_lasts(robot_model):
    monitor = JointSafetyMonitor(robot_model, safety_mode="hold")
    arms = robot_model.get_joint_group_indices("arms")
    action = make_action(robot_model, arm_target=1.0)
    finish_startup_ramp(monitor, make_obs(robot_model), action)

    for position in (0.3, 0.4, 0.5):  # still too fast, and moving
        result = step(monitor, make_obs(robot_model, arm_position=position, arm_velocity=7.0), action)
        np.testing.assert_allclose(result["action"]["q"][arms], 0.3)  # the first positions

    calm = make_obs(robot_model, arm_position=0.5)
    for _ in range(monitor.hold_duration_steps - 1):
        assert not step(monitor, calm, action)["safe_to_continue"]
    assert step(monitor, calm, action)["safe_to_continue"]


def test_hold_mode_changes_nothing_without_a_violation(robot_model):
    kill = JointSafetyMonitor(robot_model, env_type="sim")
    hold = JointSafetyMonitor(robot_model, env_type="sim", safety_mode="hold")
    obs, action = make_obs(robot_model), make_action(robot_model)
    for _ in range(kill.ramp_duration_steps + 10):
        expected, actual = step(kill, obs, action), step(hold, obs, action)
        np.testing.assert_allclose(actual["action"]["q"], expected["action"]["q"])
        assert actual["safe_to_continue"] and not actual["shutdown_required"]


def test_safety_mode_is_set_from_the_config():
    assert ControlLoopConfig(interface="sim").load_wbc_yaml()["joint_safety_mode"] == "kill"
    config = ControlLoopConfig(interface="sim", joint_safety_mode="hold")
    assert config.load_wbc_yaml()["joint_safety_mode"] == "hold"
