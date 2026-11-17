import copy

import pytest

from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig, override_wbc_config


def test_real_robot_uses_the_configured_motor_gains():
    """The real robot gets the gains in the YAML file, the same as the simulator."""
    sim = ControlLoopConfig(interface="sim").load_wbc_yaml()
    real = ControlLoopConfig(interface="real").load_wbc_yaml()
    assert real["MOTOR_KP"] == sim["MOTOR_KP"]
    assert real["MOTOR_KD"] == sim["MOTOR_KD"]


@pytest.mark.parametrize("interface", ["sim", "real"])
def test_no_motor_gain_is_negative(interface):
    wbc_config = ControlLoopConfig(interface=interface).load_wbc_yaml()
    assert min(wbc_config["MOTOR_KP"]) >= 0
    assert min(wbc_config["MOTOR_KD"]) >= 0


@pytest.mark.parametrize("key", ["MOTOR_KP", "MOTOR_KD"])
def test_negative_motor_gain_is_refused(key):
    config = ControlLoopConfig(interface="sim")
    wbc_config = copy.deepcopy(config.load_wbc_yaml())
    wbc_config[key][14] = -5
    with pytest.raises(ValueError, match=key):
        override_wbc_config(wbc_config, config)
