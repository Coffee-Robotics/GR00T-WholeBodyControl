from types import SimpleNamespace

import numpy as np
import pytest
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from decoupled_wbc.control.envs.g1.utils.command_sender import BodyCommandSender
from decoupled_wbc.control.envs.g1.utils.joint_offsets import get_joint_offsets
from decoupled_wbc.control.envs.g1.utils.state_processor import BodyStateProcessor
from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig

LEFT_ANKLE_PITCH, RIGHT_ANKLE_PITCH = 4, 10


@pytest.fixture(scope="module")
def channel():
    try:
        ChannelFactoryInitialize(0, "lo")
    except Exception:
        pass  # already initialized by another test module


@pytest.fixture
def wbc_config():
    return ControlLoopConfig(interface="sim").load_wbc_yaml()


def with_offsets(wbc_config, offsets):
    wbc_config["JOINT_OFFSETS"] = offsets
    return wbc_config


class Reads:
    """Stands in for a channel subscriber: Read() returns the same message every time."""

    def __init__(self, message):
        self.message = message

    def Read(self):
        return self.message


def measured_positions(processor, encoder_readings):
    """Run the state processor on a low state holding the given encoder readings."""
    imu = SimpleNamespace(quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=[0.0] * 3, accelerometer=[0.0] * 3)
    motors = [SimpleNamespace(q=float(q), dq=0.0, tau_est=0.0) for q in encoder_readings]
    processor.robot_lowstate_subscriber = Reads(SimpleNamespace(imu_state=imu, motor_state=motors))
    processor.secondary_imu_subscriber = Reads(imu)
    processor.odo_state_subscriber = Reads(SimpleNamespace(position=[0.0] * 3, linear_velocity=[0.0] * 3))
    num_joints = processor.num_dof
    return processor._prepare_low_state()[0, 7 : 7 + num_joints]


def test_no_offsets_by_default(wbc_config):
    np.testing.assert_array_equal(get_joint_offsets(wbc_config), np.zeros(wbc_config["NUM_JOINTS"]))
    del wbc_config["JOINT_OFFSETS"]
    np.testing.assert_array_equal(get_joint_offsets(wbc_config), np.zeros(wbc_config["NUM_JOINTS"]))


@pytest.mark.parametrize("offsets", [[0.0] * 28, [0.0] * 30, [float("nan")] + [0.0] * 28, 0.05])
def test_offsets_must_be_one_finite_number_per_joint(wbc_config, offsets):
    with pytest.raises(ValueError, match="JOINT_OFFSETS"):
        get_joint_offsets(with_offsets(wbc_config, offsets))


def test_state_processor_adds_the_offsets_to_the_measured_positions(channel, wbc_config):
    offsets = np.zeros(wbc_config["NUM_JOINTS"])
    offsets[[LEFT_ANKLE_PITCH, RIGHT_ANKLE_PITCH]] = 0.05
    encoders = np.linspace(-0.5, 0.5, 35)  # the robot reports 35 motors, the body uses the first 29

    plain = measured_positions(BodyStateProcessor(config=dict(wbc_config)), encoders)
    offset = measured_positions(BodyStateProcessor(config=with_offsets(wbc_config, offsets)), encoders)

    np.testing.assert_allclose(plain, encoders[:29])
    np.testing.assert_allclose(offset, encoders[:29] + offsets)


def test_command_sender_subtracts_the_offsets_from_the_commanded_positions(channel, wbc_config):
    num_joints = wbc_config["NUM_JOINTS"]
    offsets = np.zeros(num_joints)
    offsets[[LEFT_ANKLE_PITCH, RIGHT_ANKLE_PITCH]] = 0.05
    commanded = np.linspace(-0.5, 0.5, num_joints)
    zeros = np.zeros(num_joints)

    sender = BodyCommandSender(config=with_offsets(wbc_config, offsets))
    sender.send_command(commanded, zeros, zeros)

    sent = np.array([sender.low_cmd.motor_cmd[i].q for i in range(num_joints)])
    np.testing.assert_allclose(sent, commanded - offsets)


def test_a_joint_held_at_its_measured_position_is_sent_its_own_encoder_reading(channel, wbc_config):
    """Reading and commanding undo each other, so holding still changes nothing at the motor."""
    offsets = np.linspace(-0.05, 0.05, wbc_config["NUM_JOINTS"])
    config = with_offsets(wbc_config, offsets)
    encoders = np.linspace(-0.5, 0.5, 35)

    measured = measured_positions(BodyStateProcessor(config=config), encoders)
    sender = BodyCommandSender(config=config)
    sender.send_command(measured, np.zeros(29), np.zeros(29))

    sent = np.array([sender.low_cmd.motor_cmd[i].q for i in range(29)])
    np.testing.assert_allclose(sent, encoders[:29])
