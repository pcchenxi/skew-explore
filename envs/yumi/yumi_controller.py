import os

import numpy as np
from scipy.linalg import pinv2
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from enum import Enum
from mujoco_py import functions
import time, math

def body_index(model, body_name):
    return model.body_names.index(body_name)


def body_pos(model, body_name):
    ind = body_index(model, body_name)
    return model.body_pos[ind]


def body_quat(model, body_name):
    ind = body_index(model, body_name)
    return model.body_quat[ind]


def body_frame(env, body_name):
    """
    Returns the rotation matrix to convert to the frame of the named body
    """
    ind = body_index(env.model, body_name)
    b = env.data.body_xpos[ind]
    q = env.data.body_xquat[ind]
    qr, qi, qj, qk = q
    s = np.square(q).sum()
    R = np.array([
        [1 - 2 * s * (qj ** 2 + qk ** 2), 2 * s * (qi * qj - qk * qr), 2 * s * (qi * qk + qj * qr)],
        [2 * s * (qi * qj + qk * qr), 1 - 2 * s * (qi ** 2 + qk ** 2), 2 * s * (qj * qk - qi * qr)],
        [2 * s * (qi * qk - qj * qr), 2 * s * (qj * qk + qi * qr), 1 - 2 * s * (qi ** 2 + qj ** 2)]
    ])
    return R

def quaternion_difference(q1, q2):
    """Returns quaternion difference that satisfies ``q_diff * q1 = q2``.

    Args:
        q1 (list): quaternion [w, x, y, z]
        q2 (list): quaternion [w, x, y, z]
    Returns:
        Quaternion difference [w, x, y, z], which satisfies ``q_diff * q1 = q2``
    """
    q1_abs = np.ndarray(4)
    q1_con = np.ndarray(4)
    q1_inv = np.ndarray(4)

    q1_con[0] = q1[0]
    q1_con[1] = -q1[1]
    q1_con[2] = -q1[2]
    q1_con[3] = -q1[3]

    functions.mju_mulQuat(q1_abs, q1, q1_con)
    q1_abs[0] += q1_abs[1] + q1_abs[2] + q1_abs[3]
    q1_inv = q1_con / q1_abs[0]

    q_diff = np.ndarray(4)
    functions.mju_mulQuat(q_diff, q2, q1_inv)

    return q_diff

class Mode(Enum):
    GRAVITY_COMPENSATION = 0
    JOINT_IMP_CTRL = 1
    CART_IMP_CTRL = 2
    CART_VEL_CTRL = 3
    MOCAP_IMP_CTRL = 4


class Controller():
    """Controller for the simulated KUKA LWR 4+.

    This class is used internally and the public methods are NOT provided to the
    user normally (e.g. via ROS).

    There are three control modes: gravity compensation, joint angle control
    (joint space) and cartesian space control (workspace). Control parameters
    are evaluated differently based on what is the control mode.

    For the joint controller, the ``target_pos`` is the 7 joint angles. There
    are 7 PD parameters, which are applied directly. In case of cartesian
    controller, the ``target_pos`` is evaluated as [x, y, z, q1, q2, q3, q4],
    where {x, y, z} is the cartesian position and {q1, q2, q3, q4} is the
    quaternion for tool orientation.

    For PD controller, the quaternion "difference" is calculated as angular
    velocity (r, p ,y). The PD parameters are evaluated as [x, y, z, r, p, y].
    The 7th value is silently ignored. The torques are evaluated similarly.
    """

    def __init__(self, sim, mode = 'MOCAP_IMP_CTRL'):
        self.sim = sim
        self.target_pos = None
        self.prior_error = [0, 0, 0, 0, 0, 0, 0]

        # do we use acceleration term M(q)*qacc in model dynamics equation?
        self.acc_term = False

        # PD-controller parameters for joint angle and cartesian pos control.
        # There are 7 joints, and cartesian position is measured in 3-D.
        # Task specific, default None
        self.kp = None
        self.kd = None
        self.previous = None
        self.torque = np.zeros(sim.model.nv)

        self.end_effector = "gripper_r_finger_r"
        self.arm_index = [
            self.sim.model.get_joint_qpos_addr("yumi_joint_" + str(i+1) + '_r')
            for i in range(7)
        ]
        self.init_controller(mode)
        self.init_ee_quat = self.tool_quat.copy()
        self.init_ee_pos = self.tool_pos.copy()

    def set_mode(self, mode):
        """Update controller mode"""
        self.mode = mode

    def init_controller(self, mode, p=None, d=None, torque=None):
        """Initialize the controller with given parameters.

        `p` and `d` and the proportional gain and derivative gain, respectively.
        The torque is summed to the output of the controller, allowing to control
        the robot in situations where e.g. a constant torque should be applied.

        If these values are not set, default ones will be used.
        """
        self.set_mode(Mode[mode])

        if p is not None:
            self.kp = p
        elif Mode[mode] == Mode.JOINT_IMP_CTRL:
            self.kp = [32000, 32000, 32000, 32000, 32000, 32000, 32000]
        elif Mode[mode] == Mode.CART_IMP_CTRL:
            self.kp = [2000, 2000, 2000, 20, 20, 20, None]
            # self.kp = [0.0, 0.00, 0.00, 0.00, 0.00, 0.00, None]
            #self.kp = [2, 2, 0.9, 0.2, 0.2, 0.2] + [None]
            #self.kp = [0.2, 0.00, 0.00, 0.00, 0.00, 0.00, None]

        if d is not None:
            self.kd = d
        elif Mode[mode] == Mode.JOINT_IMP_CTRL:
            self.kd = [15, 15, 15, 15, 15, 15, 15]
        elif Mode[mode] == Mode.CART_IMP_CTRL:
            self.kd = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, None]
            #self.kd = [0.005]*6 + [None]

        if torque is not None:
            self.torque = torque
        else:
            self.torque = np.zeros(7)

    def apply_control(self):
        """Returns torque values for all joints

        The actual calculations of torque is done in the other functions, which
        are used here depending on the controller mode.

        Returns:
            A list of torque values, for each joint [A1, A2, E1, A3, A4, A5, A6]
        """
        pd_params_ok = (self.kp is not None and self.kd is not None)

        if self.mode == Mode.JOINT_IMP_CTRL and pd_params_ok:
            return self._joint_angle_control()
        elif self.mode == Mode.CART_IMP_CTRL and pd_params_ok and self.end_effector is not None:
            return self._cart_position_control()
        elif self.mode == Mode.CART_VEL_CTRL and self.end_effector is not None:
            return self._cart_velocity_control()
        elif self.mode == Mode.MOCAP_IMP_CTRL:
            "in mocap mode, the control signals are zero."
            return np.zeros(9)
        else:
            return self._model_dynamics()

    def _model_dynamics(self):
        """
        qfrc_bias is mr_rne with acceleration set to 0, however we also want
        M(q)*qacc in the dynamics this should be the same as model dynamics
        f(qpos,qvel,qacc) in FRI paper more info on mj_rne and general
        framework: http://www.mujoco.org/book/computation.html#General

        Setting acc to false disables the acceleration term.
        """
        if self.acc_term:
            rne = np.ndarray(self.sim.model.nv)
            functions.mj_rne(self.sim.model, self.sim.data, True, rne)
            return rne[self.arm_index]
        else:
            return self.sim.data.qfrc_bias[self.arm_index]  # stored, no need for computation

    def _pd_control(self, error):
        """PD controller

        Returns summed proportional and derivative part. The proportional part
        is calculated by multiplying P-parameter (self.kp) with the given error.
        The derivative is calculated by finding the difference between current
        and previous error, and dividing by timestep. This value is multiplied
        by D-parameter (self.kd).
        """

        if error.shape == (6,):
            derivative = (error - self.prior_error[:6]) / self.sim.model.opt.timestep
            pd_output = self.kp[:6] * error + self.kd[:6] * derivative
        else:
            derivative = (error - self.prior_error) / self.sim.model.opt.timestep
            pd_output = self.kp * error + self.kd * derivative

        self.prior_error = error
        return pd_output

    def _joint_angle_control(self):
        """Joint angle controller

        The error between desired and current joint angles is pushed through PD
        controller and specified torques (if any) added as well.
        """

        error = self.target_pos - self.robot_arm_pos
        return self._pd_control(error) + self.torque        

    def _cart_velocity_control(self):
        print('in vel control', self.target_pos)
        eef_vel = np.array([self.target_pos[0], self.target_pos[1], self.target_pos[2], 0., 0., 0.])
   
        jac_pos = self.sim.data.get_body_jacp(self.end_effector)
        jac_pos = jac_pos.reshape(3, self.sim.model.nv)
        jac_pos = jac_pos[:, 0:7]

        # get position jacobian of eef
        jac_rot = self.sim.data.get_body_jacr(self.end_effector)
        jac_rot = jac_rot.reshape(3, self.sim.model.nv)
        jac_rot = jac_rot[:, 0:7]

        jac_full = np.concatenate((jac_pos, jac_rot))

        jac_inv = pinv2(jac_full)
        q_dot = np.dot(jac_inv, eef_vel)

        return q_dot

    def _cart_position_control(self):
        """Cartesian position and orientation controller

        The positional error is converted to joint torques as follows: the error
        in cartesian space is pushed through PD controller, and multiplied with
        positional Jacobian.

        The orientational error is converted to joint torques as follows: the
        difference (4,) between the desired and current quaternions is calculated
        and converted to angular velocity vector (3,). This vector is passed
        through PD controller and multiplied with rotational Jacobian.

        These torques are summed with model dynamics and applied to the robot.
        """
        # Target is given as [x,y,z] and quaternion.
        ref_xyz = np.array(self.target_pos[0:3])
        ref_pose = np.array(self.target_pos[3:7])

        # Calculate difference between current and target position+orientation
        xyz_diff = ref_xyz - self.tool_pos
        quat_diff = quaternion_difference(self.tool_quat, ref_pose)

        # print('current pos', self.tool_pos)

        # Convert orientation difference into angular velocities
        ang_vel = np.ndarray(3)
        functions.mju_quat2Vel(ang_vel, quat_diff, 1)  # timestep=1

        # Stack the errors and push them through PD controller
        error = np.hstack([xyz_diff, ang_vel])  # (6,)
        out = self._pd_control(error)

        # Compute required torques using positional and rotational Jacobians
        torques_cartesian = np.dot(self.jac_pos, out[:3] + self.torque[:3])
        torques_euler = np.dot(self.jac_rot, out[3:6] + self.torque[3:6])

        #return self._model_dynamics() + torques_cartesian + torques_euler
        #print(torques_cartesian)
        #print(torques_euler)
        
        return torques_cartesian + torques_euler


    @property
    def jac_rot(self):
        """Jacobian for rotational angles, shaped (3, 7)."""
        J = self.sim.data.get_body_jacr(self.end_effector)
        J = J.reshape(3, -1)[:, 0:7].T
        return J

    @property
    def jac_pos(self):
        """Jacobian for positional coordinates, shaped (3, 7)."""
        J = self.sim.data.get_body_jacp(self.end_effector)
        J = J.reshape(3, -1)[:, 0:7].T
        return J

    @property
    def tool_quat(self):
        """Quaternion, representing the tool orientation, shaped (4, )."""
        return self.sim.data.get_body_xquat(self.end_effector)

    @property
    def tool_pos(self):
        """Cartesian position of the tool, shaped (3, )."""
        return self.sim.data.get_body_xpos(self.end_effector)

    @property
    def robot_arm_pos(self):
        """Angular positions of the joints, [A1, A2, E1, A3, A4, A5, A6]."""
        return self.sim.data.qpos[self.arm_index]

    @robot_arm_pos.setter
    def robot_arm_pos(self, value):
        self.sim.data.qpos[self.arm_index] = value

    @property
    def robot_arm_vel(self):
        """Angular velocities of the joints, [A1, A2, E1, A3, A4, A5, A6]."""
        return self.sim.data.qvel[self.arm_index]

    @property
    def robot_arm_acc(self):
        """Angular acceleration of the joints, [A1, A2, E1, A3, A4, A5, A6]."""
        return self.sim.data.qacc[self.arm_index]