"""
Spot Robot RL Keyboard Controller

Standalone controller that loads a trained PPO model and allows
manual control of target velocities via arrow keys.

Controls:
    ↑ / ↓  : Increase / Decrease forward velocity (lin_vel_x)
    ← / →  : Increase / Decrease lateral velocity (lin_vel_y)
    SPACE  : Stop (reset velocities to zero)
"""

import os
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from controller import Supervisor, Keyboard

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ================== CONFIGURATION ==================
MODEL_DIR = "../spot_gym/models"
MODEL_NAME = "final_model_v3"  # or "best_model"

VEL_INCREMENT = 0.05  # How much each key press changes velocity
MAX_LIN_VEL_X = 10.0
MIN_LIN_VEL_X = -5.0
MAX_LIN_VEL_Y = 0.5
MIN_LIN_VEL_Y = -0.5

FIXED_ANG_VEL = 0.0
FIXED_HEIGHT = 0.62  # Comfortable standing/walking height

ENV_CONFIG = {
    "tracking_sigma": 0.25,
    "base_height_target": 0.3,
    "default_joint_angles": {
        "front left shoulder abduction motor": -0.1,
        "front left shoulder rotation motor": 0.0,
        "front left elbow motor": 0.0,
        "front right shoulder abduction motor": 0.1,
        "front right shoulder rotation motor": 0.0,
        "front right elbow motor": 0.0,
        "rear left shoulder abduction motor": -0.1,
        "rear left shoulder rotation motor": 0.0,
        "rear left elbow motor": 0.0,
        "rear right shoulder abduction motor": 0.1,
        "rear right shoulder rotation motor": 0.0,
        "rear right elbow motor": 0.0,
    },
    "roll_threshold": 0.2,
    "pitch_threshold": 0.2,
    "z_min": 0.3,
    "max_episode_steps": 100_000,  # Effectively infinite for manual control
    "lin_vel_x_range": [-0.5, 1.0],
    "lin_vel_y_range": [-0.5, 0.5],
    "ang_vel_range": [-0.5, 0.5],
    "height_range": [0.2, 0.4],
    "action_scale": 0.4,
}


# ================== ENVIRONMENT ==================


class SpotEnv(Supervisor, gym.Env):
    """Gymnasium environment for Spot — identical observation/action interface as training."""

    metadata = {"render_modes": ["human"]}

    MOTOR_NAMES = [
        "front left shoulder abduction motor",
        "front left shoulder rotation motor",
        "front left elbow motor",
        "front right shoulder abduction motor",
        "front right shoulder rotation motor",
        "front right elbow motor",
        "rear left shoulder abduction motor",
        "rear left shoulder rotation motor",
        "rear left elbow motor",
        "rear right shoulder abduction motor",
        "rear right shoulder rotation motor",
        "rear right elbow motor",
    ]

    NUM_MOTORS = 12
    NUM_OBS = 46

    def __init__(self, config=None):
        Supervisor.__init__(self)
        gym.Env.__init__(self)

        self.config = config or ENV_CONFIG
        self.timestep = int(self.getBasicTimeStep())
        self.dt = self.timestep / 1000.0

        self.robot_node = self.getFromDef("SPOT")
        if self.robot_node is None:
            self.robot_node = self.getSelf()

        # Motors & sensors
        self.motors = []
        self.position_sensors = []
        for name in self.MOTOR_NAMES:
            motor = self.getDevice(name)
            self.motors.append(motor)
            if motor:
                sensor = motor.getPositionSensor()
                if sensor:
                    sensor.enable(self.timestep)
                self.position_sensors.append(sensor)
            else:
                self.position_sensors.append(None)

        self.accelerometer = self.getDevice("accelerometer")
        self.gyro = self.getDevice("gyro")
        self.gps = self.getDevice("gps")
        self.inertial_unit = self.getDevice("inertial unit")

        for sensor in [self.accelerometer, self.gyro, self.gps, self.inertial_unit]:
            if sensor:
                sensor.enable(self.timestep)

        # Spaces
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.NUM_MOTORS,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.NUM_OBS,), dtype=np.float32
        )

        # Defaults
        self.default_dof_pos = np.array(
            [self.config["default_joint_angles"][n] for n in self.MOTOR_NAMES],
            dtype=np.float32,
        )

        self.last_actions = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self.actions = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self.dof_pos = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self.dof_vel = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self.last_dof_pos = np.zeros(self.NUM_MOTORS, dtype=np.float32)

        self.commands = np.array(
            [0.0, 0.0, FIXED_ANG_VEL, FIXED_HEIGHT], dtype=np.float32
        )
        self.current_step = 0

        # Init sim
        super().step(self.timestep)

        self.init_position = None
        self.init_rotation = None
        if self.robot_node:
            tf = self.robot_node.getField("translation")
            rf = self.robot_node.getField("rotation")
            if tf:
                self.init_position = tf.getSFVec3f()
            if rf:
                self.init_rotation = rf.getSFRotation()

    # ---------- observation (must match training) ----------
    def _get_observation(self):
        obs = []

        # Angular velocity (3)
        ang_vel = (
            np.array(self.gyro.getValues(), dtype=np.float32)
            if self.gyro
            else np.zeros(3, dtype=np.float32)
        )
        obs.extend(ang_vel * 0.25)

        # Projected gravity (3)
        if self.inertial_unit:
            roll, pitch, _ = self.inertial_unit.getRollPitchYaw()
            gravity = np.array(
                [
                    -np.sin(pitch),
                    np.sin(roll) * np.cos(pitch),
                    -np.cos(roll) * np.cos(pitch),
                ],
                dtype=np.float32,
            )
        else:
            gravity = np.array([0, 0, -1], dtype=np.float32)
        obs.extend(gravity)

        # Commands (4)
        scaled = self.commands.copy()
        scaled[:2] *= 2.0
        scaled[2] *= 0.25
        obs.extend(scaled)

        # Joint positions relative to default (12)
        for i, sensor in enumerate(self.position_sensors):
            self.dof_pos[i] = sensor.getValue() if sensor else self.default_dof_pos[i]
        obs.extend(self.dof_pos - self.default_dof_pos)

        # Joint velocities (12)
        self.dof_vel = (self.dof_pos - self.last_dof_pos) / self.dt
        obs.extend(self.dof_vel * 0.05)

        # Last actions (12)
        obs.extend(self.last_actions)

        self.last_dof_pos = self.dof_pos.copy()
        return np.array(obs, dtype=np.float32)

    # ---------- step ----------
    def step(self, action):
        self.current_step += 1
        self.actions = np.clip(action, -1.0, 1.0).astype(np.float32)

        target_pos = self.actions * self.config["action_scale"] + self.default_dof_pos
        for i, motor in enumerate(self.motors):
            if motor:
                motor.setPosition(float(target_pos[i]))

        if super().step(self.timestep) == -1:
            return self._get_observation(), 0.0, True, True, {}

        obs = self._get_observation()
        self.last_actions = self.actions.copy()

        # No real termination in manual mode — just keep going
        return obs, 0.0, False, False, {}

    # ---------- reset ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulationReset()
        self.simulationResetPhysics()

        for _ in range(5):
            super().step(self.timestep)

        if self.robot_node and self.init_position and self.init_rotation:
            tf = self.robot_node.getField("translation")
            rf = self.robot_node.getField("rotation")
            if tf:
                tf.setSFVec3f(self.init_position)
            if rf:
                rf.setSFRotation(self.init_rotation)

        for i, motor in enumerate(self.motors):
            if motor:
                motor.setPosition(float(self.default_dof_pos[i]))

        for _ in range(10):
            super().step(self.timestep)

        self.last_actions = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self.actions = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self.last_dof_pos = self.dof_pos.copy()
        self.commands = np.array(
            [0.0, 0.0, FIXED_ANG_VEL, FIXED_HEIGHT], dtype=np.float32
        )
        self.current_step = 0

        return self._get_observation(), {}

    def render(self):
        pass

    def close(self):
        pass


# ================== MAIN LOOP ==================


def main():
    env = SpotEnv(config=ENV_CONFIG)

    # --- Load model ---
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: No model found at {model_path}.zip")
        print("Train a model first with the training controller.")
        return

    vec_env = DummyVecEnv([lambda: env])

    vec_norm_path = os.path.join(MODEL_DIR, f"vec_normalize_{MODEL_NAME}.pkl")
    if os.path.exists(vec_norm_path):
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print("Loaded observation normalisation stats.")

    model = PPO.load(model_path, env=vec_env)
    print(f"Model loaded from {model_path}")

    # --- Keyboard ---
    keyboard = env.getKeyboard()
    keyboard.enable(env.timestep)

    # Real-time so you can actually watch
    env.simulationSetMode(env.SIMULATION_MODE_REAL_TIME)

    # Target velocities
    target_vx = 0.3
    target_vy = 0.0

    obs = vec_env.reset()
    env.commands = np.array(
        [target_vx, target_vy, FIXED_ANG_VEL, FIXED_HEIGHT], dtype=np.float32
    )

    print("\n" + "=" * 50)
    print("  Spot Keyboard Controller")
    print("=" * 50)
    print("  ↑ / ↓   Forward / Backward")
    print("  ← / →   Strafe Left / Right")
    print("  SPACE    Stop (zero velocities)")
    print("=" * 50)
    print(f"  Velocity step : {VEL_INCREMENT}")
    print(f"  Fixed height  : {FIXED_HEIGHT}")
    print("=" * 50 + "\n")

    step_count = 0
    while True:
        # --- Read keyboard (process all queued keys) ---
        key = keyboard.getKey()
        while key != -1:
            if key == Keyboard.UP:
                target_vx = min(target_vx + VEL_INCREMENT, MAX_LIN_VEL_X)
            elif key == Keyboard.DOWN:
                target_vx = max(target_vx - VEL_INCREMENT, MIN_LIN_VEL_X)
            elif key == Keyboard.LEFT:
                target_vy = min(target_vy + VEL_INCREMENT, MAX_LIN_VEL_Y)
            elif key == Keyboard.RIGHT:
                target_vy = max(target_vy - VEL_INCREMENT, MIN_LIN_VEL_Y)
            elif key == ord(" "):
                target_vx = 0.0
                target_vy = 0.0
            key = keyboard.getKey()

        # Update commands
        env.commands = np.array(
            [target_vx, target_vy, FIXED_ANG_VEL, FIXED_HEIGHT], dtype=np.float32
        )

        # Inference
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = vec_env.step(action)

        if done[0]:
            print("Episode terminated — resetting…")
            obs = vec_env.reset()
            env.commands = np.array(
                [target_vx, target_vy, FIXED_ANG_VEL, FIXED_HEIGHT], dtype=np.float32
            )

        # HUD print every 100 steps
        step_count += 1
        if step_count % 100 == 0:
            print(f"  vx={target_vx:+.2f}  vy={target_vy:+.2f}  steps={step_count}")


if __name__ == "__main__":
    main()
