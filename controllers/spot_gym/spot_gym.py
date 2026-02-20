"""
Spot Robot RL Walking Gait Controller

This controller implements reinforcement learning for the Boston Dynamics Spot robot
using PPO from Stable Baselines3 with Gymnasium interface.

Usage:
    Training: Set MODE = "TRAIN" and run the simulation
    Evaluation: Set MODE = "EVAL" and run the simulation
"""

import os
import sys
import math
import numpy as np
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces

from controller import Supervisor

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


# ================== CONFIGURATION ==================
MODE = "EVAL"  # "TRAIN" or "EVAL"
MODEL_DIR = "models"
LOG_DIR = "spot_tensorboard"
TOTAL_TIMESTEPS = 1000_000
SAVE_FREQ = 100_000  # Check for best model every N timesteps
PRINT_FREQ = 1000  # Print metrics every N timesteps

# Reward weights (can be tuned)
REWARD_WEIGHTS = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.2,
    "height_penalty": -5.0,
    "pose_similarity": -0.1,
    "action_rate": -0.005,
    "lin_vel_z": -1.0,
    "roll_pitch": -0.5,
}

# Environment parameters
ENV_CONFIG = {
    "tracking_sigma": 0.25,
    "base_height_target": 0.3,  # Target height in meters
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
    # Termination thresholds
    "roll_threshold": 0.2,  # radians (~28 degrees)
    "pitch_threshold": 0.2,  # radians (~28 degrees)
    "z_min": 0.3,  # minimum height before termination
    "max_episode_steps": 1000,
    # Command ranges
    "lin_vel_x_range": [-0.5, 1.0],
    "lin_vel_y_range": [-0.5, 0.5],
    "ang_vel_range": [-0.5, 0.5],
    "height_range": [0.2, 0.4],
    # Action scaling
    "action_scale": 0.4,
}


class SpotEnv(Supervisor, gym.Env):
    """
    Custom Gymnasium environment for the Boston Dynamics Spot robot in Webots.

    Observation Space (45 dimensions):
        - Base angular velocity (3)
        - Projected gravity (3)
        - Commands: lin_vel_x, lin_vel_y, ang_vel, height (4)
        - Joint positions relative to default (12)
        - Joint velocities (12)
        - Last actions (12)

    Action Space (12 dimensions):
        - Target position offsets for 12 joint motors
    """

    metadata = {"render_modes": ["human"]}

    # Motor names from the Spot robot
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
    NUM_OBS = 46  # 3 + 3 + 4 + 12 + 12 + 12

    def __init__(self, config=None):
        # Initialize Webots Supervisor
        Supervisor.__init__(self)

        # Initialize Gym Env
        gym.Env.__init__(self)

        self.config = config if config else ENV_CONFIG

        # Get simulation timestep
        self.timestep = int(self.getBasicTimeStep())
        self.dt = self.timestep / 1000.0  # Convert to seconds

        # Get robot node for state access
        self.robot_node = self.getFromDef("SPOT")
        if self.robot_node is None:
            self.robot_node = self.getSelf()

        # Initialize motors
        self.motors = []
        self.position_sensors = []
        for name in self.MOTOR_NAMES:
            motor = self.getDevice(name)
            if motor:
                self.motors.append(motor)
                # Get position sensor for each motor
                sensor_name = name.replace("motor", "sensor")
                sensor = motor.getPositionSensor()
                if sensor:
                    sensor.enable(self.timestep)
                    self.position_sensors.append(sensor)
                else:
                    self.position_sensors.append(None)
            else:
                print(f"Warning: Motor '{name}' not found")
                self.motors.append(None)
                self.position_sensors.append(None)

        # Initialize sensors
        self.accelerometer = self.getDevice("accelerometer")
        self.gyro = self.getDevice("gyro")
        self.gps = self.getDevice("gps")
        self.inertial_unit = self.getDevice("inertial unit")

        # Enable sensors
        if self.accelerometer:
            self.accelerometer.enable(self.timestep)
        if self.gyro:
            self.gyro.enable(self.timestep)
        if self.gps:
            self.gps.enable(self.timestep)
        if self.inertial_unit:
            self.inertial_unit.enable(self.timestep)

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.NUM_MOTORS,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.NUM_OBS,), dtype=np.float32
        )

        # Default joint positions
        self.default_dof_pos = np.array(
            [self.config["default_joint_angles"][name] for name in self.MOTOR_NAMES],
            dtype=np.float32,
        )

        # State buffers
        self.last_actions = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self.actions = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self.dof_pos = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self.dof_vel = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self.last_dof_pos = np.zeros(self.NUM_MOTORS, dtype=np.float32)

        # Commands: [lin_vel_x, lin_vel_y, ang_vel, height]
        self.commands = np.zeros(4, dtype=np.float32)

        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_rewards = []  # Track reward components

        # Initial position for reset
        self.init_position = None
        self.init_rotation = None

        # Step the simulation once to initialize sensors
        super().step(self.timestep)

        # Store initial position
        if self.robot_node:
            trans_field = self.robot_node.getField("translation")
            rot_field = self.robot_node.getField("rotation")
            if trans_field:
                self.init_position = trans_field.getSFVec3f()
            if rot_field:
                self.init_rotation = rot_field.getSFRotation()

    def _get_observation(self):
        """Construct observation vector from sensor readings."""
        obs = []

        # Base angular velocity (3)
        if self.gyro:
            ang_vel = np.array(self.gyro.getValues(), dtype=np.float32)
        else:
            ang_vel = np.zeros(3, dtype=np.float32)
        obs.extend(ang_vel * 0.25)  # Scale angular velocity

        # Projected gravity (3) - derived from IMU
        if self.inertial_unit:
            rpy = self.inertial_unit.getRollPitchYaw()
            roll, pitch, _ = rpy
            # Project gravity into body frame
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

        # Commands (4): [lin_vel_x, lin_vel_y, ang_vel, height]
        scaled_commands = self.commands.copy()
        scaled_commands[:2] *= 2.0  # Scale linear velocity
        scaled_commands[2] *= 0.25  # Scale angular velocity
        obs.extend(scaled_commands)

        # Joint positions relative to default (12)
        for i, sensor in enumerate(self.position_sensors):
            if sensor:
                self.dof_pos[i] = sensor.getValue()
            else:
                self.dof_pos[i] = self.default_dof_pos[i]
        obs.extend((self.dof_pos - self.default_dof_pos))

        # Joint velocities (12) - approximate from position difference
        self.dof_vel = (self.dof_pos - self.last_dof_pos) / self.dt
        obs.extend(self.dof_vel * 0.05)  # Scale velocities

        # Last actions (12)
        obs.extend(self.last_actions)

        self.last_dof_pos = self.dof_pos.copy()

        return np.array(obs, dtype=np.float32)

    def _get_base_state(self):
        """Get base position, velocity, and orientation."""
        # Position from GPS
        if self.gps:
            pos = np.array(self.gps.getValues(), dtype=np.float32)
        else:
            pos = np.zeros(3, dtype=np.float32)

        # Orientation from IMU
        if self.inertial_unit:
            rpy = self.inertial_unit.getRollPitchYaw()
            roll, pitch, yaw = rpy
        else:
            roll, pitch, yaw = 0, 0, 0

        # Linear velocity from GPS (if available) or estimate
        if self.gps:
            lin_vel = np.array(self.gps.getSpeedVector(), dtype=np.float32)
        else:
            lin_vel = np.zeros(3, dtype=np.float32)

        # Angular velocity from gyro
        if self.gyro:
            ang_vel = np.array(self.gyro.getValues(), dtype=np.float32)
        else:
            ang_vel = np.zeros(3, dtype=np.float32)

        return {
            "pos": pos,
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }

    def _sample_commands(self):
        """Sample new velocity and height commands."""
        self.commands[0] = np.random.uniform(*self.config["lin_vel_x_range"])
        self.commands[1] = np.random.uniform(*self.config["lin_vel_y_range"])
        self.commands[2] = np.random.uniform(*self.config["ang_vel_range"])
        self.commands[3] = np.random.uniform(*self.config["height_range"])

    def _compute_rewards(self, base_state):
        """
        Compute all 7 reward components as specified in reward_and_termination.md

        Returns:
            tuple: (total_reward, reward_dict)
        """
        rewards = {}

        # 1. Linear Velocity Tracking Reward
        # R_lin_vel = exp(-||v_xy_ref - v_xy||^2 / sigma)
        lin_vel_xy = base_state["lin_vel"][:2]
        lin_vel_error = np.sum((self.commands[:2] - lin_vel_xy) ** 2)
        rewards["tracking_lin_vel"] = np.exp(
            -lin_vel_error / self.config["tracking_sigma"]
        )

        # 2. Angular Velocity Tracking Reward
        # R_ang_vel = exp(-(w_z_ref - w_z)^2 / sigma)
        ang_vel_z = base_state["ang_vel"][2]
        ang_vel_error = (self.commands[2] - ang_vel_z) ** 2
        rewards["tracking_ang_vel"] = np.exp(
            -ang_vel_error / self.config["tracking_sigma"]
        )

        # 3. Height Penalty
        # R_z = (z - z_ref)^2
        z = base_state["pos"][2]
        z_ref = self.commands[3]
        rewards["height_penalty"] = (z - z_ref) ** 2

        # 4. Pose Similarity Reward (Penalty)
        # R_pose_similarity = ||q - q_default||^2
        rewards["pose_similarity"] = np.sum((self.dof_pos - self.default_dof_pos) ** 2)

        # 5. Action Rate Penalty
        # R_action_rate = ||a_t - a_{t-1}||^2
        rewards["action_rate"] = np.sum((self.actions - self.last_actions) ** 2)

        # 6. Vertical Velocity Penalty
        # R_lin_vel_z = v_z^2
        v_z = base_state["lin_vel"][2]
        rewards["lin_vel_z"] = v_z**2

        # 7. Roll and Pitch Stabilization Penalty
        # R_roll_pitch = roll^2 + pitch^2
        rewards["roll_pitch"] = base_state["roll"] ** 2 + base_state["pitch"] ** 2

        # Compute total weighted reward
        total_reward = 0.0
        for name, value in rewards.items():
            weighted = value * REWARD_WEIGHTS.get(name, 0.0) * self.dt
            total_reward += weighted
            rewards[name] = weighted  # Store weighted value

        return total_reward, rewards

    def _is_healthy(self, base_state):
        """
        Check if robot is in a healthy state.

        Termination conditions from reward_and_termination.md:
        - |roll| < roll_threshold
        - |pitch| < pitch_threshold
        - z > z_min
        - steps < max_steps

        Returns:
            tuple: (is_healthy, termination_reason)
        """
        roll = base_state["roll"]
        pitch = base_state["pitch"]
        z = base_state["pos"][2]

        if abs(roll) > self.config["roll_threshold"]:
            return False, "roll_exceeded"
        if abs(pitch) > self.config["pitch_threshold"]:
            return False, "pitch_exceeded"
        if z < self.config["z_min"]:
            return False, "height_too_low"
        if self.current_step >= self.config["max_episode_steps"]:
            return False, "max_steps_reached"

        return True, None

    def step(self, action):
        """Execute one environment step."""
        self.current_step += 1

        # Store action
        self.actions = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Convert action to target joint positions
        target_pos = self.actions * self.config["action_scale"] + self.default_dof_pos

        # Apply to motors
        for i, motor in enumerate(self.motors):
            if motor:
                motor.setPosition(float(target_pos[i]))

        # Step simulation
        if super().step(self.timestep) == -1:
            # Simulation ended
            return self._get_observation(), 0.0, True, True, {}

        # Get new state
        base_state = self._get_base_state()
        obs = self._get_observation()

        # Compute rewards
        reward, reward_components = self._compute_rewards(base_state)
        self.episode_reward += reward
        self.episode_rewards.append(reward_components)

        # Check termination
        is_healthy, termination_reason = self._is_healthy(base_state)
        terminated = not is_healthy
        truncated = self.current_step >= self.config["max_episode_steps"]

        # Update last actions
        self.last_actions = self.actions.copy()

        info = {
            "reward_components": reward_components,
            "base_state": base_state,
            "commands": {
                "lin_vel_x": float(self.commands[0]),
                "lin_vel_y": float(self.commands[1]),
                "ang_vel": float(self.commands[2]),
                "height": float(self.commands[3]),
            },
        }

        # Add termination reason if episode ended
        if terminated:
            info["termination_reason"] = termination_reason

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset simulation
        self.simulationReset()
        self.simulationResetPhysics()

        # Wait for reset to complete
        for _ in range(5):
            super().step(self.timestep)

        # Reset robot position if we have the node
        if self.robot_node and self.init_position and self.init_rotation:
            trans_field = self.robot_node.getField("translation")
            rot_field = self.robot_node.getField("rotation")
            if trans_field:
                trans_field.setSFVec3f(self.init_position)
            if rot_field:
                rot_field.setSFRotation(self.init_rotation)

        # Reset motors to default positions
        for i, motor in enumerate(self.motors):
            if motor:
                motor.setPosition(float(self.default_dof_pos[i]))

        # Step to apply motor positions
        for _ in range(10):
            super().step(self.timestep)

        # Reset state buffers
        self.last_actions = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self.actions = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self.last_dof_pos = self.dof_pos.copy()

        # Sample new commands
        self._sample_commands()

        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_rewards = []

        obs = self._get_observation()
        info = {}

        return obs, info

    def render(self):
        """Rendering is handled by Webots."""
        pass

    def close(self):
        """Cleanup."""
        pass


class TensorBoardCallback(BaseCallback):
    """
    Custom callback for logging training metrics to TensorBoard.
    Also handles model checkpointing based on mean episode reward.
    """

    def __init__(self, log_dir, model_dir, save_freq=10000, print_freq=1000, verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.save_freq = save_freq
        self.print_freq = print_freq
        self.writer = None

        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf
        self.n_episodes = 0

        # Track termination reasons
        self.termination_counts = {
            "roll_exceeded": 0,
            "pitch_exceeded": 0,
            "height_too_low": 0,
            "max_steps_reached": 0,
        }

        # Track commands
        self.last_commands = {
            "lin_vel_x": 0.0,
            "lin_vel_y": 0.0,
            "ang_vel": 0.0,
            "height": 0.0,
        }

        # Ensure directories exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

    def _on_training_start(self):
        """Initialize TensorBoard writer."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.log_dir, f"run_{timestamp}")
        )
        print(f"\n{'='*60}")
        print(f"TensorBoard logging to: {self.log_dir}")
        print(f"Models saving to: {self.model_dir}")
        print(f"{'='*60}\n")

    def _on_step(self):
        """Called at each step."""
        # Log commands at each step
        if "infos" in self.locals and len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]
            if "commands" in info:
                self.last_commands = info["commands"]
                # Log commands periodically (every 100 steps to reduce overhead)
                if self.n_calls % 100 == 0:
                    for cmd_name, cmd_value in info["commands"].items():
                        self.writer.add_scalar(
                            f"commands/{cmd_name}", cmd_value, self.n_calls
                        )

        # Check for episode end
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    # Get episode info
                    if "infos" in self.locals and len(self.locals["infos"]) > i:
                        info = self.locals["infos"][i]

                        # Log termination reason
                        if "termination_reason" in info and info["termination_reason"]:
                            reason = info["termination_reason"]
                            if reason in self.termination_counts:
                                self.termination_counts[reason] += 1
                            # Log termination reason
                            self.writer.add_text(
                                "termination/reason",
                                f"Episode {self.n_episodes + 1}: {reason}",
                                self.n_episodes + 1,
                            )

                        if "episode" in info:
                            ep_reward = info["episode"]["r"]
                            ep_length = info["episode"]["l"]
                            self.episode_rewards.append(ep_reward)
                            self.episode_lengths.append(ep_length)
                            self.n_episodes += 1

                            # Log to TensorBoard
                            self.writer.add_scalar(
                                "rollout/ep_rew", ep_reward, self.n_episodes
                            )
                            self.writer.add_scalar(
                                "rollout/ep_len", ep_length, self.n_episodes
                            )

                            # Log termination counts
                            for reason, count in self.termination_counts.items():
                                self.writer.add_scalar(
                                    f"termination/{reason}", count, self.n_episodes
                                )

        # Periodic logging and checkpoint
        if self.n_calls % self.print_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_length = np.mean(self.episode_lengths[-100:])

            # Calculate termination percentages
            total_terminations = sum(self.termination_counts.values())
            term_pcts = {}
            if total_terminations > 0:
                for reason, count in self.termination_counts.items():
                    term_pcts[reason] = (count / total_terminations) * 100

            print(f"\n{'='*60}")
            print(f"Timestep: {self.n_calls}")
            print(f"Episodes: {self.n_episodes}")
            print(f"Mean Reward (last 100): {mean_reward:.2f}")
            print(f"Mean Episode Length (last 100): {mean_length:.2f}")
            print(f"Best Mean Reward: {self.best_mean_reward:.2f}")
            print(f"\nCurrent Commands:")
            print(f"  lin_vel_x: {self.last_commands['lin_vel_x']:.2f}")
            print(f"  lin_vel_y: {self.last_commands['lin_vel_y']:.2f}")
            print(f"  ang_vel:   {self.last_commands['ang_vel']:.2f}")
            print(f"  height:    {self.last_commands['height']:.2f}")
            print(f"\nTermination Reasons (total: {total_terminations}):")
            for reason, count in self.termination_counts.items():
                pct = term_pcts.get(reason, 0)
                print(f"  {reason}: {count} ({pct:.1f}%)")
            print(f"{'='*60}\n")

            # Log mean metrics
            self.writer.add_scalar("rollout/ep_rew_mean", mean_reward, self.n_calls)
            self.writer.add_scalar("rollout/ep_len_mean", mean_length, self.n_calls)

        # Check for best model
        if self.n_calls % self.save_freq == 0 and len(self.episode_rewards) >= 10:
            mean_reward = np.mean(self.episode_rewards[-100:])

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

                # Save best model
                best_model_path = os.path.join(self.model_dir, "best_model")
                self.model.save(best_model_path)

                # Save VecNormalize statistics if available
                if hasattr(self.training_env, "save"):
                    vec_norm_path = os.path.join(self.model_dir, "vec_normalize.pkl")
                    self.training_env.save(vec_norm_path)

                print(
                    f"\n*** New best model saved! Mean reward: {mean_reward:.2f} ***\n"
                )

                self.writer.add_scalar(
                    "train/best_mean_reward", mean_reward, self.n_calls
                )

            # Also save periodic checkpoint
            checkpoint_path = os.path.join(self.model_dir, f"checkpoint_{self.n_calls}")
            self.model.save(checkpoint_path)

        return True

    def _on_training_end(self):
        """Save final model and close writer."""
        final_path = os.path.join(self.model_dir, "final_model")
        self.model.save(final_path)

        if hasattr(self.training_env, "save"):
            vec_norm_path = os.path.join(self.model_dir, "vec_normalize_final.pkl")
            self.training_env.save(vec_norm_path)

        self.writer.close()
        print(f"\nTraining complete. Final model saved to {final_path}")


def train(env, total_timesteps=TOTAL_TIMESTEPS):
    """Train the Spot robot using PPO."""

    # Ensure log directory exists for Monitor
    os.makedirs(LOG_DIR, exist_ok=True)

    # Wrap with Monitor to track episode statistics (ep_rew, ep_len)
    monitored_env = Monitor(env, filename=os.path.join(LOG_DIR, "monitor"))

    # Wrap in DummyVecEnv for SB3 compatibility
    vec_env = DummyVecEnv([lambda: monitored_env])

    # Normalize observations and rewards
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=0.001,
        n_steps=4096,
        batch_size=512,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # Create callback
    callback = TensorBoardCallback(
        log_dir=LOG_DIR,
        model_dir=MODEL_DIR,
        save_freq=SAVE_FREQ,
        print_freq=PRINT_FREQ,
    )

    # Use fast simulation mode for training
    env.simulationSetMode(env.SIMULATION_MODE_FAST)

    print("\n" + "=" * 60)
    print("Starting Spot RL Training")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Save Frequency: {SAVE_FREQ}")
    print("=" * 60 + "\n")

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    return model, vec_env


def evaluate(env, model_path=None):
    """Evaluate a trained model."""

    # Load model
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "final_model")

    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first by setting MODE = 'TRAIN'")
        return

    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: env])

    # Load VecNormalize if available
    vec_norm_path = os.path.join(MODEL_DIR, "vec_normalize_final.pkl")
    if os.path.exists(vec_norm_path):
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Load model
    model = PPO.load(model_path, env=vec_env)

    # Use real-time simulation for evaluation
    env.simulationSetMode(env.SIMULATION_MODE_REAL_TIME)

    print("\n" + "=" * 60)
    print("Starting Spot RL Evaluation")
    print(f"Model: {model_path}")
    print("=" * 60 + "\n")

    # Run evaluation
    obs = vec_env.reset()
    total_reward = 0
    episode_count = 0
    step_count = 0

    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        step_count += 1

        if done[0]:
            episode_count += 1
            print(f"\nEpisode {episode_count} finished:")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Episode Length: {step_count}")

            # Reset for next episode
            total_reward = 0
            step_count = 0
            obs = vec_env.reset()

            # Set commands for evaluation (forward walking)
            if hasattr(env, "commands"):
                env.commands = np.array([0.5, 0.5, 0.0, 0.35], dtype=np.float32)


def main():
    """Main entry point."""

    # Create environment
    env = SpotEnv(config=ENV_CONFIG)

    print("\n" + "=" * 60)
    print(f"Spot RL Controller - Mode: {MODE}")
    print("=" * 60 + "\n")

    if MODE == "TRAIN":
        model, vec_env = train(env, total_timesteps=TOTAL_TIMESTEPS)
    elif MODE == "EVAL":
        evaluate(env)
    else:
        print(f"Unknown mode: {MODE}")
        print("Please set MODE to 'TRAIN' or 'EVAL'")


if __name__ == "__main__":
    main()
