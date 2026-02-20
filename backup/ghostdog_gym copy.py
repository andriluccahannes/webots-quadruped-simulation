import sys
import os
import numpy as np
import gymnasium as gym
from controller import Supervisor

# Import Stable Baselines 3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    sys.exit('Please install SB3: "pip install stable-baselines3 gymnasium"')

# ============ CONFIGURATION ============
CONFIG = {
    # Training mode
    "training_mode": True,  # Set to False for inference
    "model_name": "ppo_ghostdog_v38",
    "vecnorm_name": "vec_normalize_v38.pkl",
    # Episode parameters
    "max_episode_steps": 1000,
    "control_timestep": 0.02,  # 50 Hz control
    # Command references (for curriculum learning)
    "command_v_x_ref": 1,  # Target forward velocity (m/s)
    "command_v_y_ref": 0.0,  # Target lateral velocity (m/s)
    "command_w_z_ref": 0.0,  # Target yaw rate (rad/s)
    "command_z_ref": 0.25,  # Target base height (m) - raised from 0.25
    # Randomization (for robust policy)
    "noise_pos": 0.05,  # Position noise on reset
    "noise_vel": 0.1,  # Velocity noise on reset
    "noise_joint": 0.1,  # Joint position noise on reset
    # Reward function weights (REFERENCE ALIGNED - stability >> speed)
    "reward_weights": {
        "lin_vel_tracking": 8.0,  # w1: Track commanded velocity (INCREASED 8x - dominant reward)
        "ang_vel_tracking": 0.5,  # w2: Track commanded yaw rate (increased from 0.2)
        "height_penalty": 2,  # w3: Maintain target height (REDUCED from 50.0)
        "pose_penalty": 0.1,  # w4: Stay near default pose (ref: 0.1)
        "action_rate_penalty": 0.005,  # w5: Smooth actions (ref: 0.005)
        "lin_vel_z_penalty": 1,  # w6: Discourage vertical bouncing (ref: 1.0)
        "orientation_penalty": 1,  # w7: Stay level (REDUCED from 5.0)
        # NEW: Gait-specific rewards for natural walking
    },
    # Command randomization for curriculum learning
    "randomize_commands": True,  # Randomize v_x, v_y, w_z during training
    "command_ranges": {
        "v_x": (-1, 2),  # Forward/backward velocity (ref values)
        "v_y": (0.0, 0.0),  # Lateral velocity
        "w_z": (0.0, 0.0),  # Yaw rate
        "z": (0.2, 0.25),  # Height range (NEW - randomize height!)
    },
    # Dynamic commands in inference
    "inference_command_interval": 200,  # Change commands every 4s (was 10s)
    # Reward function tuning parameters
    "sigma_vel": 1,  # Variance for velocity tracking reward
    "sigma_ang": 0.1,  # Variance for angular velocity tracking
    # Observation scaling (CRITICAL - from reference)
    "obs_scales": {
        "lin_vel": 1,  # Scale velocities UP
        "ang_vel": 1,  # Scale angular velocities DOWN
        "dof_pos": 1.0,  # No scaling
        "dof_vel": 1,  # Scale joint velocities DOWN significantly
    },
    # Action scaling (CRITICAL - from reference)
    "action_scale": 0.4,  # Scale actions to [-0.25, 0.25] before applying
    # Termination conditions (BALANCED - progressive tightening through curriculum)
    "min_height": 0.18,  # Terminate if base too low
    "max_roll": 0.3,  # Max roll: 17 degrees (0.3 rad - BALANCED between strict and loose)
    "max_pitch": 0.3,  # Max pitch: 17 degrees (0.3 rad)
    # Default joint positions (natural quadruped stance - standing crouch)
    # [hip0, hip1, hip2, hip3, knee0, knee1, knee2, knee3]
    # Hips: -0.6 rad (forward tilt), Knees: 0.6 rad (bent for stable crouch)
    "default_joint_positions": [
        0,  # hip0 (front-right)
        0,  # hip1 (rear-left)
        0,  # hip2 (front-left)
        0,  # hip3 (rear-right)
        0,  # knee0 (front-right thigh)
        0,  # knee1 (rear-left thigh)
        0,  # knee2 (front-left thigh)
        0,  # knee3 (rear-right thigh)
    ],
    # PPO hyperparameters (ADJUSTED - closer to reference)
    "ppo_learning_rate": 1e-3,  # Increased from 3e-4 (ref: 1e-3)
    "ppo_n_steps": 2048,
    "ppo_batch_size": 64,
    "ppo_n_epochs": 5,  # Reduced from 10 (ref: 5)
    "ppo_gamma": 0.99,
    "ppo_gae_lambda": 0.95,
    "ppo_clip_range": 0.2,
    "ppo_ent_coef": 0.01,  # Increased from 0.01 for more exploration
    # Training
    "total_timesteps": 500_000,
    # Curriculum learning for progressive difficulty
    "curriculum": {
        "enabled": True,
        "stages": [
            # Stage 0: Learn to stand (0-200k timesteps)
            {
                "name": "standing",
                "timesteps": 400_000,
                "command_ranges": {
                    "v_x": (-1, 2),  # Very slow forward only
                    "v_y": (0.0, 0.0),  # No lateral
                    "w_z": (0.0, 0.0),  # No turning
                    "z": (0.2, 0.25),
                },
            },
            # # Stage 1: Learn slow walking (200k-600k timesteps)
            # {
            #     "name": "slow_walk",
            #     "timesteps": 600_000,
            #     "command_ranges": {
            #         "v_x": (4, 5),  # Slow forward/backward
            #         "v_y": (0, 0),  # Slow lateral
            #         "w_z": (0, 0),  # Slow turning
            #         "z": (0.20, 0.25),
            #     },
            # },
            # # Stage 2: Full speed (600k+ timesteps)
            # {
            #     "name": "full_speed",
            #     "timesteps": float("inf"),
            #     "command_ranges": {
            #         "v_x": (-5, 8),  # Full range
            #         "v_y": (0, 0),
            #         "w_z": (0, 0),
            #         "z": (0.20, 0.25),
            #     },
            # },
        ],
    },
}
# ========================================


class GhostDogCurriculumEnv(Supervisor, gym.Env):
    """
    Ghost Dog Curriculum-Based Walking Environment

    Features:
    - 8-joint control (4 hips + 4 knees)
    - ~36 dimensional observation space
    - Exponential tracking rewards + quadratic penalties
    - Randomized reset for robustness
    - Config-driven design for easy tuning
    """

    def __init__(self, config=CONFIG):
        super().__init__()

        self.config = config
        self.timestep = int(self.getBasicTimeStep())

        # Get robot node
        self.robot_node = self.getFromDef("GHOST_DOG")
        if self.robot_node is None:
            sys.exit("ERROR: Ghost Dog robot node not found!")

        # ============ MOTORS ============
        # Initialize 8 motors: 4 hips + 4 knees
        motor_names = [
            "hip0",
            "hip1",
            "hip2",
            "hip3",
            "knee0",
            "knee1",
            "knee2",
            "knee3",
        ]
        self.motors = []
        self.motor_limits = []

        for name in motor_names:
            motor = self.getDevice(name)
            if motor is None:
                print(f"WARNING: Motor '{name}' not found! Using zero control.")
                self.motors.append(None)
                self.motor_limits.append((-1.0, 1.0))  # Default limits
            else:
                motor.setPosition(0)
                motor.setVelocity(motor.getMaxVelocity())
                motor.enableTorqueFeedback(self.timestep)
                self.motors.append(motor)

                min_pos = motor.getMinPosition()
                max_pos = motor.getMaxPosition()
                self.motor_limits.append((min_pos, max_pos))

        print(f"✓ Initialized {sum(1 for m in self.motors if m is not None)}/8 motors")

        # ============ SENSORS ============
        # Touch sensors for foot contact
        touch_sensor_names = ["touch0", "touch1", "touch2", "touch3"]
        self.touch_sensors = []
        for name in touch_sensor_names:
            sensor = self.getDevice(name)
            if sensor is not None:
                sensor.enable(self.timestep)
                self.touch_sensors.append(sensor)

        if len(self.touch_sensors) > 0:
            print(f"✓ Initialized {len(self.touch_sensors)} touch sensors")
        else:
            print("WARNING: No touch sensors found")

        # ============ ACTION/OBSERVATION SPACES ============
        # Action space: 8 continuous actions [-1, 1] for joint position targets
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Observation space: ~36 dimensions
        # [v_x, v_y, v_z] (3) + [w_x, w_y, w_z] (3) + [roll, pitch] (2) +
        # [q_0..q_7] (8) + [q_dot_0..q_dot_7] (8) + [a_t-1] (8) +
        # [v_x_ref, v_y_ref, w_z_ref, z_ref] (4) = 36
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
        )

        # ============ STATE TRACKING ============
        self.last_action = np.zeros(8)
        self.joint_positions = np.array(config["default_joint_positions"])
        self.joint_velocities = np.zeros(8)
        self.episode_steps = 0

        # ============ COMMAND TRACKING (for dynamic curriculum) ============
        # Initialize with CONFIG defaults, will be randomized in reset()
        self.command_v_x = config["command_v_x_ref"]
        self.command_v_y = config["command_v_y_ref"]
        self.command_w_z = config["command_w_z_ref"]
        self.command_z = config["command_z_ref"]

        # ============ CURRICULUM LEARNING TRACKING ============
        self.total_timesteps = 0
        self.curriculum_stage = 0

        # ============ GAIT TRACKING ============
        # Track last 50 foot contacts per foot for gait analysis
        self.foot_contact_history = {i: [] for i in range(4)}

        # ============ ACTION DIAGNOSTICS ============
        # Track actions for spring diagnostics
        self.episode_actions = []

        # ============ REWARD DIAGNOSTICS ============
        self.reward_components = {
            "lin_vel_tracking": [],
            "ang_vel_tracking": [],
            "height_penalty": [],
            "pose_penalty": [],
            "action_rate_penalty": [],
            "lin_vel_z_penalty": [],
            "orientation_penalty": [],
            "total": [],
        }

        print(f"\n{'='*60}")
        print("GHOST DOG CURRICULUM ENVIRONMENT")
        print(f"8-joint control | ~36D obs space | Exponential rewards")
        print(f"Target velocity: {config['command_v_x_ref']} m/s")
        print(f"Target height: {config['command_z_ref']} m")
        print(f"{'='*60}\n")

    def reset(self, seed=None, options=None):
        """
        Reset environment with randomization for robustness
        """
        super().reset(seed=seed)

        # Print reward breakdown from last episode
        if len(self.reward_components["total"]) > 0:
            print(f"\nEpisode Stats:")
            print(
                f"  {'Velocity tracking':20s}: {np.mean(self.reward_components['lin_vel_tracking']):8.2f}"
            )
            print(
                f"  {'Angular tracking':20s}: {np.mean(self.reward_components['ang_vel_tracking']):8.2f}"
            )
            print(
                f"  {'Height penalty':20s}: {np.mean(self.reward_components['height_penalty']):8.2f}"
            )
            print(
                f"  {'Pose penalty':20s}: {np.mean(self.reward_components['pose_penalty']):8.2f}"
            )
            print(
                f"  {'Action rate':20s}: {np.mean(self.reward_components['action_rate_penalty']):8.2f}"
            )
            print(
                f"  {'Lin vel z':20s}: {np.mean(self.reward_components['lin_vel_z_penalty']):8.2f}"
            )
            print(
                f"  {'Orientation':20s}: {np.mean(self.reward_components['orientation_penalty']):8.2f}"
            )
            print(f"  {'TOTAL':20s}: {np.mean(self.reward_components['total']):8.2f}")

            # Action range diagnostics (to check if springs dominate knee control)
            if hasattr(self, "episode_actions") and len(self.episode_actions) > 0:
                actions = np.array(self.episode_actions)
                print(f"\n  Action Diagnostics:")
                print(
                    f"    Hip actions:     [{np.min(actions[:, :4]):.2f}, {np.max(actions[:, :4]):.2f}]"
                )
                print(
                    f"    Knee actions:    [{np.min(actions[:, 4:]):.2f}, {np.max(actions[:, 4:]):.2f}]"
                )

            # Clear reward history
            for key in self.reward_components:
                self.reward_components[key] = []

            # Clear action history
            if hasattr(self, "episode_actions"):
                self.episode_actions = []

        # Reset simulation
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.timestep)

        # Reset robot to neutral pose with randomization
        default_joints = np.array(self.config["default_joint_positions"])
        noise = np.random.uniform(
            -self.config["noise_joint"], self.config["noise_joint"], size=8
        )
        initial_joints = default_joints + noise

        # Set motor positions
        for i, motor in enumerate(self.motors):
            if motor is not None:
                motor.setPosition(float(initial_joints[i]))

        # Let physics settle
        for _ in range(20):
            super().step(self.timestep)

        # Initialize tracking variables
        self.last_action = np.zeros(8)
        self.episode_steps = 0
        self.episode_actions = []  # Track actions for diagnostics

        # ============ CURRICULUM STAGE SELECTION ============
        # Update curriculum stage based on total timesteps
        ranges = self.config["command_ranges"]  # Default ranges
        if self.config.get("curriculum", {}).get("enabled", False):
            stages = self.config["curriculum"]["stages"]
            cumulative_steps = 0
            for i, stage in enumerate(stages):
                cumulative_steps += stage["timesteps"]
                if self.total_timesteps < cumulative_steps:
                    self.curriculum_stage = i
                    break

            # Use stage-specific command ranges
            current_stage = stages[self.curriculum_stage]
            ranges = current_stage["command_ranges"]

            print(
                f"[Curriculum Stage {self.curriculum_stage}: {current_stage['name']} | Total timesteps: {self.total_timesteps}]"
            )

        # Randomize command references for curriculum learning
        if (
            self.config.get("randomize_commands", False)
            and self.config["training_mode"]
        ):
            # TRAINING: Randomize commands using curriculum ranges
            self.command_v_x = np.random.uniform(*ranges["v_x"])
            self.command_v_y = np.random.uniform(*ranges["v_y"])
            self.command_w_z = np.random.uniform(*ranges["w_z"])
            self.command_z = np.random.uniform(*ranges["z"])  # Randomize height too!
            print(
                f"  Commands: v_x={self.command_v_x:.2f}, v_y={self.command_v_y:.2f}, w_z={self.command_w_z:.2f}, z={self.command_z:.2f}"
            )
        else:
            # INFERENCE: Use CONFIG defaults (will be updated dynamically in step())
            self.command_v_x = self.config["command_v_x_ref"]
            self.command_v_y = self.config["command_v_y_ref"]
            self.command_w_z = self.config["command_w_z_ref"]
            self.command_z = self.config["command_z_ref"]

        # Get initial observation
        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        """
        Execute one environment step
        """
        # Track total timesteps for curriculum learning
        if self.config["training_mode"]:
            self.total_timesteps += 1

        # Dynamic command updates in inference mode
        if not self.config["training_mode"]:
            interval = self.config.get("inference_command_interval", 500)
            if self.episode_steps > 0 and self.episode_steps % interval == 0:
                # Generate new random command every N steps
                ranges = self.config["command_ranges"]
                self.command_v_x = np.random.uniform(*ranges["v_x"])
                self.command_v_y = np.random.uniform(*ranges["v_y"])
                self.command_w_z = np.random.uniform(*ranges["w_z"])
                print(
                    f"[Step {self.episode_steps}] New command: v_x={self.command_v_x:.2f}, v_y={self.command_v_y:.2f}, w_z={self.command_w_z:.2f}"
                )

        # Apply action to motors
        self._apply_action(action)

        # Step simulation
        super().step(self.timestep)

        # Get new observation
        obs = self._get_obs()

        # Calculate reward
        reward, components = self._calculate_reward(obs, action)

        # Store reward components for diagnostics
        for key, value in components.items():
            self.reward_components[key].append(value)

        # Store action for diagnostics
        self.episode_actions.append(action.copy())

        # Check termination
        terminated = self._is_terminated(obs)

        # Update episode counter
        self.episode_steps += 1
        truncated = self.episode_steps >= self.config["max_episode_steps"]

        # Update last action
        self.last_action = action.copy()

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """
        Get observation vector (~36 dimensions)

        Returns:
        --------
        obs : np.array(36)
            [v_x, v_y, v_z] (3) - base linear velocities
            [w_x, w_y, w_z] (3) - base angular velocities
            [roll, pitch] (2) - base orientation (no yaw)
            [q_0..q_7] (8) - joint positions
            [q_dot_0..q_dot_7] (8) - joint velocities
            [a_{t-1}_0..a_{t-1}_7] (8) - previous actions
            [v_x_ref, v_y_ref, w_z_ref, z_ref] (4) - command references
        """
        # Get observation scales from config
        scales = self.config.get(
            "obs_scales",
            {"lin_vel": 1.0, "ang_vel": 1.0, "dof_pos": 1.0, "dof_vel": 1.0},
        )

        # Base velocities (SCALED)
        velocity = self.robot_node.getVelocity()
        lin_vel = np.array(velocity[0:3]) * scales["lin_vel"]  # [v_x, v_y, v_z] * 2.0
        ang_vel = np.array(velocity[3:6]) * scales["ang_vel"]  # [w_x, w_y, w_z] * 0.25

        # Base orientation (roll, pitch only - no yaw) - NO SCALING
        roll, pitch, _ = self._get_orientation()
        orientation = np.array([roll, pitch])

        # Joint positions and velocities (SCALED)
        joint_positions = np.zeros(8)
        joint_velocities = np.zeros(8)

        for i, motor in enumerate(self.motors):
            if motor is not None:
                # Get current joint state
                joint_positions[i] = (
                    motor.getTargetPosition() * scales["dof_pos"]
                )  # * 1.0
                # Approximate velocity from position change
                if hasattr(self, "joint_positions"):
                    dt = self.timestep / 1000.0
                    # Calculate velocity from raw positions (not scaled)
                    raw_pos = motor.getTargetPosition()
                    raw_prev_pos = (
                        self.joint_positions[i] / scales["dof_pos"]
                        if scales["dof_pos"] != 0
                        else self.joint_positions[i]
                    )
                    joint_velocities[i] = ((raw_pos - raw_prev_pos) / dt) * scales[
                        "dof_vel"
                    ]  # * 0.05

        # Store for next iteration
        self.joint_positions = joint_positions.copy()
        self.joint_velocities = joint_velocities.copy()

        # Command references (use instance variables for dynamic commands)
        commands = np.array(
            [
                self.command_v_x,
                self.command_v_y,
                self.command_w_z,
                self.command_z,
            ]
        )

        # ============ GAIT TRACKING: Track foot contacts ============
        # Get touch sensor readings
        contacts = np.zeros(4)
        for i, sensor in enumerate(self.touch_sensors):
            if sensor is not None:
                value = sensor.getValue()
                contacts[i] = 1.0 if value > 0.0 else 0.0

        # Track foot contact history for gait analysis
        for i in range(4):
            self.foot_contact_history[i].append(contacts[i])
            if len(self.foot_contact_history[i]) > 50:  # Keep last 50 steps (1 second)
                self.foot_contact_history[i].pop(0)

        # Concatenate all observations
        obs = np.concatenate(
            [
                lin_vel,  # 3
                ang_vel,  # 3
                orientation,  # 2
                joint_positions,  # 8
                joint_velocities,  # 8
                self.last_action,  # 8
                commands,  # 4
            ]
        )  # Total: 36

        # Handle NaN/Inf values
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs.astype(np.float32)

    def _apply_action(self, action):
        """
        Apply scaled actions to motors (simplified quadruped approach)

        Actions [-1, 1] are scaled by 0.25 rad and added to default positions.
        This provides smooth, stable control with fixed deviation limits.
        """
        default_pos = np.array(self.config["default_joint_positions"])

        for i, motor in enumerate(self.motors):
            if motor is not None:
                min_pos, max_pos = self.motor_limits[i]
                # Fixed scaling: ±0.25 rad from default position
                target_pos = default_pos[i] + action[i] * self.config["action_scale"]
                # Clip to joint limits
                target_pos = np.clip(target_pos, min_pos, max_pos)
                motor.setPosition(float(target_pos))

    def _calculate_reward(self, obs, action):
        """
        Calculate reward using exponential tracking + quadratic penalties

        Reward structure:
        - Exponential rewards for velocity/angular tracking
        - Quadratic penalties for deviations and instabilities

        NOTE: Observations are scaled for the policy, but rewards must be
        computed on real physical values, so we unscale them here.
        """
        # Get observation scales
        scales = self.config.get(
            "obs_scales",
            {"lin_vel": 1.0, "ang_vel": 1.0, "dof_pos": 1.0, "dof_vel": 1.0},
        )

        # Extract relevant observations (SCALED) and unscale for reward calculation
        lin_vel_scaled = obs[0:3]  # [v_x, v_y, v_z]
        ang_vel_scaled = obs[3:6]  # [w_x, w_y, w_z]
        orientation = obs[6:8]  # [roll, pitch] - not scaled
        joint_pos_scaled = obs[8:16]  # [q_0..q_7]
        commands = obs[28:32]  # [v_x_ref, v_y_ref, w_z_ref, z_ref] - not scaled

        # Unscale for reward calculation
        lin_vel = lin_vel_scaled / scales["lin_vel"]
        ang_vel = ang_vel_scaled / scales["ang_vel"]
        joint_pos = joint_pos_scaled / scales["dof_pos"]

        # Get actual height (not from observations)
        height = self.robot_node.getPosition()[2]

        # Get config parameters
        w = self.config["reward_weights"]
        sigma_vel = self.config["sigma_vel"]
        sigma_ang = self.config["sigma_ang"]
        default_joints = np.array(self.config["default_joint_positions"])

        # ============ TRACKING REWARDS (exponential) ============

        # 1. Linear velocity tracking (x-y plane)
        v_xy_ref = commands[0:2]  # [v_x_ref, v_y_ref]
        v_xy = lin_vel[0:2]  # [v_x, v_y]
        vel_error_sq = np.sum((v_xy_ref - v_xy) ** 2)
        R_lin_vel = np.exp(-vel_error_sq / sigma_vel)

        # 2. Angular velocity tracking (yaw rate)
        w_z_ref = commands[2]
        w_z = ang_vel[2]
        ang_error_sq = (w_z_ref - w_z) ** 2
        R_ang_vel = np.exp(-ang_error_sq / sigma_ang)

        # ============ PENALTIES (quadratic) ============

        # 3. Height penalty
        z_ref = commands[3]
        P_height = (height - z_ref) ** 2

        # 4. Pose penalty (stay near default joint positions)
        P_pose = np.sum((joint_pos - default_joints) ** 2)

        # 5. Action rate penalty (smooth actions)
        P_action_rate = np.sum((action - self.last_action) ** 2)

        # 6. Vertical velocity penalty (discourage bouncing)
        P_lin_vel_z = lin_vel[2] ** 2

        # 7. Orientation penalty (stay level)
        roll, pitch = orientation
        P_orientation = roll**2 + pitch**2


        # ============ TOTAL REWARD ============
        reward = (
            +w["lin_vel_tracking"] * R_lin_vel
            + w["ang_vel_tracking"] * R_ang_vel
            - w["height_penalty"] * P_height
            - w["pose_penalty"] * P_pose
            - w["action_rate_penalty"] * P_action_rate
            - w["lin_vel_z_penalty"] * P_lin_vel_z
            - w["orientation_penalty"] * P_orientation
        )

        # Components for diagnostics
        components = {
            "lin_vel_tracking": w["lin_vel_tracking"] * R_lin_vel,
            "ang_vel_tracking": w["ang_vel_tracking"] * R_ang_vel,
            "height_penalty": -w["height_penalty"] * P_height,
            "pose_penalty": -w["pose_penalty"] * P_pose,
            "action_rate_penalty": -w["action_rate_penalty"] * P_action_rate,
            "lin_vel_z_penalty": -w["lin_vel_z_penalty"] * P_lin_vel_z,
            "orientation_penalty": -w["orientation_penalty"] * P_orientation,
            "total": reward,
        }

        return reward, components

    def _is_terminated(self, obs):
        """
        Check termination conditions
        """
        # Extract orientation and height
        roll, pitch = obs[6:8]
        height = self.robot_node.getPosition()[2]

        # Terminate if too low (robot fell)
        if height < self.config["min_height"]:
            print(f"Terminated: Height ({height:.3f}m < {self.config['min_height']}m)")
            return True

        # Terminate if tilted too much
        max_roll = self.config["max_roll"]
        max_pitch = self.config["max_pitch"]

        if abs(roll) > max_roll or abs(pitch) > max_pitch:
            print(
                f"Terminated: Tilt (roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°)"
            )
            return True

        return False

    def _get_orientation(self):
        """
        Get robot orientation (roll, pitch, yaw) from orientation matrix
        """
        matrix = self.robot_node.getOrientation()
        R = np.array(matrix).reshape(3, 3)

        # Extract Euler angles
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

        return roll, pitch, yaw


# ============ TRAINING CALLBACK ============


class CurriculumLoggingCallback(BaseCallback):
    """
    Logging callback for curriculum training
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self):
        if self.locals.get("dones")[0]:
            info = self.locals["infos"][0]
            if "episode" in info:
                self.episode_count += 1
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

                if len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    max_reward = np.max(self.episode_rewards[-10:])
                    print(f"\n{'='*60}")
                    print(f"Episodes: {len(self.episode_rewards)}")
                    print(f"Avg Reward (last 10): {avg_reward:.2f}")
                    print(f"Max Reward (last 10): {max_reward:.2f}")
                    print(f"Avg Length (last 10): {avg_length:.1f}")
                    print(f"{'='*60}\n")

        return True


# ============ MAIN TRAINING LOOP ============


def main():
    config = CONFIG

    print("=" * 60)
    if config["training_mode"]:
        print("GHOST DOG CURRICULUM TRAINING")
    else:
        print("GHOST DOG CURRICULUM INFERENCE")
    print("=" * 60)

    # Create environment
    env = GhostDogCurriculumEnv(config=config)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    if config["training_mode"]:
        # ============ TRAINING MODE ============

        model_path = config["model_name"]
        vecnorm_path = config["vecnorm_name"]

        # Check for existing checkpoint
        model_exists = os.path.exists(f"{model_path}.zip")
        vecnorm_exists = os.path.exists(vecnorm_path)

        if model_exists and vecnorm_exists:
            # Resume training
            print("\n" + "=" * 60)
            print("RESUMING TRAINING")
            print(f"Loading: {model_path}.zip")
            print("=" * 60)

            env = VecNormalize.load(vecnorm_path, env)
            env.training = True
            env.norm_reward = True

            model = PPO.load(model_path, env=env)
            reset_num_timesteps = False

        else:
            # Start fresh training
            print("\n" + "=" * 60)
            print("STARTING FRESH TRAINING")
            print("=" * 60)

            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=config["ppo_learning_rate"],
                n_steps=config["ppo_n_steps"],
                batch_size=config["ppo_batch_size"],
                n_epochs=config["ppo_n_epochs"],
                gamma=config["ppo_gamma"],
                gae_lambda=config["ppo_gae_lambda"],
                clip_range=config["ppo_clip_range"],
                ent_coef=config["ppo_ent_coef"],
                tensorboard_log="./ghostdog_tensorboard/",
            )
            reset_num_timesteps = True

        print("\nStarting Training in FAST mode...")
        env.envs[0].unwrapped.simulationSetMode(
            env.envs[0].unwrapped.SIMULATION_MODE_FAST
        )

        callback = CurriculumLoggingCallback()

        try:
            model.learn(
                total_timesteps=config["total_timesteps"],
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=reset_num_timesteps,
            )

            print("\nTraining Finished!")
            model.save(model_path)
            env.save(vecnorm_path)
            print("Model saved!")

        except KeyboardInterrupt:
            print("\nInterrupted!")
            model.save(model_path + "_interrupted")
            env.save(vecnorm_path.replace(".pkl", "_interrupted.pkl"))
            print("Progress saved!")
            return

        # Test after training
        print("\nTesting in REAL TIME...")
        env.envs[0].unwrapped.simulationSetMode(
            env.envs[0].unwrapped.SIMULATION_MODE_REAL_TIME
        )
        env.training = False
        env.norm_reward = False

    else:
        # ============ INFERENCE MODE ============
        model_path = config["model_name"]
        vecnorm_path = config["vecnorm_name"]

        if not os.path.exists(f"{model_path}.zip"):
            print(f"\nERROR: Model file '{model_path}.zip' not found!")
            return

        print(f"\nLoading model: {model_path}")
        model = PPO.load(model_path, env=env)

        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)

        env.training = False
        env.norm_reward = False

        print("\nRunning in REAL TIME...")
        env.envs[0].unwrapped.simulationSetMode(
            env.envs[0].unwrapped.SIMULATION_MODE_REAL_TIME
        )

    # ============ EVALUATION LOOP ============
    obs = env.reset()
    episode_reward = 0
    episode_length = 0

    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            episode_reward += reward[0]
            episode_length += 1

            if done[0]:
                print(
                    f"\nEpisode: Length={episode_length}, Reward={episode_reward:.2f}"
                )
                obs = env.reset()
                episode_reward = 0
                episode_length = 0

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
