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
TRAINING_MODE = True  # Set to False for inference/evaluation mode
MODEL = "ppo_ghostdog_v05"  # v03: Zombie gait fixes
VECNORM = "vec_normalize_ghostdog_v05.pkl"
# ============================================


class GhostDogEnv(Supervisor, gym.Env):
    """
    NATURAL GAIT VERSION - Fixes unnatural walking behavior

    === ZOMBIE GAIT FIX (v03) ===
    Addresses degenerate "zombie gait" (single-leg dragging) via:
    1. Tighter termination (height: 0.20m, tilt: 50°, stuck detection)
    2. Feet air time rewards (encourages rhythmic lifting ~0.4s)
    3. Symmetry penalties (penalizes asymmetric leg usage)
    4. Torque regularization (energy efficiency)
    5. Joint acceleration penalties (smoother motion)

    Original features:
    - Strong penalty for pitch (rearing up)
    - Reward for keeping all feet on ground
    - Reward for natural body orientation
    - Action smoothness penalties
    """

    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())

        self.robot_node = self.getFromDef("GHOST_DOG")
        if self.robot_node is None:
            sys.exit("ERROR: Ghost Dog robot node not found!")

        # Get motors
        hip_motor_names = ["hip0", "hip1", "hip2", "hip3"]
        self.hip_motors = []
        self.hip_motor_limits = []

        for name in hip_motor_names:
            motor = self.getDevice(name)
            if motor is None:
                sys.exit(f"ERROR: Motor '{name}' not found!")
            motor.setPosition(0)
            motor.setVelocity(motor.getMaxVelocity())

            # Enable torque feedback for energy regularization
            motor.enableTorqueFeedback(self.timestep)

            self.hip_motors.append(motor)

            min_pos = motor.getMinPosition()
            max_pos = motor.getMaxPosition()
            self.hip_motor_limits.append((min_pos, max_pos))

        print(f"✓ Initialized {len(self.hip_motors)} hip motors")

        # Spine motor
        self.spine_motor = self.getDevice("spine")
        if self.spine_motor:
            self.spine_motor.setPosition(0)

        # Touch sensors
        touch_sensor_names = [
            "touch0",  # Front-right foot
            "touch1",  # Rear-left foot
            "touch2",  # Front-left foot
            "touch3",  # Rear-right foot
        ]
        self.touch_sensors = []
        for name in touch_sensor_names:
            sensor = self.getDevice(name)
            if sensor is not None:
                sensor.enable(self.timestep)
                self.touch_sensors.append(sensor)

        if len(self.touch_sensors) > 0:
            print(f"✓ Initialized {len(self.touch_sensors)} touch sensors")
        else:
            print("WARNING: No touch sensors - foot contact rewards disabled")

        # Spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )

        # Tracking
        self.last_hip_positions = None
        self.last_action = None
        self.last_hip_velocities = None  # For acceleration penalty
        self.episode_steps = 0
        self.max_episode_steps = 500
        self.initial_x = 0.0
        self.last_x = 0.0

        # Feet air time tracking (for rhythmic gait rewards)
        self.feet_air_time = np.zeros(4)  # Time each foot has been in air
        self.feet_contact_history = np.zeros(4)  # Previous contact state

        # Reward diagnostics
        self.reward_components = {
            "velocity": [],
            "alive": [],
            "survival": [],
            "pitch": [],
            "roll": [],
            "contacts": [],
            "air_time": [],  # NEW: Feet air time reward
            "symmetry": [],  # NEW: Symmetry penalty
            "smoothness": [],
            "torque": [],  # NEW: Torque penalty
            "acceleration": [],  # NEW: Joint acceleration penalty
            "total": [],
        }

        print(f"\n{'='*60}")
        print("NATURAL GAIT VERSION")
        print("Addresses: Walking on back feet, front lifted")
        print(f"{'='*60}\n")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Print reward breakdown from last episode
        if len(self.reward_components["total"]) > 0:
            print(f"\nEpisode Stats (last episode):")
            print(f"  Velocity:   {np.mean(self.reward_components['velocity']):.2f}")
            print(f"  Survival:   {np.mean(self.reward_components['survival']):.2f}")
            print(f"  Pitch pen:  {np.mean(self.reward_components['pitch']):.2f}")
            print(f"  Roll pen:   {np.mean(self.reward_components['roll']):.2f}")
            print(f"  Contacts:   {np.mean(self.reward_components['contacts']):.2f}")
            print(f"  Air Time:   {np.mean(self.reward_components['air_time']):.2f}")
            print(f"  Symmetry:   {np.mean(self.reward_components['symmetry']):.2f}")
            print(f"  Smoothness: {np.mean(self.reward_components['smoothness']):.2f}")
            print(f"  Torque:     {np.mean(self.reward_components['torque']):.2f}")
            print(
                f"  Accel:      {np.mean(self.reward_components['acceleration']):.2f}"
            )
            print(f"  Total:      {np.mean(self.reward_components['total']):.2f}")

            for key in self.reward_components:
                self.reward_components[key] = []

        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.timestep)

        neutral_positions = [0.1, -0.2, 0.1, -0.2]
        for motor, pos in zip(self.hip_motors, neutral_positions):
            motor.setPosition(pos)

        if self.spine_motor:
            self.spine_motor.setPosition(0)

        for _ in range(20):
            super().step(self.timestep)

        translation = self.robot_node.getPosition()
        self.initial_x = translation[0]
        self.last_x = self.initial_x

        self.last_hip_positions = np.array(
            [motor.getTargetPosition() for motor in self.hip_motors]
        )
        self.last_action = None
        self.last_hip_velocities = None
        self.episode_steps = 0

        # Reset air time tracking
        self.feet_air_time = np.zeros(4)
        self.feet_contact_history = np.zeros(4)

        obs = self._get_obs()
        print(f"Reset: Height={translation[2]:.3f}m")

        return obs, {}

    def step(self, action):
        # Apply actions
        for i, (motor, act) in enumerate(zip(self.hip_motors, action)):
            min_pos, max_pos = self.hip_motor_limits[i]
            target_pos = min_pos + (act + 1) / 2 * (max_pos - min_pos)
            motor.setPosition(target_pos)

        super().step(self.timestep)
        obs = self._get_obs()

        # Update feet air time tracking
        contacts = obs[17:21]  # Foot contacts
        dt = self.timestep / 1000.0
        for i in range(4):
            if contacts[i] == 0:  # Foot in air
                self.feet_air_time[i] += dt
            else:  # Foot on ground
                self.feet_air_time[i] = 0.0

        # Calculate reward
        reward, components = self._calculate_reward(obs, action)

        # Store for diagnostics
        for key, value in components.items():
            self.reward_components[key].append(value)

        # Check termination
        terminated = self._is_terminated(obs)

        self.episode_steps += 1
        truncated = self.episode_steps >= self.max_episode_steps

        # Update tracking
        translation = self.robot_node.getPosition()
        self.last_x = translation[0]
        self.last_hip_positions = obs[9:13]
        self.last_action = action.copy()

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        roll, pitch, yaw = self._get_orientation()
        angular_vel = self.robot_node.getVelocity()[3:6]
        linear_vel = self.robot_node.getVelocity()[0:3]

        hip_positions = np.array(
            [motor.getTargetPosition() for motor in self.hip_motors]
        )

        dt = self.timestep / 1000.0
        if self.last_hip_positions is not None:
            hip_velocities = (hip_positions - self.last_hip_positions) / dt
        else:
            hip_velocities = np.zeros(4)

        if len(self.touch_sensors) > 0:
            contacts = np.array(
                [1.0 if sensor.getValue() > 0 else 0.0 for sensor in self.touch_sensors]
            )
        else:
            contacts = np.zeros(4)

        translation = self.robot_node.getPosition()
        height = translation[2]
        distance_traveled = translation[0] - self.initial_x
        time_normalized = self.episode_steps / self.max_episode_steps
        spine_angle = self.spine_motor.getTargetPosition() if self.spine_motor else 0.0

        obs = np.concatenate(
            [
                [roll, pitch, yaw],
                angular_vel,
                linear_vel,
                hip_positions,
                hip_velocities,
                contacts,
                [time_normalized],
                [height],
                [distance_traveled],
                [spine_angle],
            ]
        )

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs.astype(np.float32)

    def _get_orientation(self):
        matrix = self.robot_node.getOrientation()
        R = np.array(matrix).reshape(3, 3)
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return roll, pitch, yaw

    def _calculate_reward(self, obs, action):
        """
        IMPROVED REWARD FOR NATURAL GAIT - ZOMBIE GAIT FIX

        New additions to prevent zombie/crawling gait:
        1. Feet air time reward - encourages rhythmic lifting
        2. Symmetry penalty - penalizes asymmetric leg usage
        3. Torque penalty - encourages energy efficiency
        4. Joint acceleration penalty - smoother motion

        Original features:
        - Strong pitch/roll penalties
        - Foot contact rewards
        - Action smoothness penalty
        """
        roll, pitch, yaw = obs[0], obs[1], obs[2]
        lin_vel_x = obs[6]  # Forward velocity
        lin_vel_z = obs[8]  # Vertical velocity
        contacts = obs[
            17:21
        ]  # Foot contacts [touch0=FR, touch1=BL, touch2=FL, touch3=BR]
        hip_velocities = obs[13:17]  # Joint velocities
        height = self.robot_node.getPosition()[2]
        dt = self.timestep / 1000.0

        # ============ REWARD COMPONENTS ============

        # 1. FORWARD VELOCITY - Main goal
        velocity_reward = 2.0 * lin_vel_x

        # 2. ALIVE BONUS - Staying upright
        alive_bonus = 1.0

        # 3. SURVIVAL BONUS - Living longer
        survival_bonus = 0.05 * self.episode_steps

        # 4. PITCH PENALTY
        pitch_penalty = -2.0 * abs(pitch)

        # 5. ROLL PENALTY
        roll_penalty = -0.2 * abs(roll)

        # 6. FOOT CONTACT REWARDS
        if len(self.touch_sensors) > 0:
            num_contacts = np.sum(contacts)

            # Basic contact reward (scaled by velocity)
            if lin_vel_x > 0.5:
                # Running gait: 2-3 feet okay
                if num_contacts >= 2:
                    contact_bonus = 0.5
                else:
                    contact_bonus = -0.5
            else:
                # Walking gait: 3-4 feet preferred
                if num_contacts >= 3:
                    contact_bonus = 0.5
                else:
                    contact_bonus = 0.0

            # Front foot bonus
            front_contacts = contacts[0] + contacts[2]  # FR + FL
            if front_contacts >= 1:
                front_foot_bonus = 0.5
            else:
                front_foot_bonus = -1.0

            total_contact_reward = contact_bonus + front_foot_bonus
        else:
            total_contact_reward = 0.0

        # 7. NEW: FEET AIR TIME REWARD
        # Reward feet spending ~0.3-0.5s in air (rhythmic gait)
        air_time_reward = 0.0
        target_air_time = 0.4  # 0.4 seconds
        if len(self.touch_sensors) > 0:
            for i in range(4):
                # Reward when foot completes a lift cycle (just landed)
                if contacts[i] > 0 and self.feet_contact_history[i] == 0:
                    time_in_air = self.feet_air_time[i]
                    # Gaussian reward centered on target_air_time
                    air_time_reward += np.exp(
                        -((time_in_air - target_air_time) ** 2) / 0.1
                    )

            air_time_reward *= 1.0  # Scale factor
            self.feet_contact_history = contacts.copy()

        # 8. NEW: SYMMETRY PENALTY
        # Penalize asymmetric leg usage (prevents zombie gait)
        symmetry_penalty = 0.0
        if len(self.touch_sensors) > 0:
            # Left vs right contacts (touch1=BL, touch2=FL vs touch0=FR, touch3=BR)
            left_contacts = contacts[1] + contacts[2]  # BL + FL
            right_contacts = contacts[0] + contacts[3]  # FR + BR

            # Penalize imbalance
            contact_imbalance = abs(left_contacts - right_contacts)
            symmetry_penalty = -0.5 * contact_imbalance

            # STRONG penalty for using <2 feet (zombie gait)
            if num_contacts < 2:
                symmetry_penalty -= 2.0

        # 9. Vertical velocity penalty
        z_vel_penalty = -0.1 * abs(lin_vel_z)

        # 10. ACTION SMOOTHNESS PENALTY
        if self.last_action is not None:
            action_diff = np.abs(action - self.last_action)
            action_rate_penalty = -1.0 * np.sum(action_diff)
        else:
            action_rate_penalty = 0.0

        # 11. NEW: TORQUE PENALTY
        # Enable torque feedback on motors in __init__ if not already enabled
        torque_penalty = 0.0
        try:
            torques = np.array([motor.getTorqueFeedback() for motor in self.hip_motors])
            torque_penalty = -0.0002 * np.sum(torques**2)
        except:
            # If torque feedback not available, skip
            torque_penalty = 0.0

        # 12. NEW: JOINT ACCELERATION PENALTY
        acceleration_penalty = 0.0
        if self.last_hip_velocities is not None:
            hip_accelerations = (hip_velocities - self.last_hip_velocities) / dt
            acceleration_penalty = -2.5e-7 * np.sum(hip_accelerations**2)

        self.last_hip_velocities = hip_velocities.copy()

        # ============ TOTAL REWARD ============
        total_reward = (
            velocity_reward
            + alive_bonus
            + survival_bonus
            + pitch_penalty
            + roll_penalty
            + total_contact_reward
            + air_time_reward  # NEW
            + symmetry_penalty  # NEW
            + z_vel_penalty
            + action_rate_penalty
            + torque_penalty  # NEW
            + acceleration_penalty  # NEW
        )

        # Components for diagnostics
        components = {
            "velocity": velocity_reward,
            "alive": alive_bonus,
            "survival": survival_bonus,
            "pitch": pitch_penalty,
            "roll": roll_penalty,
            "contacts": total_contact_reward,
            "air_time": air_time_reward,  # NEW
            "symmetry": symmetry_penalty,  # NEW
            "smoothness": action_rate_penalty,
            "torque": torque_penalty,  # NEW
            "acceleration": acceleration_penalty,  # NEW
            "total": total_reward,
        }

        return total_reward, components

    def _is_terminated(self, obs):
        """
        Check termination conditions

        Updated termination conditions to prevent zombie/crawling gaits:
        - Higher height threshold (0.20m vs 0.10m)
        - Tighter tilt limits (50° vs 70°)
        - Stuck detection (not moving forward)
        """
        roll, pitch, yaw = obs[0], obs[1], obs[2]
        height = self.robot_node.getPosition()[2]
        lin_vel_x = obs[6]  # Forward velocity

        # Terminate if tilted too much (tightened from 70° to 50°)
        tilt_limit = 0.873  # ~50 degrees (was 1.22 = 70°)
        if abs(roll) > tilt_limit or abs(pitch) > tilt_limit:
            print(
                f"Terminated: Tilt (roll={np.degrees(roll):.1f}°, pitch={np.degrees(pitch):.1f}°)"
            )
            return True

        # Terminate if too low (raised from 0.10m to 0.20m to prevent dragging)
        min_height = 0.20  # Was 0.10 - zombie gait could drag at 0.11-0.15m
        if height < min_height:
            print(f"Terminated: Height ({height:.3f}m < {min_height}m)")
            return True

        # Terminate if stuck (not making forward progress)
        if self.episode_steps > 50:  # After initial settling period
            distance_moved = self.robot_node.getPosition()[0] - self.last_x
            if abs(lin_vel_x) < 0.05 and abs(distance_moved) < 0.01:
                print(
                    f"Terminated: Robot stuck (v={lin_vel_x:.3f}, dist={distance_moved:.3f})"
                )
                return True

        return False


class LoggingCallback(BaseCallback):
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
                    max_length = np.max(self.episode_lengths[-10:])
                    print(f"\n{'='*60}")
                    print(f"Episodes: {len(self.episode_rewards)}")
                    print(f"Avg Reward (last 10): {avg_reward:.2f}")
                    print(f"Avg Length (last 10): {avg_length:.1f}")
                    print(f"Max Length (last 10): {max_length}")
                    print(f"{'='*60}\n")

        return True


def main():
    print("=" * 60)
    if TRAINING_MODE:
        print("Ghost Dog - TRAINING MODE")
    else:
        print("Ghost Dog - INFERENCE MODE")
    print("=" * 60)

    # Setup environment
    env = GhostDogEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    if TRAINING_MODE:
        # ============ TRAINING MODE ============

        # Check if we should resume training or start fresh
        model_exists = os.path.exists(f"{MODEL}.zip")
        vecnorm_exists = os.path.exists(VECNORM)

        if model_exists and vecnorm_exists:
            # Resume training from saved checkpoint
            print("\n" + "=" * 60)
            print("RESUMING TRAINING FROM SAVED MODEL")
            print(f"Loading: {MODEL}.zip")
            print(f"Loading: {VECNORM}")
            print("=" * 60)

            # Load VecNormalize wrapper with existing stats
            env = VecNormalize.load(VECNORM, env)
            env.training = True
            env.norm_reward = True

            # Load saved PPO model
            model = PPO.load(MODEL, env=env)

            # Set to continue TensorBoard logging from previous checkpoint
            reset_num_timesteps = False
            print("TensorBoard will continue from previous timestep count")

        else:
            # Start fresh training
            print("\n" + "=" * 60)
            print("NO SAVED MODEL FOUND - STARTING FRESH TRAINING")
            print("=" * 60)

            # Initialize VecNormalize wrapper
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

            # Initialize fresh PPO model
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                tensorboard_log="./ghostdog_tensorboard/",
            )

            # Start TensorBoard logging from timestep 0
            reset_num_timesteps = True

        print("\nStarting Training in FAST mode...")
        env.envs[0].unwrapped.simulationSetMode(
            env.envs[0].unwrapped.SIMULATION_MODE_FAST
        )

        callback = LoggingCallback()

        try:
            model.learn(
                total_timesteps=900000,
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=reset_num_timesteps,  # Continue or restart TensorBoard
            )

            print("\nTraining Finished!")
            model.save(MODEL)
            env.save(VECNORM)
            print("Model saved!")

        except KeyboardInterrupt:
            print("\nInterrupted!")
            model.save(MODEL + "_interrupted")
            env.save(VECNORM.replace(".pkl", "_interrupted.pkl"))
            print("Progress saved!")
            return

        # Test after training
        print("\nTesting trained model in REAL TIME...")
        env.envs[0].unwrapped.simulationSetMode(
            env.envs[0].unwrapped.SIMULATION_MODE_REAL_TIME
        )
        env.training = False
        env.norm_reward = False

    else:
        # ============ INFERENCE MODE ============
        model_path = MODEL
        vec_normalize_path = VECNORM

        # Check if model files exist
        if not os.path.exists(f"{model_path}.zip"):
            print(f"\nERROR: Model file '{model_path}.zip' not found!")
            print("Please train a model first by setting TRAINING_MODE = True")
            return

        if not os.path.exists(vec_normalize_path):
            print(f"\nWARNING: VecNormalize file '{vec_normalize_path}' not found!")
            print("Continuing without normalization stats...")
            vec_normalize_path = None

        print(f"\nLoading model from '{model_path}'...")
        model = PPO.load(model_path, env=env)

        if vec_normalize_path:
            print(f"Loading VecNormalize stats from '{vec_normalize_path}'...")
            env = VecNormalize.load(vec_normalize_path, env)

        print("Model loaded successfully!")
        print("\nRunning in REAL TIME mode...")
        env.envs[0].unwrapped.simulationSetMode(
            env.envs[0].unwrapped.SIMULATION_MODE_REAL_TIME
        )
        env.training = False
        env.norm_reward = False

    # ============ EVALUATION LOOP (both modes) ============
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
