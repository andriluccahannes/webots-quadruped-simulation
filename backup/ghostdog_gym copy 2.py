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
# All values are matched to the reference go2_train.py / go2_env.py
# where applicable, adapted for Ghost Dog's 8-DOF structure.
CONFIG = {
    # Add these new lines:
    "kp": 4,  # Stiffness (Position gain) - Start LOW for small robots
    "ki": 0.0,  # Integral gain - Usually keep at 0 for walking
    "kd": 0.1,  # Damping (Velocity gain) - Stops oscillation
    # ---- Mode ----
    "training_mode": True,  # False for inference
    "model_name": "ppo_ghostdog_v39",
    "vecnorm_name": "vec_normalize_v39.pkl",
    # ---- Episode (ref: episode_length_s=20, dt=0.02 → 1000 steps) ----
    "max_episode_steps": 1000,
    "dt": 0.02,  # 50 Hz control loop
    # ---- Command resampling during episodes (ref: resampling_time_s=4.0) ----
    "resampling_time_s": 4.0,  # resample commands every 4 s → every 200 steps
    # ---- Observation scaling (ref: go2_train.py obs_cfg) ----
    "obs_scales": {
        "lin_vel": 2.0,  # used for command scaling only (lin vel NOT in obs)
        "ang_vel": 0.25,  # angular velocity scale
        "dof_pos": 1.0,  # joint position residual scale
        "dof_vel": 0.05,  # joint velocity scale (critical – tames noisy vel)
    },
    # ---- Action (ref: action_scale=0.25, clip_actions=100) ----
    "action_scale": 0.2,  # NN output * 0.25 + default = target pos
    "clip_actions": 100.0,  # effectively no clip on raw NN output
    "simulate_action_latency": True,  # 1-step delay (ref: True)
    # ---- Default joint positions ----
    # [hip0, hip1, hip2, hip3, knee0, knee1, knee2, knee3]
    "default_joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # ---- Reward weights (ref: go2_train.py reward_cfg) ----
    # Signs are baked in: positive = reward, negative = penalty.
    # All weights are multiplied by dt inside the env (ref does the same).
    "reward_weights": {
        "tracking_lin_vel": 10.0,  # exp(−error/σ)  – track vx, vy
        "tracking_ang_vel": 0.2,  # exp(−error/σ)  – track wz
        "lin_vel_z": -1.0,  # vz²            – no bouncing
        "base_height": -50.0,  # (z−z_cmd)²     – STRONG height lock
        "action_rate": -0.3,  # Σ(a_t − a_{t−1})²  – smoothness
        "similar_to_default": -0.1,  # Σ|q − q_def|   – stay near default (L1)
    },
    "tracking_sigma": 0.25,  # ref value (tighter than your old σ=1.0)
    "base_height_target": 0.25,  # used on reset to set initial height cmd
    # ---- Command ranges (ref: go2_train.py command_cfg) ----
    "randomize_commands": True,
    "command_ranges": {
        "v_x": (-1.0, 2.0),
        "v_y": (0.0, 0.0),
        "w_z": (0.0, 0.0),
        "z": (0.16, 0.24),
    },
    # ---- Termination (ref: 10 rad roll/pitch → effectively disabled) ----
    # The reference relies on the −50 height penalty, NOT hard termination.
    "termination_if_roll_greater_than": 0.6,  # radians (≈ never)
    "termination_if_pitch_greater_than": 0.6,  # radians (≈ never)
    # NOTE: no min_height termination – matches reference
    # ---- PPO hyperparameters (ref: go2_train.py) ----
    "ppo_learning_rate": 1e-3,
    "ppo_n_steps": 2048,  # single-env, so large rollout
    "ppo_batch_size": 64,
    "ppo_n_epochs": 5,  # ref: 5
    "ppo_gamma": 0.99,
    "ppo_gae_lambda": 0.95,
    "ppo_clip_range": 0.2,
    "ppo_ent_coef": 0.01,
    # ---- Network architecture (ref: [512, 256, 128] for actor & critic) ----
    "policy_kwargs": {
        "net_arch": dict(pi=[512, 256, 128], vf=[512, 256, 128]),
    },
    # ---- Training budget ----
    "total_timesteps": 500_000,
    # ---- Inference defaults ----
    "command_v_x_ref": 1.0,
    "command_v_y_ref": 0.0,
    "command_w_z_ref": 0.0,
    "command_z_ref": 0.25,
    "inference_command_interval": 200,
    # ---- Reset noise ----
    "noise_joint": 0.1,
}
# ========================================


class GhostDogEnv(Supervisor, gym.Env):
    """
     Ghost Dog Walking Environment — Reference-Aligned
     ==================================================
     Structural changes from the previous version, all taken from go2_env.py:

     1. ACTUAL joint positions via PositionSensor (was: motor.getTargetPosition)
     2. Joint velocities by differentiating actual positions (was: target deltas)
     3. Projected gravity vector replaces roll/pitch in observation
     4. Base linear velocity REMOVED from observations (ref: proprioception only)
     5. 1-step action latency (was: immediate execution)
     6. Residual joint positions (q − q_default) in observation
     7. Observation scaling matched to reference
     8. Reward weights matched to reference (height −50, σ=0.25, L1 pose)
     9. Reward × dt scaling (ref: reward_scales *= dt)
    10. No orientation penalty (ref: not used)
    11. No height/orientation hard termination (ref: effectively disabled)
    12. Command resampling every 4 s during episodes (ref: resampling_time_s)
    13. Network [512, 256, 128] (was: SB3 default [64, 64])

     Observation (34D):
         ang_vel_body(3) + projected_gravity(3) + commands_scaled(4)
         + dof_pos_residual(8) + dof_vel(8) + current_actions(8)
    """

    def __init__(self, config=CONFIG):
        super().__init__()
        self.config = config
        self.timestep = int(self.getBasicTimeStep())
        self.dt = config["dt"]

        # Robot node (Supervisor access for state queries)
        self.robot_node = self.getFromDef("GHOST_DOG")
        if self.robot_node is None:
            sys.exit("ERROR: Ghost Dog robot node not found!")

        # ============ MOTORS + POSITION SENSORS ============
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
        self.num_actions = len(motor_names)
        self.motors = []
        self.position_sensors = []
        self.motor_limits = []

        for name in motor_names:
            motor = self.getDevice(name)
            if motor is None:
                print(f"WARNING: Motor '{name}' not found!")
                self.motors.append(None)
                self.position_sensors.append(None)
                self.motor_limits.append((-1.0, 1.0))
            else:
                motor.setPosition(0)
                motor.setVelocity(motor.getMaxVelocity())
                motor.setControlPID(config["kp"], config["ki"], config["kd"])
                self.motors.append(motor)
                self.motor_limits.append(
                    (motor.getMinPosition(), motor.getMaxPosition())
                )
                # ★ KEY CHANGE 1: enable position sensor for ACTUAL readings
                ps = motor.getPositionSensor()
                if ps is not None:
                    ps.enable(self.timestep)
                self.position_sensors.append(ps)

        n_motors = sum(1 for m in self.motors if m is not None)
        n_sensors = sum(1 for s in self.position_sensors if s is not None)
        print(
            f"✓ Motors: {n_motors}/{self.num_actions} | "
            f"PositionSensors: {n_sensors}/{self.num_actions}"
        )

        # ============ TOUCH SENSORS (kept for optional diagnostics) ============
        self.touch_sensors = []
        for name in ["touch0", "touch1", "touch2", "touch3"]:
            ts = self.getDevice(name)
            if ts is not None:
                ts.enable(self.timestep)
                self.touch_sensors.append(ts)

        # ============ SPACES ============
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )
        # Observation: 3+3+4+8+8+8 = 34
        self.num_obs = 34
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )

        # ============ DEFAULT JOINT POSITIONS ============
        self.default_dof_pos = np.array(
            config["default_joint_positions"], dtype=np.float32
        )

        # ============ COMMAND SCALING (ref: commands_scale tensor) ============
        s = config["obs_scales"]
        self.commands_scale = np.array(
            [s["lin_vel"], s["lin_vel"], s["ang_vel"], s["lin_vel"]],
            dtype=np.float32,
        )

        # ============ STATE BUFFERS ============
        self.actions = np.zeros(self.num_actions, dtype=np.float32)
        self.last_actions = np.zeros(self.num_actions, dtype=np.float32)
        self.dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        self.dof_vel = np.zeros(self.num_actions, dtype=np.float32)
        self.prev_dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        self.commands = np.zeros(4, dtype=np.float32)
        self.episode_steps = 0
        self.total_timesteps_count = 0

        # ============ REWARD SCALES × dt (ref: reward_scales[name] *= dt) ====
        self.reward_scales = {}
        for name, scale in config["reward_weights"].items():
            self.reward_scales[name] = scale * self.dt

        # ============ DIAGNOSTICS ============
        self.reward_sums = {name: 0.0 for name in config["reward_weights"]}
        self.reward_sums["total"] = 0.0

        print(f"\n{'='*60}")
        print("GHOST DOG ENV (Reference-Aligned)")
        print(
            f"  Obs:  {self.num_obs}D  "
            f"[ang_vel(3) gravity(3) cmd(4) dof_pos(8) dof_vel(8) act(8)]"
        )
        print(
            f"  Act:  {self.num_actions}D  scale={config['action_scale']}  "
            f"latency={config['simulate_action_latency']}"
        )
        print(
            f"  σ_tracking={config['tracking_sigma']}  "
            f"height_w={config['reward_weights']['base_height']}"
        )
        print(f"  Network: {config['policy_kwargs']['net_arch']}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ---- Print episode summary ----
        if self.episode_steps > 0:
            print(f"\nEpisode summary (steps={self.episode_steps}):")
            for name in self.config["reward_weights"]:
                avg = self.reward_sums[name] / max(self.episode_steps, 1)
                print(f"  {name:25s}: {avg:+.5f} /step")
            avg_total = self.reward_sums["total"] / max(self.episode_steps, 1)
            print(f"  {'TOTAL':25s}: {avg_total:+.5f} /step")
            print(f"  {'EPISODE TOTAL':25s}: {self.reward_sums['total']:+.2f}")

        # Clear diagnostic accumulators
        for k in self.reward_sums:
            self.reward_sums[k] = 0.0

        # ---- Reset simulation ----
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.timestep)

        # ---- Randomized initial joint positions (ref: noise on reset) ----
        noise = np.random.uniform(
            -self.config["noise_joint"],
            self.config["noise_joint"],
            size=self.num_actions,
        )
        initial_pos = self.default_dof_pos + noise
        for i, motor in enumerate(self.motors):
            if motor is not None:
                target = np.clip(initial_pos[i], *self.motor_limits[i])
                motor.setPosition(float(target))

        # Let physics settle
        for _ in range(20):
            super().step(self.timestep)

        # ---- Reset buffers ----
        self.actions[:] = 0.0
        self.last_actions[:] = 0.0
        self.episode_steps = 0

        # Read actual joint state after settling
        self._read_joint_state()
        self.prev_dof_pos[:] = self.dof_pos[:]
        self.dof_vel[:] = 0.0  # no velocity at start

        # ---- Sample commands (ref: _sample_commands on reset) ----
        if self.config["randomize_commands"] and self.config["training_mode"]:
            self._sample_commands()
        else:
            self.commands[:] = [
                self.config["command_v_x_ref"],
                self.config["command_v_y_ref"],
                self.config["command_w_z_ref"],
                self.config["command_z_ref"],
            ]
        # ref: on reset, set height cmd to base_height_target
        self.commands[3] = self.config["base_height_target"]

        return self._build_obs(), {}

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------
    def step(self, action):
        if self.config["training_mode"]:
            self.total_timesteps_count += 1

        # ---- Dynamic commands in inference ----
        if not self.config["training_mode"]:
            interval = self.config.get("inference_command_interval", 200)
            if self.episode_steps > 0 and self.episode_steps % interval == 0:
                r = self.config["command_ranges"]
                self.commands[0] = np.random.uniform(*r["v_x"])
                self.commands[1] = np.random.uniform(*r["v_y"])
                self.commands[2] = np.random.uniform(*r["w_z"])
                print(
                    f"  [Step {self.episode_steps}] cmd: " f"vx={self.commands[0]:.2f}"
                )

        # ============ ACTION PROCESSING (ref: go2_env.step) ============
        # ★ KEY CHANGE 5: store current action, but EXECUTE last action
        self.actions = np.clip(
            action, -self.config["clip_actions"], self.config["clip_actions"]
        ).astype(np.float32)

        if self.config["simulate_action_latency"]:
            exec_actions = self.last_actions  # 1-step delay
        else:
            exec_actions = self.actions

        # ref: target = exec_actions * action_scale + default_dof_pos
        target_dof_pos = (
            exec_actions * self.config["action_scale"] + self.default_dof_pos
        )
        for i, motor in enumerate(self.motors):
            if motor is not None:
                t = np.clip(target_dof_pos[i], *self.motor_limits[i])
                motor.setPosition(float(t))

        # ---- Step simulation ----
        super().step(self.timestep)
        self.episode_steps += 1

        # ---- Read actual joint state ----
        self._read_joint_state()

        # ---- Command resampling mid-episode (ref: resampling_time_s) ----
        resample_every = int(self.config["resampling_time_s"] / self.dt)
        if (
            self.config["training_mode"]
            and self.config["randomize_commands"]
            and self.episode_steps % resample_every == 0
        ):
            self._sample_commands()

        # ---- Compute reward ----
        reward = self._compute_reward()

        # ---- Termination (ref: only roll/pitch/episode length) ----
        roll, pitch, _ = self._get_orientation()
        terminated = False
        if abs(roll) > self.config["termination_if_roll_greater_than"]:
            terminated = True
        if abs(pitch) > self.config["termination_if_pitch_greater_than"]:
            terminated = True

        truncated = self.episode_steps >= self.config["max_episode_steps"]

        # ---- Observation ----
        obs = self._build_obs()

        # ---- Update buffers (ref: last_actions = actions at end) ----
        self.last_actions[:] = self.actions[:]
        self.prev_dof_pos[:] = self.dof_pos[:]

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------
    def _read_joint_state(self):
        """
        ★ KEY CHANGE 1 + 2: read ACTUAL joint positions from PositionSensor
        and compute velocities by differentiating real readings.
        (was: motor.getTargetPosition → only sees what was commanded)
        """
        for i, sensor in enumerate(self.position_sensors):
            if sensor is not None:
                self.dof_pos[i] = sensor.getValue()
            elif self.motors[i] is not None:
                # fallback – should not happen if proto is correct
                self.dof_pos[i] = self.motors[i].getTargetPosition()

        # Finite-difference velocity (ref: simulator gives this directly)
        self.dof_vel[:] = (self.dof_pos - self.prev_dof_pos) / self.dt

    def _build_obs(self):
        """
        ★ KEY CHANGE 3, 4, 6, 7: observation vector (34D)

        Matches reference go2_env.py observation structure:
            [base_ang_vel * 0.25,             # 3  body-frame
             projected_gravity,               # 3  gravity in body frame
             commands * commands_scale,        # 4  scaled commands
             (dof_pos − default) * 1.0,       # 8  residual joint pos
             dof_vel * 0.05,                  # 8  joint velocities
             actions]                          # 8  current NN output

        Changes from old version:
        - REMOVED base linear velocity (ref does not observe it)
        - REMOVED roll/pitch → replaced by projected_gravity (richer, no gimbal)
        - Joint positions are now RESIDUAL from default (not raw)
        - Actual positions from sensors (not target positions)
        - Obs includes CURRENT actions (not last_actions)
        """
        scales = self.config["obs_scales"]

        # 1. Angular velocity in BODY frame (ref: transform_by_quat)
        vel = self.robot_node.getVelocity()  # [vx,vy,vz, wx,wy,wz] world
        ang_vel_world = np.array(vel[3:6], dtype=np.float64)
        R = self._get_rotation_matrix()  # body → world
        ang_vel_body = R.T @ ang_vel_world  # world → body
        ang_vel_scaled = ang_vel_body * scales["ang_vel"]

        # 2. Projected gravity (ref: transform_by_quat(global_gravity, inv_quat))
        gravity_world = np.array([0.0, 0.0, -1.0])
        projected_gravity = R.T @ gravity_world

        # 3. Scaled commands
        commands_scaled = self.commands * self.commands_scale

        # 4. Residual joint positions (ref: dof_pos − default_dof_pos)
        dof_pos_res = (self.dof_pos - self.default_dof_pos) * scales["dof_pos"]

        # 5. Joint velocities
        dof_vel_scaled = self.dof_vel * scales["dof_vel"]

        # 6. Current actions (ref: self.actions, NOT last_actions)
        act = self.actions

        obs = np.concatenate(
            [
                ang_vel_scaled,  # 3
                projected_gravity,  # 3
                commands_scaled,  # 4
                dof_pos_res,  # 8
                dof_vel_scaled,  # 8
                act,  # 8
            ]
        )  # = 34

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _compute_reward(self):
        """
        ★ KEY CHANGE 8, 9, 10: reward structure exactly from reference

        Key differences from old version:
        - tracking_sigma = 0.25 (was 1.0)  → 4× tighter tracking
        - base_height   = −50   (was −2)   → 25× stronger height lock
        - similar_to_default uses L1 norm  (was L2)
        - NO orientation penalty           (ref does not use one)
        - All weights × dt                 (ref: reward_scales *= dt)
        - Velocities in BODY frame for reward computation
        """
        # Base state in body frame
        vel = self.robot_node.getVelocity()
        lin_vel_world = np.array(vel[0:3], dtype=np.float64)
        ang_vel_world = np.array(vel[3:6], dtype=np.float64)
        R = self._get_rotation_matrix()
        base_lin_vel = R.T @ lin_vel_world
        base_ang_vel = R.T @ ang_vel_world
        height = self.robot_node.getPosition()[2]
        sigma = self.config["tracking_sigma"]

        # ---- ref: _reward_tracking_lin_vel ----
        lin_vel_error = np.sum((self.commands[0:2] - base_lin_vel[0:2]) ** 2)
        r_track_lin = np.exp(-lin_vel_error / sigma)

        # ---- ref: _reward_tracking_ang_vel ----
        ang_vel_error = (self.commands[2] - base_ang_vel[2]) ** 2
        r_track_ang = np.exp(-ang_vel_error / sigma)

        # ---- ref: _reward_lin_vel_z ----
        r_lin_vel_z = base_lin_vel[2] ** 2

        # ---- ref: _reward_base_height ----
        r_height = (height - self.commands[3]) ** 2

        # ---- ref: _reward_action_rate (L2) ----
        r_action_rate = np.sum((self.last_actions - self.actions) ** 2)

        # ---- ref: _reward_similar_to_default (L1!) ----
        r_similar = np.sum(np.abs(self.dof_pos - self.default_dof_pos))

        # ---- Total (all scales already include ×dt) ----
        reward = (
            self.reward_scales["tracking_lin_vel"] * r_track_lin
            + self.reward_scales["tracking_ang_vel"] * r_track_ang
            + self.reward_scales["lin_vel_z"] * r_lin_vel_z
            + self.reward_scales["base_height"] * r_height
            + self.reward_scales["action_rate"] * r_action_rate
            + self.reward_scales["similar_to_default"] * r_similar
        )

        # Accumulate for diagnostics
        self.reward_sums["tracking_lin_vel"] += (
            self.reward_scales["tracking_lin_vel"] * r_track_lin
        )
        self.reward_sums["tracking_ang_vel"] += (
            self.reward_scales["tracking_ang_vel"] * r_track_ang
        )
        self.reward_sums["lin_vel_z"] += self.reward_scales["lin_vel_z"] * r_lin_vel_z
        self.reward_sums["base_height"] += self.reward_scales["base_height"] * r_height
        self.reward_sums["action_rate"] += (
            self.reward_scales["action_rate"] * r_action_rate
        )
        self.reward_sums["similar_to_default"] += (
            self.reward_scales["similar_to_default"] * r_similar
        )
        self.reward_sums["total"] += reward

        return float(reward)

    def _sample_commands(self):
        """ref: _sample_commands – uniform random within ranges"""
        r = self.config["command_ranges"]
        self.commands[0] = np.random.uniform(*r["v_x"])
        self.commands[1] = np.random.uniform(*r["v_y"])
        self.commands[2] = np.random.uniform(*r["w_z"])
        self.commands[3] = np.random.uniform(*r["z"])

    def _get_rotation_matrix(self):
        """3×3 rotation matrix, body → world"""
        m = self.robot_node.getOrientation()
        return np.array(m, dtype=np.float64).reshape(3, 3)

    def _get_orientation(self):
        """Euler angles (roll, pitch, yaw) in radians"""
        R = self._get_rotation_matrix()
        pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return roll, pitch, yaw


# ============ CALLBACK ============


class TrainingCallback(BaseCallback):
    """Logging callback — prints every 10 episodes."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        if self.locals.get("dones")[0]:
            info = self.locals["infos"][0]
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                if len(self.episode_rewards) % 10 == 0:
                    rr = self.episode_rewards[-10:]
                    ll = self.episode_lengths[-10:]
                    print(
                        f"\n[Ep {len(self.episode_rewards):4d}]  "
                        f"R: {np.mean(rr):+8.2f} (max {np.max(rr):+.2f})  "
                        f"L: {np.mean(ll):.0f}"
                    )
        return True


# ============ MAIN ============


def main():
    config = CONFIG

    print("=" * 60)
    mode = "TRAINING" if config["training_mode"] else "INFERENCE"
    print(f"GHOST DOG – {mode} (Reference-Aligned)")
    print("=" * 60)

    # Create environment
    env = GhostDogEnv(config=config)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    if config["training_mode"]:
        # ============ TRAINING ============
        model_path = config["model_name"]
        vecnorm_path = config["vecnorm_name"]
        model_exists = os.path.exists(f"{model_path}.zip")
        vecnorm_exists = os.path.exists(vecnorm_path)

        if model_exists and vecnorm_exists:
            print(f"\nRESUMING from {model_path}")
            env = VecNormalize.load(vecnorm_path, env)
            env.training = True
            env.norm_reward = True
            model = PPO.load(model_path, env=env)
            reset_num_timesteps = False
        else:
            print("\nSTARTING FRESH")
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
                policy_kwargs=config["policy_kwargs"],
                tensorboard_log="./ghostdog_tensorboard/",
            )
            reset_num_timesteps = True

        # FAST mode
        env.envs[0].unwrapped.simulationSetMode(
            env.envs[0].unwrapped.SIMULATION_MODE_FAST
        )

        callback = TrainingCallback()
        try:
            model.learn(
                total_timesteps=config["total_timesteps"],
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=reset_num_timesteps,
            )
            print("\nTraining complete!")
            model.save(model_path)
            env.save(vecnorm_path)
            print("Model saved.")
        except KeyboardInterrupt:
            print("\nInterrupted — saving progress…")
            model.save(model_path + "_interrupted")
            env.save(vecnorm_path.replace(".pkl", "_interrupted.pkl"))
            return

        # Switch to real-time eval
        env.envs[0].unwrapped.simulationSetMode(
            env.envs[0].unwrapped.SIMULATION_MODE_REAL_TIME
        )
        env.training = False
        env.norm_reward = False

    else:
        # ============ INFERENCE ============
        model_path = config["model_name"]
        vecnorm_path = config["vecnorm_name"]
        if not os.path.exists(f"{model_path}.zip"):
            print(f"ERROR: {model_path}.zip not found!")
            return

        print(f"Loading model: {model_path}")
        model = PPO.load(model_path, env=env)
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        env.envs[0].unwrapped.simulationSetMode(
            env.envs[0].unwrapped.SIMULATION_MODE_REAL_TIME
        )

    # ============ EVALUATION LOOP ============
    obs = env.reset()
    ep_reward = 0.0
    ep_length = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_length += 1
            if done[0]:
                print(f"\nEval episode: L={ep_length}, R={ep_reward:.2f}")
                obs = env.reset()
                ep_reward = 0.0
                ep_length = 0
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
