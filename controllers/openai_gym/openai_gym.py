import sys
import numpy as np
import gymnasium as gym
from controller import Supervisor

# Import Stable Baselines 3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    sys.exit('Please install SB3: "pip install stable-baselines3 shimmy gymnasium"')

class InvertedPendulumEnv(Supervisor, gym.Env):
    def __init__(self):
        # 1. Initialize Supervisor
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())

        # 2. Get Devices
        self.motor = self.getDevice('linear motor')
        self.motor.setPosition(float('inf'))
        self.motor.setVelocity(0)
        
        self.pole_sensor = self.getDevice('pole_sensor')
        self.pole_sensor.enable(self.timestep)

        self.cart_sensor = self.getDevice('cart_sensor')
        self.cart_sensor.enable(self.timestep)

        # 3. Define Action Space (Discrete: 0=Left, 1=Right)
        # self.action_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Box(-2.5, 2.5)

        # 4. Define Observation Space
        # [Cart Position, Cart Velocity, Pole Angle, Pole Velocity]
        # FIXED: Made bounds consistent with termination conditions
        high = np.array([1.0, 10.0, 0.3, 10.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Variables for velocity calculation
        self.last_cart_pos = 0.0
        self.last_pole_angle = 0.0
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = 1000
        
        print(f"Environment initialized. Timestep: {self.timestep}ms")

    def reset(self, seed=None, options=None):
        """
        Resets the simulation to the start state.
        """
        super().reset(seed=seed)
        
        # Reset simulation
        self.simulationReset()
        
        # Step once to stabilize
        super().step(self.timestep)
        
        # Set random initial pole angle
        pole_joint_node = self.getFromDef("POLE_JOINT")
        random_angle = np.random.uniform(-0.05, 0.05)
        
        if pole_joint_node is not None:
            joint_params = pole_joint_node.getField("jointParameters").getSFNode()
            if joint_params:
                pole_pos_field = joint_params.getField("position")
                pole_pos_field.setSFFloat(random_angle)
        else:
            print("WARNING: 'POLE_JOINT' DEF not found in world file!")
        
        # Reset motor
        self.motor.setPosition(float('inf'))
        self.motor.setVelocity(0)
        
        # Step again to apply changes
        super().step(self.timestep)
        
        # Read initial sensor values
        self.last_cart_pos = self.cart_sensor.getValue()
        self.last_pole_angle = self.pole_sensor.getValue()
        
        self.episode_steps = 0
        
        obs = self._get_obs()
        print(f"Reset complete. Initial obs: {obs}")
        
        return obs, {}

    def step(self, action):
        # 1. Apply Action with higher force
        # FIXED: Increased speed for more responsive control
        # speed = 2.5 if action == 1 else -2.5
        speed = action
       
        self.motor.setVelocity(speed)

        # 2. Step Simulation
        super().step(self.timestep)

        # 3. Get Observation
        obs = self._get_obs()
        
        # 4. Calculate Reward
        # Reward based on pole angle (upright is better)
        angle_reward = 1.0 - abs(obs[2]) / 0.3
        # Small penalty for cart position away from center
        position_penalty = -0.1 * abs(obs[0]) / 1.0
        
        reward = angle_reward + position_penalty
        
        # 5. Check Termination
        # Fail if cart hits edge OR pole tilts too far
        terminated = bool(
            obs[0] < -0.8 or obs[0] > 0.8 or  # Cart position limit
            obs[2] < -0.25 or obs[2] > 0.25   # Pole angle limit (~14 degrees)
        )
        
        # Truncate after max steps
        self.episode_steps += 1
        truncated = self.episode_steps >= self.max_episode_steps

        if terminated:
            reward = -10.0  # Penalty for failure
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Read sensors
        cart_pos = self.cart_sensor.getValue()
        pole_angle = self.pole_sensor.getValue()

        # Calculate velocities (derivative)
        dt = self.timestep / 1000.0
        cart_vel = (cart_pos - self.last_cart_pos) / dt
        pole_vel = (pole_angle - self.last_pole_angle) / dt

        # Update for next iteration
        self.last_cart_pos = cart_pos
        self.last_pole_angle = pole_angle

        obs = np.array([cart_pos, cart_vel, pole_angle, pole_vel], dtype=np.float32)
        
        # Clip to observation space bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        
        return obs

def main():
    print("="*60)
    print("Starting Inverted Pendulum RL Training")
    print("="*60)
    
    # 1. Create Environment
    env = DummyVecEnv([lambda: InvertedPendulumEnv()])
    
    # 2. Add Normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Create Model with better hyperparameters
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
        ent_coef=0.0,
    )

    # 4. Train
    print("\n" + "="*60)
    print("Starting Training in FAST mode...")
    print("="*60 + "\n")
    
    # Set to fast mode for training
    env.envs[0].simulationSetMode(env.envs[0].SIMULATION_MODE_FAST)
    
    try:
        # Train for longer
        model.learn(total_timesteps=100000)
        print("\n" + "="*60)
        print("Training Finished Successfully!")
        print("="*60)
        
        # 5. Save
        model.save("ppo_cartpole_custom")
        env.save("vec_normalize.pkl")
        print("Model and normalization stats saved!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        model.save("ppo_cartpole_custom_interrupted")
        env.save("vec_normalize_interrupted.pkl")
        return

    # 6. Replay
    print("\n" + "="*60)
    print("Starting Replay in REAL TIME mode...")
    print("="*60 + "\n")
    
    env.envs[0].simulationSetMode(env.envs[0].SIMULATION_MODE_REAL_TIME)
    
    # Turn off training mode for replay
    env.training = False
    env.norm_reward = False

    obs = env.reset()
    episode_reward = 0
    episode_length = 0
    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        episode_reward += reward[0]
        episode_length += 1
        
        if done[0]:
            print(f"Episode finished: Length={episode_length}, Reward={episode_reward:.2f}")
            obs = env.reset()
            episode_reward = 0
            episode_length = 0

if __name__ == '__main__':
    main()