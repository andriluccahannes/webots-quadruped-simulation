# Reward Design

The reward function is designed to guide the robot to walk effectively while adhering to user-specified references for speed and altitude. Rewards are given for achieving objectives, and penalties are applied when deviations occur.

## Reward Terms

### 1. Linear Velocity Tracking Reward
The robot is encouraged to track $v_x, v_y$ references commanded by the user.
$$R_{lin\_vel} = \exp(-\|v_{xy}^{ref} - v_{xy}\|^2)$$
Where:
- $v_{xy}^{ref} = [v_x^{ref}, v_y^{ref}]$ is the commanded velocity.
- $v_{xy} = [v_x, v_y]$ is the actual velocity.

### 2. Angular Velocity Tracking Reward
The robot is encouraged to track $w_z$ reference commanded by the user.
$$R_{ang\_vel} = \exp(-(w_z^{ref} - w_z)^2)$$
Where:
- $w_{cmd,z}$ is the commanded yaw velocity.
- $w_{base,z}$ is the actual yaw velocity.

### 3. Height Penalty
The robot is encouraged to maintain a desired height as specified by the commanded altitude. A penalty is applied for deviations from this target height:
$$R_z = (z - z_{ref})^2$$
Where:
- $z$ is the current base height.
- $z_{ref}$ is the target height specified in the commands.

### 4. Pose Similarity Reward
To keep the robot's joint poses close to a default configuration, a penalty is applied for large deviations from the default joint positions:
$$R_{pose\_similarity} = \|q - q_{default}\|^2$$
Where:
- $q$ is the current joint position.
- $q_{default}$ is the default joint position.

### 5. Action Rate Penalty
To ensure smooth control and discourage abrupt changes in actions, a penalty is applied based on the difference between consecutive actions:
$$R_{action\_rate} = \|a_t - a_{t-1}\|^2$$
Where:
- $a_t$ and $a_{t-1}$ are the actions at the current and previous time steps, respectively.

### 6. Vertical Velocity Penalty
To discourage unnecessary movement along the vertical ($z$) axis, a penalty is applied to the squared $z$-axis velocity of the base when the robot is not actively jumping. The reward is:
$$R_{lin\_vel\_z} = v_z^2$$
Where:
- $v_z$ is the vertical velocity of the base.

### 7. Roll and Pitch Stabilization Penalty
To ensure the robot maintains stability, a penalty is applied to discourage large roll and pitch deviations of the base. This reward is:
$$R_{roll\_pitch} = roll^2 + pitch^2$$
Where:
- $roll$ is the roll angle of the base.
- $pitch$ is the pitch angle of the base.

---

# Episode Termination Condition

During training, episodes are terminated when specific criteria are met to ensure the robot remains in a healthy and functional state. The termination conditions include:

- $|roll| < roll_{min}$: Robot roll is below a certain threshold.
- $|pitch| < pitch_{min}$: Robot pitch is below a certain threshold.
- $z > z_{min}$: Robot altitude is above a minimum value.
- $steps \ge max\_steps$: Maximum number of steps reached.

## Implementation Example

```python
# check whether robot current state is healthy
def is_healthy(self, obs, curr_step):
    roll = obs[6]
    pitch = obs[7]
    z = obs[2]
    
    if (abs(roll) > self.roll_th or abs(pitch) > pitch_th or abs(z) < self.z_min or curr_step > max_steps):
        return False # dead
    else:
        return True # alive
```