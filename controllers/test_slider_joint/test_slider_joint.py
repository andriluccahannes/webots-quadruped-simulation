from controller import Robot, Keyboard

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get motor
motor = robot.getDevice('linear motor')
motor.setPosition(float('inf'))
motor.setVelocity(0)

# Enable keyboard
keyboard = Keyboard()
keyboard.enable(timestep)

velocity = 0.0
increment = 0.1  # change speed step


while robot.step(timestep) != -1:
    key = keyboard.getKey()
   

    # Handle keyboard input
    if key == Keyboard.RIGHT:
        velocity += increment
    elif key == Keyboard.LEFT:
        velocity -= increment
    elif key == ord('R'):  # reset key
        velocity = 0.0

    # Apply velocity to motor
    motor.setVelocity(velocity)

    print(f"Velocity: {velocity:.2f}")
