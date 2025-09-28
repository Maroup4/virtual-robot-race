# status_Robot.py
# Maintains global state of the robot (e.g., waiting for start, running, etc.)

# State constants
WAITING_START = 0     # Waiting for the start signal
RUN_STRAIGHT = 1      # Currently running on a straight segment
RUN_CORNER = 2        # Placeholder for future cornering logic

# Global state variable
robot_state = WAITING_START

def get_state():
    """Returns the current global robot state."""
    return robot_state

def set_state(state):
    """Sets the current global robot state."""
    global robot_state
    robot_state = state
