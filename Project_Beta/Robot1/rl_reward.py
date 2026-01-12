# rl_reward.py (Robot1 version)
# ==============================================================================
# Reinforcement Learning Reward Function
# ==============================================================================
#
# This module defines the reward function for RL training.
# Modify this file to experiment with different reward designs.
#
# The reward function determines what behavior the agent learns:
#   - Positive rewards encourage certain behaviors
#   - Negative rewards (penalties) discourage behaviors
#
# IMPORTANT: Beta 1.4 Limitation
# ------------------------------
# Currently, Unity does not send real-time telemetry (pos_x, pos_z, status).
# Only SOC and step count are available during runtime.
# Full telemetry (position, status) is only written to metadata.csv at run end.
#
# This reward function uses available data:
#   - SOC (battery level)
#   - step count (proxy for survival)
#   - action smoothness
#
# Future enhancement (Beta 1.5+): Unity real-time telemetry for richer rewards.
#
# ==============================================================================

# Reward weights (tune these to change agent behavior)
REWARD_WEIGHTS = {
    # Per-step rewards (applied every frame)
    'survival_bonus': 0.1,        # Small reward for staying alive each step
    'soc_efficiency': 0.5,        # Reward for maintaining SOC
    'action_smoothness': 0.05,    # Reward for smooth actions (less jitter)

    # Penalties
    'soc_low_penalty': -1.0,      # Penalty when SOC drops below threshold
    'soc_empty_penalty': -100.0,  # Large penalty for running out of battery
    'extreme_action_penalty': -0.1,  # Penalty for extreme steering

    # Terminal rewards (applied at episode end - set by external caller)
    'finish_bonus': 500.0,        # Bonus for finishing (2 laps)
    'lap_complete': 100.0,        # Bonus for completing 1 lap
    'fallen_penalty': -200.0,     # Penalty for falling
}

# Thresholds
SOC_LOW_THRESHOLD = 0.2  # SOC below this triggers penalty
STEERING_EXTREME_THRESHOLD = 0.45  # ~26 degrees, close to max

# State for smoothness calculation
_prev_action = {'drive': 0.0, 'steer': 0.0}


def calculate_reward(prev_state, curr_state, action):
    """
    Calculate reward based on state transition and action.

    Args:
        prev_state: dict with previous state info
            - soc: battery level (0.0-1.0)
            - step: current step count
        curr_state: dict with current state info (same format)
        action: dict with action taken
            - drive: drive torque (-1.0 to 1.0)
            - steer: steering angle (radians)

    Returns:
        float: reward value

    Note:
        Terminal rewards (finish, lap complete, fallen) should be added
        by the caller when episode ends, using get_terminal_reward().
    """
    global _prev_action
    reward = 0.0

    # === (1) Survival Bonus ===
    # Small positive reward for each step (encourages staying in race)
    reward += REWARD_WEIGHTS['survival_bonus']

    # === (2) SOC-based Rewards ===
    curr_soc = curr_state.get('soc', 1.0)
    prev_soc = prev_state.get('soc', 1.0)

    # Reward for maintaining SOC (not wasting energy)
    # Higher SOC = higher reward
    reward += curr_soc * REWARD_WEIGHTS['soc_efficiency']

    # Penalty for low SOC
    if curr_soc < SOC_LOW_THRESHOLD:
        reward += REWARD_WEIGHTS['soc_low_penalty']

    # Large penalty for empty battery
    if curr_soc <= 0:
        reward += REWARD_WEIGHTS['soc_empty_penalty']
        print(f"[Reward] SOC empty! Penalty={REWARD_WEIGHTS['soc_empty_penalty']}")

    # === (3) Action Smoothness Reward ===
    # Reward for not changing actions too drastically
    drive_diff = abs(action.get('drive', 0) - _prev_action.get('drive', 0))
    steer_diff = abs(action.get('steer', 0) - _prev_action.get('steer', 0))
    smoothness = 1.0 - (drive_diff + steer_diff) / 2.0  # 0 to 1
    reward += smoothness * REWARD_WEIGHTS['action_smoothness']

    # === (4) Extreme Action Penalty ===
    # Discourage extreme steering (often leads to instability)
    if abs(action.get('steer', 0)) > STEERING_EXTREME_THRESHOLD:
        reward += REWARD_WEIGHTS['extreme_action_penalty']

    # Update previous action for next smoothness calculation
    _prev_action = action.copy()

    return reward


def get_terminal_reward(final_status, final_soc=None, race_time_ms=None):
    """
    Get terminal reward based on how the episode ended.

    Args:
        final_status: str - 'Finish', 'Lap1', 'Fallen', 'Force end', etc.
        final_soc: float - remaining SOC at end (optional)
        race_time_ms: int - total race time in ms (optional)

    Returns:
        float: terminal reward value

    Usage:
        Call this when episode ends to get the final reward.
    """
    reward = 0.0

    if final_status == 'Finish':
        reward += REWARD_WEIGHTS['finish_bonus']
        print(f"[Reward] Finish! +{REWARD_WEIGHTS['finish_bonus']}")

        # Bonus for remaining SOC at finish
        if final_soc is not None and final_soc > 0:
            soc_bonus = final_soc * 50.0  # Extra bonus for efficiency
            reward += soc_bonus
            print(f"[Reward] SOC efficiency bonus: +{soc_bonus:.1f}")

    elif final_status == 'Lap1':
        reward += REWARD_WEIGHTS['lap_complete']
        print(f"[Reward] Lap 1 completed! +{REWARD_WEIGHTS['lap_complete']}")

    elif final_status == 'Fallen':
        reward += REWARD_WEIGHTS['fallen_penalty']
        print(f"[Reward] Fallen! {REWARD_WEIGHTS['fallen_penalty']}")

    return reward


def reset_state():
    """Reset internal state for new episode."""
    global _prev_action
    _prev_action = {'drive': 0.0, 'steer': 0.0}


def get_reward_info():
    """Return current reward weight configuration."""
    return REWARD_WEIGHTS.copy()


def set_reward_weight(key, value):
    """Update a reward weight."""
    if key in REWARD_WEIGHTS:
        REWARD_WEIGHTS[key] = value
        print(f"[Reward] Updated {key} = {value}")
    else:
        print(f"[Reward] Unknown weight key: {key}")


# ==============================================================================
# NOTES FOR ENGINEERS
# ==============================================================================
#
# Reward Shaping Tips:
# --------------------
# 1. Start simple, add complexity gradually
# 2. Balance positive and negative rewards
# 3. Make terminal rewards (finish, fallen) much larger than step rewards
# 4. Normalize rewards if they vary too much in scale
#
# Common Issues:
# --------------
# - Agent drives in circles: Add penalty for not progressing
# - Agent too conservative: Reduce penalties or increase speed bonus
# - Agent crashes often: Increase fallen penalty, add steering smoothness reward
# - Agent ignores battery: Increase SOC-related rewards/penalties
#
# ==============================================================================
