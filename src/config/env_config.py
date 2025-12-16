"""
Environment Configuration

Helper functions for creating and configuring environments consistently.
"""

import gymnasium as gym
from typing import Dict, Any, Optional


def make_parking_env(
    render_mode: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> gym.Env:
    """
    Create a parking environment with optional configuration.
    
    Args:
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        config: Optional environment configuration dictionary
        
    Returns:
        Configured parking environment
    """
    if config is None:
        config = {}
    
    env = gym.make("parking-v0", render_mode=render_mode, **config)
    return env


def get_default_parking_config() -> Dict[str, Any]:
    """
    Get default configuration for the parking environment.
    
    Returns:
        Dictionary of default configuration parameters
    """
    return {
        "observation": {
            "type": "Kinematics",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "normalize": False,
            "vehicles_count": 1,
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            }
        },
        "action": {
            "type": "ContinuousAction"
        },
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "duration": 20,
        "screen_width": 600,
        "screen_height": 600,
        "scaling": 5.5,
        "centering_position": [0.5, 0.5],
        "parking": {
            "type": "on_lane",
            "vehicles_count": 0
        }
    }

