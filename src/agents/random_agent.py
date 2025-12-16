"""
Random Agent for Baseline Benchmarking

A simple agent that selects random actions from the environment's action space.
Used as a baseline to compare against trained RL agents.
"""

import numpy as np
from typing import Dict, Any, Optional


class RandomAgent:
    """
    A random agent that samples actions uniformly from the action space.
    
    This agent serves as a baseline benchmark for comparison with trained
    reinforcement learning agents.
    """
    
    def __init__(self, action_space, seed: Optional[int] = None):
        """
        Initialize the random agent.
        
        Args:
            action_space: The environment's action space (gymnasium.Space)
            seed: Optional random seed for reproducibility
        """
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)
        
    def predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Predict an action given an observation.
        
        For a random agent, the observation is ignored and a random action
        is sampled from the action space.
        
        Args:
            observation: Current observation from the environment
            deterministic: Ignored for random agent (always random)
            
        Returns:
            A random action from the action space
        """
        return self.action_space.sample()
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Get an action for the given observation (alias for predict).
        
        Args:
            observation: Current observation from the environment
            
        Returns:
            A random action from the action space
        """
        return self.predict(observation)
    
    def reset(self):
        """Reset the agent's internal state (no-op for random agent)."""
        pass

