"""
Benchmarking and Evaluation Utilities

Provides functions for running benchmarks, collecting metrics, and comparing
agent performance.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import gymnasium as gym


def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy types and other non-JSON-serializable objects
    to native Python types. Compatible with NumPy 1.x and 2.x.
    """
    # Check for numpy integer types
    if isinstance(obj, np.integer):
        return int(obj)
    # Check for numpy floating point types
    elif isinstance(obj, np.floating):
        return float(obj)
    # Check for numpy boolean
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Check for numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle dictionaries recursively
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    # Handle lists and tuples recursively
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    # For other types, try JSON serialization first
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # If it fails, convert to string as last resort
            return str(obj)


@dataclass
class EpisodeMetrics:
    """Metrics collected from a single episode."""
    episode_length: int
    total_reward: float
    success: bool
    info: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class BenchmarkResults:
    """Results from running a benchmark over multiple episodes."""
    agent_name: str
    num_episodes: int
    episodes: List[EpisodeMetrics] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate across all episodes."""
        if not self.episodes:
            return 0.0
        return sum(1 for ep in self.episodes if ep.success) / len(self.episodes)
    
    @property
    def mean_reward(self) -> float:
        """Calculate the mean reward across all episodes."""
        if not self.episodes:
            return 0.0
        return np.mean([ep.total_reward for ep in self.episodes])
    
    @property
    def std_reward(self) -> float:
        """Calculate the standard deviation of rewards."""
        if not self.episodes:
            return 0.0
        return np.std([ep.total_reward for ep in self.episodes])
    
    @property
    def mean_episode_length(self) -> float:
        """Calculate the mean episode length."""
        if not self.episodes:
            return 0.0
        return np.mean([ep.episode_length for ep in self.episodes])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "num_episodes": self.num_episodes,
            "success_rate": self.success_rate,
            "mean_reward": float(self.mean_reward),
            "std_reward": float(self.std_reward),
            "mean_episode_length": float(self.mean_episode_length),
            "episodes": [
                {
                    "episode_length": ep.episode_length,
                    "total_reward": float(ep.total_reward),
                    "success": bool(ep.success),  # Ensure it's a Python bool
                    "info": _make_json_serializable(ep.info)  # Convert numpy types
                }
                for ep in self.episodes
            ]
        }
    
    def save(self, filepath: Path):
        """Save results to a JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'BenchmarkResults':
        """Load results from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = cls(
            agent_name=data["agent_name"],
            num_episodes=data["num_episodes"]
        )
        results.episodes = [
            EpisodeMetrics(
                episode_length=ep["episode_length"],
                total_reward=ep["total_reward"],
                success=ep["success"],
                info=ep.get("info", {})
            )
            for ep in data["episodes"]
        ]
        return results
    
    def print_summary(self):
        """Print a summary of the benchmark results."""
        print(f"\n{'='*60}")
        print(f"Benchmark Results: {self.agent_name}")
        print(f"{'='*60}")
        print(f"Episodes: {self.num_episodes}")
        print(f"Success Rate: {self.success_rate:.2%}")
        print(f"Mean Reward: {self.mean_reward:.4f} ± {self.std_reward:.4f}")
        print(f"Mean Episode Length: {self.mean_episode_length:.2f} steps")
        print(f"{'='*60}\n")


def run_benchmark(
    agent: Any,
    env: gym.Env,
    num_episodes: int = 100,
    max_steps_per_episode: Optional[int] = None,
    render: bool = False,
    verbose: bool = True,
    seed: Optional[int] = None
) -> BenchmarkResults:
    """
    Run a benchmark evaluation of an agent on an environment.
    
    Args:
        agent: The agent to evaluate (must have a predict() method)
        env: The gymnasium environment
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode (None = no limit)
        render: Whether to render the environment
        verbose: Whether to print progress
        seed: Optional random seed for environment
        
    Returns:
        BenchmarkResults object containing all collected metrics
    """
    if seed is not None:
        env.reset(seed=seed)
    
    results = BenchmarkResults(
        agent_name=agent.__class__.__name__,
        num_episodes=num_episodes
    )
    
    agent.reset()
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            if max_steps_per_episode and episode_length >= max_steps_per_episode:
                truncated = True
                break
                
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        # Determine success from info dict (parking-v0 typically has 'is_success')
        # Convert numpy bool to Python bool for JSON serialization
        success = bool(info.get('is_success', False))
        
        episode_metrics = EpisodeMetrics(
            episode_length=episode_length,
            total_reward=episode_reward,
            success=success,
            info=info.copy()
        )
        results.episodes.append(episode_metrics)
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Reward: {episode_reward:.2f}, "
                  f"Length: {episode_length}, "
                  f"Success: {success}")
    
    if render:
        env.close()
    
    return results

