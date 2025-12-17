"""
Evaluate SAC+HER model trained with training.py on parking-v0 environment.

This script loads a trained model and evaluates it with visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import numpy as np

import highway_env  # noqa: F401  (register envs)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


# -------------------------
# Environment factory (same as training.py)
# -------------------------
def make_env(rank: int, seed: int, render_mode=None):
    """Create environment with same config as training."""
    def _init():
        env = gym.make("parking-v0", render_mode=render_mode)

        # Same config as training.py
        config = {
            "observation": {
                "type": "KinematicsGoal",
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": True,
            },
            "action": {"type": "ContinuousAction"},
            "controlled_vehicles": 1,
            "vehicles_count": 0,
            "add_walls": False,
            "duration": 120,
            "simulation_frequency": 15,
            "policy_frequency": 5,
        }

        if hasattr(env, "configure"):
            env.configure(config)
        else:
            env.unwrapped.configure(config)

        env.reset(seed=seed + rank)
        return env

    return _init


# -------------------------
# Evaluation
# -------------------------
def evaluate_model(
    model_path: Path,
    num_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True,
    seed: int = 42
):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to model file (.zip)
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        deterministic: Use deterministic policy
        seed: Random seed
    """
    print(f"Loading model from: {model_path}")
    
    # Create environment FIRST (required for HER models)
    render_mode = "human" if render else None
    env = DummyVecEnv([make_env(0, seed, render_mode=render_mode)])
    env = VecMonitor(env)
    
    # Load model with environment
    model = SAC.load(str(model_path), env=env)
    
    print(f"\nEvaluating model for {num_episodes} episodes...")
    print(f"Deterministic: {deterministic}, Render: {render}")
    print("-" * 60)
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    try:
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            while not done:
                # Get action from model
                action, _ = model.predict(obs, deterministic=deterministic)
                
                # Step environment
                obs, rewards, dones, infos = env.step(action)
                
                # Unpack results (vectorized env)
                done = bool(dones[0])
                reward = float(rewards[0])
                info = infos[0] if isinstance(infos, list) else infos
                
                episode_reward += reward
                episode_length += 1
                
                # Render if enabled
                if render:
                    env.render()
                
                # Check for episode end
                if done:
                    success = info.get("is_success", False)
                    if success:
                        success_count += 1
                    
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    
                    print(f"  Reward: {episode_reward:.2f}, "
                          f"Length: {episode_length}, "
                          f"Success: {success}")
                    
                    # Print additional info if available
                    if "crashed" in info:
                        print(f"  Crashed: {info['crashed']}")
                    if "offroad" in info:
                        print(f"  Offroad: {info['offroad']}")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    
    finally:
        env.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Success Rate: {success_count}/{len(episode_rewards)} ({100*success_count/len(episode_rewards):.1f}%)")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate SAC+HER model on parking-v0"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model file (.zip) - can be final_model.zip or best/best_model.zip"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate (default: 10)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment (show car parking)"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy instead of deterministic"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    evaluate_model(
        model_path=model_path,
        num_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

