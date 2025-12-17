"""
Run Random Agent Benchmark

Script to run a benchmark evaluation of the random agent on the parking environment.
This establishes a baseline for comparison with trained RL agents.
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import gymnasium as gym
from src.agents.random_agent import RandomAgent
from src.evaluation.benchmark import run_benchmark
import highway_env  # Register environments


def main():
    parser = argparse.ArgumentParser(
        description="Run random agent benchmark on parking environment"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run (default: 100)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode (default: no limit)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/logs/random_agent_benchmark.json",
        help="Output file path for results (default: data/logs/random_agent_benchmark.json)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Create environment (using same approach as new_try.py)
    print("Creating parking environment...")
    env = gym.make("parking-v0", render_mode="human" if args.render else None)
    
    # Configure environment similar to new_try.py
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
    elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "configure"):
        env.unwrapped.configure(config)
    
    # Create random agent
    print("Initializing random agent...")
    agent = RandomAgent(env.action_space, seed=args.seed)
    
    # Run benchmark
    print(f"\nRunning benchmark with {args.episodes} episodes...")
    results = run_benchmark(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        render=args.render,
        verbose=not args.quiet,
        seed=args.seed
    )
    
    # Print summary
    results.print_summary()
    
    # Save results
    output_path = Path(args.output)
    print(f"Saving results to {output_path}...")
    results.save(output_path)
    print("Done!")
    
    env.close()


if __name__ == "__main__":
    main()

