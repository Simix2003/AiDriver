"""
Train SAC + HER on highway-env parking-v0 (goal-conditioned) the correct way.

Key fixes vs your current setup:
- DO NOT wrap parking-v0: it's already GoalEnv-style (KinematicsGoal obs + compute_reward).
- Turn OFF walls (add_walls=False) to prevent constant early crashes.
- Keep the env's own reward/termination logic (HER relies on compute_reward consistency).
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import numpy as np

import highway_env  # noqa: F401  (register envs)

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed


# -------------------------
# Environment factory
# -------------------------
def make_env(rank: int, seed: int, render_mode=None):
    def _init():
        env = gym.make("parking-v0", render_mode=render_mode)

        # IMPORTANT: parking-v0 default observation is already "KinematicsGoal"
        # and includes achieved_goal / desired_goal + compute_reward().
        config = {
            "observation": {
                "type": "KinematicsGoal",
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": True,  # helps learning stability
            },
            "action": {"type": "ContinuousAction"},
            "controlled_vehicles": 1,
            "vehicles_count": 0,

            # BIG FIX: remove walls so the agent doesn't just crash all the time
            "add_walls": False,

            # longer training episodes
            "duration": 120,

            # keep default-ish dynamics
            "simulation_frequency": 15,
            "policy_frequency": 5,
        }

        if hasattr(env, "configure"):
            env.configure(config)
        else:
            # fallback (some wrappers expose .unwrapped.configure)
            env.unwrapped.configure(config)

        env.reset(seed=seed + rank)
        return env

    return _init


# -------------------------
# Training
# -------------------------
def main():
    seed = 123
    set_random_seed(seed)

    # Output
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("models") / "parking" / f"sac_her_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parallel envs (faster)
    n_envs = max(2, os.cpu_count() // 2)

    env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    env = VecMonitor(env)

    # Eval env must be deterministic & single env
    eval_env = DummyVecEnv([make_env(0, seed + 10_000, render_mode=None)])
    eval_env = VecMonitor(eval_env)

    # HER settings (good defaults)
    # goal_selection_strategy: "future" is the typical best starting point
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=8,
            goal_selection_strategy="future",
        ),
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=20_000,
        batch_size=2048,
        tau=0.02,
        gamma=0.98,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=str(out_dir / "tb"),
        device="auto",
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="sac_her",
        save_replay_buffer=True,
    )

    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(out_dir / "best"),
        log_path=str(out_dir / "eval"),
        eval_freq=25_000,
        n_eval_episodes=50,
        deterministic=True,
        render=False,
    )

    total_steps = 1_000_000  # 100k is usually too low for stable parking success

    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
        log_interval=10,
    )

    model.save(str(out_dir / "final_model.zip"))
    env.close()
    eval_env.close()

    print(f"\n✅ Done. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
