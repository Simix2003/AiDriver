"""
Visualization Utilities

Tools for visualizing training progress, rewards, and episode replays.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


@dataclass
class RewardComponents:
    """Breakdown of reward components."""
    total: float = 0.0
    distance_penalty: float = 0.0
    collision_penalty: float = 0.0
    success_bonus: float = 0.0
    step_penalty: float = 0.0
    other: float = 0.0


class RewardTracker:
    """
    Tracks reward components throughout an episode.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize reward tracker.
        
        Args:
            max_history: Maximum number of steps to track
        """
        self.max_history = max_history
        self.reset()
    
    def reset(self):
        """Reset tracker for new episode."""
        self.rewards: deque = deque(maxlen=self.max_history)
        self.components: deque = deque(maxlen=self.max_history)
        self.step = 0
        self.episode_reward = 0.0
    
    def add_reward(
        self,
        reward: float,
        info: Optional[Dict[str, Any]] = None
    ) -> RewardComponents:
        """
        Add a reward and parse its components.
        
        Args:
            reward: Total reward for this step
            info: Environment info dict (may contain reward breakdown)
            
        Returns:
            RewardComponents object
        """
        components = RewardComponents(total=reward)
        
        # Try to extract reward components from info
        if info is not None:
            # Common reward component keys in highway-env
            components.distance_penalty = info.get("distance_penalty", 0.0)
            components.collision_penalty = info.get("collision_penalty", 0.0)
            components.success_bonus = info.get("success_bonus", 0.0)
            components.step_penalty = info.get("step_penalty", 0.0)
            
            # If components not in info, try to infer from reward
            if components.distance_penalty == 0.0 and components.collision_penalty == 0.0:
                # Heuristic: if reward is positive and large, likely success
                if reward > 0.5:
                    components.success_bonus = reward
                # If reward is negative, likely penalty
                elif reward < -0.1:
                    components.distance_penalty = reward
        else:
            # No info available, use heuristics
            if reward > 0.5:
                components.success_bonus = reward
            elif reward < -0.1:
                components.distance_penalty = reward
            else:
                components.step_penalty = reward
        
        self.rewards.append(reward)
        self.components.append(components)
        self.episode_reward += reward
        self.step += 1
        
        return components
    
    def get_current_components(self) -> RewardComponents:
        """Get reward components for the current step."""
        if self.components:
            return self.components[-1]
        return RewardComponents()
    
    def get_episode_summary(self) -> Dict[str, float]:
        """Get summary statistics for the episode."""
        if not self.rewards:
            return {
                "total_reward": 0.0,
                "mean_reward": 0.0,
                "max_reward": 0.0,
                "min_reward": 0.0,
                "num_steps": 0
            }
        
        rewards_array = np.array(self.rewards)
        return {
            "total_reward": float(self.episode_reward),
            "mean_reward": float(np.mean(rewards_array)),
            "max_reward": float(np.max(rewards_array)),
            "min_reward": float(np.min(rewards_array)),
            "num_steps": len(self.rewards)
        }


class RealTimeVisualizer:
    """
    Real-time visualization of training progress with reward overlays.
    """
    
    def __init__(
        self,
        window_name: str = "Parking Training",
        font_scale: float = 0.6,
        font_thickness: int = 1
    ):
        """
        Initialize real-time visualizer.
        
        Args:
            window_name: Name of the display window
            font_scale: Scale factor for text
            font_thickness: Thickness of text
        """
        self.window_name = window_name
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Colors (BGR format for OpenCV)
        self.color_positive = (0, 255, 0)  # Green
        self.color_negative = (0, 0, 255)  # Red
        self.color_neutral = (255, 255, 255)  # White
        self.color_info = (255, 255, 0)  # Cyan
    
    def overlay_rewards(
        self,
        frame: np.ndarray,
        reward_tracker: RewardTracker,
        episode: int,
        step: int,
        success: bool = False
    ) -> np.ndarray:
        """
        Overlay reward information on a frame.
        
        Args:
            frame: Image frame from environment
            reward_tracker: RewardTracker instance
            episode: Current episode number
            step: Current step in episode
            success: Whether episode was successful
            
        Returns:
            Frame with overlays
        """
        # Create a copy to avoid modifying original
        overlay = frame.copy()
        
        # Get current reward components
        components = reward_tracker.get_current_components()
        summary = reward_tracker.get_episode_summary()
        
        # Position for text (top-left corner)
        y_offset = 25
        line_height = 25
        
        # Episode and step info
        cv2.putText(
            overlay,
            f"Episode: {episode} | Step: {step}",
            (10, y_offset),
            self.font,
            self.font_scale,
            self.color_info,
            self.font_thickness
        )
        y_offset += line_height
        
        # Success indicator
        success_text = "SUCCESS!" if success else "In Progress"
        success_color = self.color_positive if success else self.color_neutral
        cv2.putText(
            overlay,
            f"Status: {success_text}",
            (10, y_offset),
            self.font,
            self.font_scale,
            success_color,
            self.font_thickness
        )
        y_offset += line_height
        
        # Current step reward
        reward_color = self.color_positive if components.total >= 0 else self.color_negative
        cv2.putText(
            overlay,
            f"Step Reward: {components.total:.3f}",
            (10, y_offset),
            self.font,
            self.font_scale,
            reward_color,
            self.font_thickness
        )
        y_offset += line_height
        
        # Cumulative episode reward
        cv2.putText(
            overlay,
            f"Episode Reward: {summary['total_reward']:.2f}",
            (10, y_offset),
            self.font,
            self.font_scale,
            self.color_info,
            self.font_thickness
        )
        y_offset += line_height
        
        # Reward breakdown
        if components.success_bonus > 0:
            cv2.putText(
                overlay,
                f"  Success Bonus: +{components.success_bonus:.3f}",
                (10, y_offset),
                self.font,
                self.font_scale * 0.8,
                self.color_positive,
                self.font_thickness
            )
            y_offset += int(line_height * 0.8)
        
        if components.distance_penalty < 0:
            cv2.putText(
                overlay,
                f"  Distance Penalty: {components.distance_penalty:.3f}",
                (10, y_offset),
                self.font,
                self.font_scale * 0.8,
                self.color_negative,
                self.font_thickness
            )
            y_offset += int(line_height * 0.8)
        
        if components.collision_penalty < 0:
            cv2.putText(
                overlay,
                f"  Collision Penalty: {components.collision_penalty:.3f}",
                (10, y_offset),
                self.font,
                self.font_scale * 0.8,
                self.color_negative,
                self.font_thickness
            )
            y_offset += int(line_height * 0.8)
        
        if components.step_penalty < 0:
            cv2.putText(
                overlay,
                f"  Step Penalty: {components.step_penalty:.3f}",
                (10, y_offset),
                self.font,
                self.font_scale * 0.8,
                self.color_negative,
                self.font_thickness
            )
        
        return overlay
    
    def display_frame(self, frame: np.ndarray):
        """Display a frame in a window."""
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)
    
    def close(self):
        """Close the display window."""
        cv2.destroyAllWindows()


@dataclass
class EpisodeData:
    """Data structure for storing episode information."""
    episode: int
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    reward_components: List[RewardComponents] = field(default_factory=list)
    info: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    total_reward: float = 0.0
    length: int = 0


class EpisodeReplay:
    """
    Replay saved episodes with reward annotations.
    """
    
    def __init__(self, playback_speed: float = 1.0):
        """
        Initialize episode replay.
        
        Args:
            playback_speed: Speed multiplier for playback (1.0 = real-time)
        """
        self.playback_speed = playback_speed
        self.visualizer = RealTimeVisualizer(window_name="Episode Replay")
    
    def save_episode(
        self,
        episode_data: EpisodeData,
        filepath: Path
    ):
        """
        Save episode data to file.
        
        Args:
            episode_data: EpisodeData to save
            filepath: Path to save file
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        data = {
            "episode": episode_data.episode,
            "success": episode_data.success,
            "total_reward": episode_data.total_reward,
            "length": episode_data.length,
            "rewards": [float(r) for r in episode_data.rewards],
            "reward_components": [
                {
                    "total": float(c.total),
                    "distance_penalty": float(c.distance_penalty),
                    "collision_penalty": float(c.collision_penalty),
                    "success_bonus": float(c.success_bonus),
                    "step_penalty": float(c.step_penalty),
                    "other": float(c.other)
                }
                for c in episode_data.reward_components
            ],
            "info": [
                {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
                 for k, v in info.items()}
                for info in episode_data.info
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_episode(self, filepath: Path) -> EpisodeData:
        """
        Load episode data from file.
        
        Args:
            filepath: Path to episode file
            
        Returns:
            EpisodeData object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        reward_components = [
            RewardComponents(
                total=c["total"],
                distance_penalty=c["distance_penalty"],
                collision_penalty=c["collision_penalty"],
                success_bonus=c["success_bonus"],
                step_penalty=c["step_penalty"],
                other=c["other"]
            )
            for c in data["reward_components"]
        ]
        
        return EpisodeData(
            episode=data["episode"],
            rewards=data["rewards"],
            reward_components=reward_components,
            info=data["info"],
            success=data["success"],
            total_reward=data["total_reward"],
            length=data["length"]
        )
    
    def replay_episode(
        self,
        env,
        episode_data: EpisodeData,
        render: bool = True
    ):
        """
        Replay an episode with reward annotations.
        
        Args:
            env: Environment to replay in
            episode_data: EpisodeData to replay
            render: Whether to render the replay
        """
        if not render:
            return
        
        # Reset environment to initial state
        obs, info = env.reset()
        
        # Create reward tracker for visualization
        tracker = RewardTracker()
        tracker.rewards = deque(episode_data.rewards, maxlen=len(episode_data.rewards))
        tracker.components = deque(episode_data.reward_components, maxlen=len(episode_data.reward_components))
        tracker.episode_reward = episode_data.total_reward
        tracker.step = 0
        
        # Replay each step
        for step, (action, reward, info_step) in enumerate(zip(
            episode_data.actions,
            episode_data.rewards,
            episode_data.info
        )):
            # Step environment
            obs, _, terminated, truncated, _ = env.step(action)
            
            # Get frame
            frame = env.render()
            if frame is None:
                continue
            
            # Overlay rewards
            overlay = self.visualizer.overlay_rewards(
                frame,
                tracker,
                episode_data.episode,
                step,
                episode_data.success and (terminated or truncated)
            )
            
            # Display
            self.visualizer.display_frame(overlay)
            
            # Control playback speed
            delay = int(1000 / (30 * self.playback_speed))  # 30 FPS base
            cv2.waitKey(delay)
            
            tracker.step += 1
            
            if terminated or truncated:
                break
        
        # Keep window open briefly
        cv2.waitKey(2000)
    
    def plot_reward_timeline(
        self,
        episode_data: EpisodeData,
        save_path: Optional[Path] = None
    ):
        """
        Plot reward timeline for an episode.
        
        Args:
            episode_data: EpisodeData to plot
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        steps = np.arange(len(episode_data.rewards))
        
        # Plot total reward
        axes[0].plot(steps, episode_data.rewards, 'b-', linewidth=2, label='Total Reward')
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Reward')
        axes[0].set_title(f'Episode {episode_data.episode} - Reward Timeline')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot reward components
        components = episode_data.reward_components
        if components:
            success_bonuses = [c.success_bonus for c in components]
            distance_penalties = [c.distance_penalty for c in components]
            collision_penalties = [c.collision_penalty for c in components]
            step_penalties = [c.step_penalty for c in components]
            
            axes[1].plot(steps, success_bonuses, 'g-', linewidth=1.5, label='Success Bonus', alpha=0.7)
            axes[1].plot(steps, distance_penalties, 'r-', linewidth=1.5, label='Distance Penalty', alpha=0.7)
            axes[1].plot(steps, collision_penalties, 'm-', linewidth=1.5, label='Collision Penalty', alpha=0.7)
            axes[1].plot(steps, step_penalties, 'y-', linewidth=1.5, label='Step Penalty', alpha=0.7)
            axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Reward Component')
            axes[1].set_title('Reward Components Breakdown')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved reward plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

