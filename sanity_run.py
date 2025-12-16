import gymnasium as gym
import highway_env  # registers envs

def main():
    env = gym.make("parking-v0", render_mode="human")
    obs, info = env.reset()

    for _ in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
