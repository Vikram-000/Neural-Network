import gymnasium as gym

def main():
    # Initialize the CartPole environment with human rendering mode
    env = gym.make("CartPole-v1", render_mode="human")

    # Reset the environment to start the first episode
    observation, info = env.reset()

    num_episodes = 5
    max_steps_per_episode = 500

    print("Starting CartPole simulation with random actions...")
    for episode in range(1, num_episodes + 1):
        total_reward = 0
        observation, info = env.reset()

        for step in range(max_steps_per_episode):
            # Sample a random action from the action space (0: push left, 1: push right)
            action = env.action_space.sample()

            # Apply action to the environment
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # If the pole falls too far or cart goes off-screen, or max steps reached
            if terminated or truncated:
                print(f"Episode {episode} finished after {step + 1} steps with total reward: {total_reward}")
                break

    # Close the environment window cleanly
    env.close()
    print("Simulation finished.")

if __name__ == "__main__":
    main()
