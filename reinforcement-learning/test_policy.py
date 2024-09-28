import gym
import numpy as np
import matplotlib.pyplot as plt
from CartPole_with_gym import discretize_state

np.random.seed(0)
env = gym.make('CartPole-v1')

File = np.load('run1.npz')
Q = File['Q']
rewards_per_episode = File['rewards_per_episode']
max_steps = File['max_steps']

fig, (a,b) = plt.subplots(2, 1, figsize=(10, 8))

# Plot training rewards
rewards_per_episode = np.asarray(rewards_per_episode)
a.plot(rewards_per_episode)
window_size = 50
moving_average = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')
a.plot(moving_average)
a.set_title('Rewards per Episode')
a.set_xlabel('Episode')
a.set_ylabel('Total Reward')


# Test the trained agent
num_test_episodes = 100
test_rewards = []

for _ in range(num_test_episodes):
    state, _ = env.reset()
    total_reward = 0
    
    for _ in range(max_steps):
        state_disc = discretize_state(state)
        action = np.argmax(Q[state_disc])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if done:
            break
    
    test_rewards.append(total_reward)

print(f"\nTesting the trained agent over {num_test_episodes} episodes:")
print(f"Average reward: {np.mean(test_rewards):.2f}")
print(f"Standard deviation: {np.std(test_rewards):.2f}")
print(f"Minimum reward: {np.min(test_rewards):.2f}")
print(f"Maximum reward: {np.max(test_rewards):.2f}")

# Plot the distribution of test rewards
b.hist(test_rewards, bins=20, edgecolor='black')
b.set_title('Distribution of Rewards in Test Episodes')
b.set_xlabel('Total Reward')
b.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
