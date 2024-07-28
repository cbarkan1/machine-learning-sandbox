import gym
import numpy as np
import matplotlib.pyplot as plt

# Create the CartPole environment
#env = gym.make('CartPole-v1')
env = gym.make('CartPole-v1', render_mode="human")

# Discretize the continuous state space
def discretize_state(state):
    discretized = list()
    for i in range(len(state)):
        if i == 1 or i == 3:  # Ignore velocities
            discretized.append(0)
        else:
            discretized.append(int(round((state[i] + 2.4) / 4.8 * 9)))
    return tuple(discretized)

# Initialize Q-table
state_size = (10, 1, 10, 1)
action_size = env.action_space.n
Q = np.zeros(state_size + (action_size,))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01

# Training parameters
n_episodes = 10000
max_steps = 500

# Training loop
episode_rewards = []

for episode in range(n_episodes):
    state = env.reset()
    state = discretize_state(state[0])  # New gym API returns tuple (state, info)
    total_reward = 0
    
    for step in range(max_steps):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        # Take action and observe new state and reward
        next_state, reward, done, _, _ = env.step(action)  # New gym API returns additional info
        env.render()
        next_state = discretize_state(next_state)
        total_reward += reward
        
        # Q-value update
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state + (best_next_action,)]
        td_error = td_target - Q[state + (action,)]
        Q[state + (action,)] += alpha * td_error
        
        state = next_state
        
        if done:
            break
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    episode_rewards.append(total_reward)
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-100:])}")

# Plot learning curve
plt.plot(episode_rewards)
plt.title('Learning Curve')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

# Test the trained agent
state = env.reset()
state = discretize_state(state[0])
total_reward = 0

for _ in range(max_steps):
    action = np.argmax(Q[state])
    next_state, reward, done, _, _ = env.step(action)
    next_state = discretize_state(next_state)
    total_reward += reward
    state = next_state
    if done:
        break

print(f"Test episode total reward: {total_reward}")
