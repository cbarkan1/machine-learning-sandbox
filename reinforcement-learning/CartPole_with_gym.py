"""
Uses Q-learning, a model-free RL algorithm
"Q" stands for "quality".
S: state space (discretized)
A: action space
Q: S x A -> Reals


"""


import gym
import numpy as np
import matplotlib.pyplot as plt
from time import time
np.random.seed(0)

# Create CartPole environment
# This environment has only two possible actions: left or right
env = gym.make('CartPole-v1')

# Discretize the state space
def discretize_state(state):
    cart_pos, cart_vel, pole_ang, pole_vel = state
    
    # Actual cart position can be in [-4.8,4.8]
    # BUT the simulation terminates if position is outside of [-2.4,2.4]
    cart_pos_bins = np.linspace(-2.4, 2.4, num_bins)

    # Actual cart velocity can be in [-inf,inf]
    cart_vel_bins = np.linspace(-4, 4, num_bins)

    # Actual pole angle is in [-.42, .42] (It falls once its past this range)
    # But the simulation terminates if angle is outside of [-0.2095, 0.2095]
    pole_ang_bins = np.linspace(-0.2095, 0.2095, num_bins)

    # Actual pole velocity can be in [-inf,inf]
    pole_vel_bins = np.linspace(-4, 4, num_bins)
    
    discretized = [
        np.digitize(cart_pos, cart_pos_bins) - 1,
        np.digitize(cart_vel, cart_vel_bins) - 1,
        np.digitize(pole_ang, pole_ang_bins) - 1,
        np.digitize(pole_vel, pole_vel_bins) - 1
    ]
    return tuple(discretized)

# Initialize Q-table
num_bins = 11
state_space_shape = (num_bins, num_bins, num_bins, num_bins)
num_actions = 2  # Two actions: push cart left, or push cart right.

def train_model(num_episodes, filename):
    # Q.shape = (11, 11, 11, 11, 2), i.e. the shape of S x A
    # state_space + (action_space,) = (11, 11, 11, 11) + (2, ) = (11, 11, 11, 11, 2)
    Q = np.random.uniform(low=-1, high=1, size=state_space_shape + (num_actions,))


    # Hyperparameters
    alpha = 0.5  # Learning rate
    gamma = 0.99  # Discount factor

    # epsilon  is the probability of taking a random action
    # epsilon decayse exponentially at rate epsilon_decay
    epsilon_start = 1.0
    epsilon_end = 0.001
    epsilon_decay = 0.999

    # We play up to 500 steps (this is the maximum allowed by CartPole-v1)
    max_steps = 500

    rewards_per_episode = []

    time0 = time()
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        
        for step in range(max_steps):

            #state_disc are the indices (i1,i2,i3,i4) of the digitized state
            state_disc = discretize_state(state)
            
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # Choose random action
                action = env.action_space.sample()
            else:
                # Choose predicted best action based on current Q
                # Q[state_disc] lists the Q-value for each action (it's a 2-element list)
                action = np.argmax(Q[state_disc])
            
            # See the consequence of the action
            # terminated = True if game is failed (cart or pole go outside of allowed range), False otherwise
            # truncated = True if game has reached 500 steps, False otherwise
            # reward is 1 always, i.e. just taking a step from an allowed state earns a reward
            # hence, total_reward is the total number of steps taken so far.
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # If we failed or reached the limit, done = True
            done = terminated or truncated
            
            # what next state will be, which may be outside of allowed playing range (if game is failed), but NOT outside full bounds!
            next_state_disc = discretize_state(next_state)
            
            # Q update:
            best_next_action = np.argmax(Q[next_state_disc])
            pre_action_quality_estimate = Q[state_disc + (action,)]
            post_action_quality_estimate = reward + gamma * Q[next_state_disc + (best_next_action,)]

            Q[state_disc + (action,)] = alpha*pre_action_quality_estimate + (1-alpha)*post_action_quality_estimate

            state = next_state
            total_reward += reward
            
            if done:
                # if terminated or truncated
                break
        
        rewards_per_episode.append(total_reward)
        

        if episode % 100 == 0:
            time_now = time()
            print(f"Episode {episode}, Average Reward: {np.mean(rewards_per_episode[-100:]):.2f}, Epsilon: {epsilon:.4f}, time/episode: {1000*(time_now-time0)/(episode+1):.2f}ms")

    np.savez(filename+'.npz',Q=Q, rewards_per_episode=rewards_per_episode, max_steps=max_steps)
    return Q, rewards_per_episode


if __name__ == "__main__":
    filename = 'run2'
    num_episodes = 10000
    Q, rewards_per_episode = train_model(num_episodes, filename)

    # Plot the rewards
    plt.figure(figsize=(12, 6))
    rewards_per_episode = np.asarray(rewards_per_episode)
    plt.plot(rewards_per_episode)
    window_size = 50
    moving_average = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_average)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


