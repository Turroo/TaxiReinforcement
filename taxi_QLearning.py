import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def run():          #runs the training
    env = gym.make("Taxi-v3")

    q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 400 x 6 array, q table, starting from all zeros

    learning_rate = 0.90                 #setting up hyperparameters
    n_episodes = 1500
    exploration_episodes = 1200
    max_steps = 400
    epsilon = 1.0
    epsilon_decay = 1.0/exploration_episodes
    discount_factor = 0.9
    rng=np.random.default_rng()         #random number generator
    
    rewards_per_episode = np.zeros(n_episodes)      #initializes a zero value array for episode rewards

    for i in range(n_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        step=0
        episode_reward=0

        while(not terminated and not truncated and step < max_steps):
            step += 1
            if rng.random() < epsilon:
                action = env.action_space.sample()  #chooses a random action
            else:
                action=np.argmax(q[state,:])        #follows q table
            
            next_state, reward, terminated, truncated, _ = env.step(action)

            q[state,action] = q[state,action] + learning_rate * (
                        reward + discount_factor * np.max(q[next_state,:]) - q[state,action]                   #q function
            )
            
            episode_reward+=reward
            state = next_state


        epsilon=max(epsilon - epsilon_decay, 0)     #after each episode, decrease epsilon until it gets to 0

        if(epsilon==0):     #stabilizing q_values after exploration
            learning_rate=0.0001
        rewards_per_episode[i] = episode_reward

        if (i + 1) % 50 == 0:
            print(f"[Episode {i + 1}] Reward: {episode_reward}")
        


    env.close()


    window = 100
    rolling_avg = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_avg, color='blue', label=f"Rolling Average (window={window})")
    plt.title("Average reward per episode (rolling average)")
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.grid(True)
    plt.legend()
    plt.show()

    # episode_bins = 100
    # rewards_grouped = rewards_per_episode.reshape(-1, episode_bins).sum(axis=1)
    # x = np.arange(len(rewards_grouped)) * episode_bins  # Episodi di inizio blocco

    # plt.figure(figsize=(10, 5))
    # plt.plot(x, rewards_grouped, marker='o', linestyle='-', color='blue')
    # plt.title(f'Successi ogni {episode_bins} episodi')
    # plt.xlabel('Episodio')
    # plt.ylabel(f'Successi (su {episode_bins})')
    # plt.grid(True)
    # plt.show()

if __name__ == '__main__':
    run()