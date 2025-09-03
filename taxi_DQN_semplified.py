from collections import deque
import gymnasium as gym
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self,in_states,h1_nodes,h2_nodes,out_actions):
        super().__init__()
        
        #defining network layers
        self.fc1 = nn.Linear(in_states,h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)  
        self.out = nn.Linear(h2_nodes,out_actions)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    

#experience replay class   
class ReplayMemory():
    def __init__(self,maximumLen):
        self.memory = deque([],maxlen = maximumLen)

    def append(self,transition):
        self.memory.append(transition)

    def sample(self,sample_size):
        return random.sample(self.memory,sample_size)
    
    def __len__(self):
        return len(self.memory)
    
class TaxiDQL():
    
    # Hyperparameters 
    learning_rate_a = 0.005         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)    
    replay_memory_size = 10000      # size of replay memory
    mini_batch_size = 32            # size of the training data set sampled from the replay memory
    max_steps = 300                 # maximum number of steps
   
    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later

    ACTIONS = ['D','U','R','L','PU','DO']   #for debug

    def train(self,episodes):
        env = gym.make("Taxi-v3")
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        

        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)      #initialization of replay memory (size 1000 in this case)

        # creation of policy network (singola rete, niente target)
        policy_dqn = DQN(in_states=num_states, h1_nodes=64,h2_nodes=64, out_actions=num_actions)


        # declaring Policy network optimizer "Adam" (edits weight based on some parameters)
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay over time
        epsilon_history = []


        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False
            truncated = False
            step = 0
            episode_reward=0

            while(not terminated and not truncated and step < self.max_steps):
                step += 1
                if random.random() < epsilon:
                    action = env.action_space.sample()  #chooses a random action
                else:
                    # select best action            
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()     #policy network used to calculate the set of Q values and take the maximum value one
                
                new_state, reward, terminated, truncated, _ = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated))

                episode_reward+=reward
                
                # Move to the next state
                state = new_state
            
            rewards_per_episode[i] = episode_reward

            # --- DEBUG ---
            if (i+1) % 50 == 0:
                print(f'Episode {i+1}/{episodes} | Reward: {episode_reward} | Epsilon: {epsilon:.3f} | Memory size: {len(memory)}')

            # Optimization. Check if enough experience has been collected and if there is at least 1 value >= 0, otherwise no point in optimization
            if len(memory)>self.mini_batch_size and np.any(rewards_per_episode >=0):
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn)        

            #epsilon decay
            epsilon_start = 1.0
            epsilon_min = 0.01
            decay_episodes = 13000

            if i < decay_episodes:
                epsilon = epsilon_start - (i / decay_episodes) * (epsilon_start - epsilon_min)
            else:
                epsilon = epsilon_min

        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "taxi_dql.pt")

        plt.figure(figsize=(12,6))
        plt.plot(rewards_per_episode, color='blue', label='Reward per episode')
        plt.title("Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()
        plt.show()

        rewards_per_episode = np.array(rewards_per_episode)  # converte in array NumPy

        window = 100  # finestra della media mobile
        moving_avg = np.zeros_like(rewards_per_episode, dtype=float)

        for i in range(len(rewards_per_episode)):
            moving_avg[i] = np.mean(rewards_per_episode[max(0, i-window+1):(i+1)])

        plt.figure(figsize=(12,6))
        plt.plot(moving_avg, color='orange', label=f"Moving Average (window={window})")
        plt.title("Reward per Episode (Moving Average)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()
        plt.show()

    def optimize(self, mini_batch, policy_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []     #output layers values of policy network
        target_q_list = []      #target values calcolati manualmente

        for state, action, new_state, reward, terminated in mini_batch:
            
            #applicazione della formula DQN
            if terminated: 
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value usando la stessa rete
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * policy_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Calculating the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Creiamo una copia dei Q-values correnti e aggiorniamo solo l’azione scelta
            target_q = current_q.clone().detach()
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss function for the whole minibatch. Taking target q values and using them to train the current q values.
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Converts an state (int) to a tensor representation.
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor


    # Run the Taxi environment with the learned policy
    def test(self, episodes):
        env = gym.make("Taxi-v3")
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=64,h2_nodes=64, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load("taxi_dql.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      
            truncated = False
            step = 0
            episode_reward = 0

            print(f"\n--- Episode {i+1} ---")

            while(not terminated and not truncated and step < self.max_steps):  
                step += 1

                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                # Optional: print each step
                # print(f"Step {step}: State {state}, Action {self.ACTIONS[action]}, Reward {reward}")

                state = new_state

            print(f"Total Reward: {episode_reward}")

        env.close()




if __name__ == '__main__':

    taxi = TaxiDQL()
    taxi.train(15000)
    taxi.test(10)
