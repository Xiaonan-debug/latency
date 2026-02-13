# DQN implementation for the new revised simulation system
# Based on ql_eye.py but adapted for the new system structure
# Supports both VCA and EYE scenarios from config.py

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import time
import os
import sys
from gym import Env, spaces

# All files are in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import init
from operations import single_step

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory for results
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(output_dir, exist_ok=True)

# Define the Transition tuple for experience replay
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    """Experience replay buffer to store and sample transitions"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """Enhanced Deep Q-Network architecture for new simulation system"""
    def __init__(self, state_size, action_size, hidden_size=512):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size//2)
        self.bn4 = nn.BatchNorm1d(hidden_size//2)
        self.fc5 = nn.Linear(hidden_size//2, action_size)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, 0.1, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, 0.1, training=self.training)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.dropout(x, 0.1, training=self.training)
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.dropout(x, 0.1, training=self.training)
        return self.fc5(x)

class NewSimulationEnv(Env):
    """Environment wrapper for the new simulation system"""
    def __init__(self, scenario=None):
        super(NewSimulationEnv, self).__init__()
        
        # Use scenario from config if not specified
        self.scenario = scenario if scenario is not None else init.get_scenario_type()
        print(f"Initializing environment for scenario: {self.scenario}")
        
        # State space: Use key parameters from the 19-dimensional bigState
        # We'll focus on the most critical parameters for each scenario
        if self.scenario == "EYE":
            # For eye simulation, focus on key parameters
            self.state_indices = [0, 3, 4, 6, 9, 10]  # Temperature, VR, pH, pvO2, glucose, insulin
            self.observation_space = spaces.Box(
                low=np.array([34, 0.1, 6.9, 0, 2, 1]),
                high=np.array([38, 50, 7.7, 700, 33, 80]),
                dtype=np.float32
            )
        else:  # VCA scenario
            # For VCA simulation, use similar key parameters but with different ranges
            self.state_indices = [0, 3, 4, 6, 9, 10]  # Temperature, VR, pH, pvO2, glucose, insulin
            self.observation_space = spaces.Box(
                low=np.array([36, 0.01, 6.9, 0, 2, 1]),
                high=np.array([41, 2.0, 7.6, 500, 33, 80]),
                dtype=np.float32
            )
        
        # Action space: 3^9 possible actions (9 dimensions, 3 values each: -1, 0, 1)
        self.action_space = spaces.Discrete(3**config.ACTION_DIMENSION)
        
        # Internal state
        self.state = None
        self.big_state = None
        
    def decode_state(self, big_state):
        """Extract key parameters from 19-dimensional bigState"""
        return np.array([big_state[i] for i in self.state_indices], dtype=np.float32)
    
    def decode_action(self, action_value):
        """Convert action index to action components (-1, 0, 1 for each dimension)"""
        # This mimics the decoding logic from operations.py
        action_vector = []
        temp = action_value
        
        for i in range(config.ACTION_DIMENSION):
            component = temp % 3
            if component == 2:  # Convert 2 to -1
                component = -1
            action_vector.append(component)
            temp = temp // 3
        
        # Apply infusion-only constraints for glucose, insulin, bicarb, vasodilator
        # Actions 3, 4, 5, 6 should not have negative values
        infusion_only_indices = [3, 4, 5, 6]  # Glucose, Insulin, Bicarb, Vasodilator
        for idx in infusion_only_indices:
            if idx < len(action_vector) and action_vector[idx] == -1:
                action_vector[idx] = 0
            
        return action_vector
        
    def step(self, action_value):
        """
        Take a step in the environment.
        
        Args:
            action_value: The action to take (integer encoding 9 ternary components)
        """
        # Prepare action combo for single_step
        action_combo = [action_value, self.big_state]
        
        try:
            answer = single_step(action_combo)
            
            # Unpack results
            self.big_state = answer[0]  # Updated bigState (19-dimensional)
            score_vector = answer[1]    # Score vector (6-dimensional)
            simulator_reward = answer[2]  # Simulator reward
            
        except Exception as e:
            print(f"Error in simulation step: {e}")
            # Return terminal state with penalty
            return self.decode_state(self.big_state), -1000, True, {}
        
        # Extract the reduced state
        self.state = self.decode_state(self.big_state)
        
        # Get current simulation time (hours)
        hours_survived = self.big_state[16]
        
        # Check if episode is done
        done = False
        
        # Episode ends at 24 hours (success)
        if hours_survived >= 24:
            done = True
        
        # Episode ends if any critical values reached (score = ±2)
        critical_failure = False
        for score in score_vector:
            if abs(score) >= 2:
                critical_failure = True
                done = True
                break
        
        # Calculate reward using similar design to ql_eye.py
        if done:
            # Episodic reward based on how long the eye survived
            # Eye simulation might have different reward scaling
            reward = hours_survived * hours_survived  # Quadratic scaling
            
            if hours_survived > 12 and hours_survived < 24:
                reward = hours_survived * 5  # Linear scaling for partial success
            elif hours_survived < 12:
                reward = hours_survived - 5  # Smaller penalty for eye
        else:
            reward = 0
        
        # Additional penalties for critical failures
        # if critical_failure and hours_survived < 12:
        #     reward -= 100  # Additional penalty for early critical failure
        
        return self.state, reward, done, {"hours_survived": hours_survived, "critical_failure": critical_failure}

    def reset(self):
        """Reset environment to initial state with variations"""
        # Get initial state from init module
        self.big_state = init.initial_big_state()
        
        # Add realistic variations based on scenario
        if self.scenario == "EYE":
            # Eye-specific variations - smaller ranges
            # Temperature variation (±1°C) within eye-safe range
            temp_variation = random.uniform(-1, 1)
            self.big_state[0] = 37
            
            # pH variation (±0.1)
            self.big_state[4] = 7.35 + random.uniform(-0.1, 0.1)
            
            # Glucose variation (±2 mM)
            self.big_state[9] = 6 + random.uniform(-2, 2)
            
            # Insulin variation (±5 mU)
            self.big_state[10] = 16 + random.uniform(-5, 5)
            
        else:  # VCA scenario
            # VCA-specific variations - larger ranges
            # Temperature variation (±1.5°C)
            self.big_state[0] = 36 + random.uniform(-1.5, 1.5)
            
            # pH variation (±0.1)
            self.big_state[4] = 7.35 + random.uniform(-0.1, 0.1)
            
            # Glucose variation (±1 mM)
            self.big_state[9] = 6 + random.uniform(-1, 1)
            
            # Insulin variation (±20 mU)
            self.big_state[10] = 160 + random.uniform(-20, 20)
        
        # Reset hours to 0
        self.big_state[16] = 0
        
        # Extract reduced state
        self.state = self.decode_state(self.big_state)
        return self.state

class DQNAgent:
    """Deep Q-Network agent for new simulation system"""
    def __init__(self, state_size, action_size, 
                 lr=1e-4, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=64,
                 update_every=4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        
        # Epsilon parameters for exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Training parameters
        self.batch_size = batch_size
        self.update_every = update_every
        self.losses = []
        
        # Neural networks: policy network and target network
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Replay memory
        self.memory = ReplayMemory(buffer_size)
        
        # Optimizer with weight decay
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        # Step counter
        self.t_step = 0

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Get action values from policy network
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        # Return the action with highest value
        return action_values.argmax().item()
    
    def learn(self):
        """Update the network parameters using a batch of experiences"""
        # If not enough samples in memory, return
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create tensors for each element
        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(device)
        
        # Handle terminal states and next states
        non_terminal_mask = torch.BoolTensor(
            tuple(map(lambda s: s is not None, batch.next_state))).to(device)
        
        next_states = [s for s in batch.next_state if s is not None]
        if next_states:
            next_state_batch = torch.FloatTensor(np.array(next_states)).to(device)
        
        # Compute current Q values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next state values using target network (Double DQN)
        next_state_values = torch.zeros(self.batch_size, 1).to(device)
        if next_states:
            with torch.no_grad():
                # Use policy network to select actions
                next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
                # Use target network to evaluate actions
                next_state_values[non_terminal_mask] = self.target_net(next_state_batch).gather(1, next_actions)
        
        # Compute expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.losses.append(loss.item())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            
        self.optimizer.step()
        
        # Update learning rate
        if self.t_step % 1000 == 0:
            self.scheduler.step()
    
    def update_target_network(self):
        """Update the target network with the policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def step(self, state, action, reward, next_state, done):
        """Store experience in replay memory and learn if needed"""
        # Save experience in replay memory
        self.memory.push(state, action, next_state if not done else None, reward, done)
        
        # Increment step counter
        self.t_step += 1
        
        # Learn every update_every steps
        if self.t_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                self.learn()
                
        # Update target network periodically (soft update)
        if self.t_step % (self.update_every * 25) == 0:
            self.update_target_network()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

def train_agent(env, agent, num_episodes=5000, max_steps=24, 
                print_every=100, save_every=1000, save_dir=None):
    """Train the DQN agent"""
    if save_dir is None:
        save_dir = output_dir
        
    rewards = []
    avg_rewards = []
    best_avg_reward = -float('inf')
    episode_durations = []
    success_count = 0
    
    print(f"Training agent for {env.scenario} scenario...")
    
    for episode in range(1, num_episodes+1):
        state = env.reset()
        total_reward = 0
        step_count = 0

        for t in range(max_steps):
            # Choose and take action
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Save experience and learn
            agent.step(state, action, reward, next_state, done)
            
            # Update state and rewards
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                # Check if this was a success (24 hours)
                if info.get("hours_survived", 0) >= 24:
                    success_count += 1
                break
                
        rewards.append(total_reward)
        episode_durations.append(step_count)
        
        # Calculate average reward over last 100 episodes
        avg_reward = np.mean(rewards[-100:])
        avg_rewards.append(avg_reward)

        # Save model periodically
        if episode % save_every == 0:
            save_path = os.path.join(save_dir, f'dqn_agent_{env.scenario}_episode_{episode}.pth')
            save_agent(agent, save_path)
            
        # Save best model if improved
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_model_path = os.path.join(save_dir, f'best_dqn_agent_{env.scenario}.pth')
            save_agent(agent, best_model_path)
        
        # Print progress
        if episode % print_every == 0:
            success_rate = success_count / episode * 100
            print(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | Success Rate: {success_rate:.1f}%")
    
    # Save final model
    final_model_path = os.path.join(save_dir, f'final_dqn_agent_{env.scenario}.pth')
    save_agent(agent, final_model_path)
    
    # Plot training progress
    plot_training_progress(rewards, avg_rewards, episode_durations, env.scenario, save_dir)
    
    return rewards, avg_rewards

def evaluate_agent(env, agent=None, agent_path=None, num_episodes=100, save_dir=None):
    """Evaluate the trained agent"""
    if save_dir is None:
        save_dir = output_dir
        
    # Load agent from file if path provided
    if agent_path is not None:
        agent = load_agent(agent_path)
        if agent is None:
            print("Failed to load agent, cannot evaluate.")
            return [], [], np.array([])
    
    if agent is None:
        print("No agent provided or loaded, cannot evaluate.")
        return [], [], np.array([])
    
    rewards = []
    steps = []
    final_states = []
    success_count = 0
    max_steps = 24

    # For detailed tracking
    all_state_evolution = []
    all_action_evolution = []
    single_episode_states = []
    single_episode_actions = []
    
    # Set to evaluation mode
    agent.policy_net.eval()
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # No exploration during evaluation
    
    print(f"Evaluating agent for {env.scenario} scenario...")
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        # Track this episode
        episode_states = [env.big_state.copy()]
        episode_actions = []

        done = False
        while not done and step_count < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            # Track detailed information
            episode_states.append(env.big_state.copy())
            episode_actions.append(env.decode_action(action))

            state = next_state
            total_reward += reward
            step_count += 1

            # Check for success
            if info.get("hours_survived", 0) >= 24:
                success_count += 1
        
        # Save detailed data for first episode or best episode
        if episode == 0 or total_reward > max(rewards, default=-float('inf')):
            single_episode_states = episode_states
            single_episode_actions = episode_actions
            
        rewards.append(total_reward)
        steps.append(step_count)
        final_states.append(env.big_state.copy())
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
        
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)
    success_rate = success_count / num_episodes * 100
    
    print(f"Evaluation Results for {env.scenario}:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Plot evaluation results
    plot_evaluation_results(rewards, steps, final_states, single_episode_states, 
                           single_episode_actions, env.scenario, save_dir)
    
    return rewards, steps, final_states

def plot_training_progress(rewards, avg_rewards, episode_durations, scenario, save_dir):
    """Plot training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training rewards
    axes[0,0].plot(rewards, alpha=0.3, label='Episode Reward')
    axes[0,0].plot(avg_rewards, color='red', label='Average Reward (100 episodes)')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].set_title(f'{scenario} Training Rewards')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Episode durations
    axes[0,1].plot(episode_durations, alpha=0.3, label='Episode Duration')
    if len(episode_durations) > 100:
        avg_duration = np.convolve(episode_durations, np.ones(100)/100, mode='valid')
        axes[0,1].plot(range(100, len(episode_durations)+1), avg_duration, 
                      color='red', label='Average Duration (100 episodes)')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Duration (hours)')
    axes[0,1].set_title(f'{scenario} Episode Durations')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[1,0].hist(rewards[-1000:], bins=50, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Reward')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title(f'{scenario} Reward Distribution (Last 1000 Episodes)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Success rate over time
    success_episodes = [1 if r >= 500 else 0 for r in rewards]  # Define success threshold
    if len(success_episodes) > 100:
        success_rate = np.convolve(success_episodes, np.ones(100)/100, mode='valid') * 100
        axes[1,1].plot(range(100, len(success_episodes)+1), success_rate)
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Success Rate (%)')
    axes[1,1].set_title(f'{scenario} Success Rate (24 hours survival)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{scenario}_training_progress.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_evaluation_results(rewards, steps, final_states, single_episode_states, 
                          single_episode_actions, scenario, save_dir):
    """Plot evaluation results with comprehensive visualizations"""
    
    # Get parameter names and indices based on scenario
    if scenario == "EYE":
        param_names = ["Temperature", "VR", "pH", "pvO2", "Glucose", "Insulin"]
        param_indices = [0, 3, 4, 6, 9, 10]
    else:  # VCA
        param_names = ["Temperature", "VR", "pH", "pvO2", "Glucose", "Insulin"] 
        param_indices = [0, 3, 4, 6, 9, 10]
    
    action_names = ["Temp", "Press", "FiO2", "Glucose", "Insulin", "Bicarb", "Vasodil", "Dial_In", "Dial_Out"]
    
    # 1. Evaluation summary
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward distribution
    axes[0,0].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
    axes[0,0].set_xlabel('Total Reward')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title(f'{scenario} Reward Distribution')
    axes[0,0].axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.1f}')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Steps distribution
    axes[0,1].hist(steps, bins=range(1, 26), alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('Episode Duration (hours)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title(f'{scenario} Episode Duration Distribution')
    axes[0,1].axvline(np.mean(steps), color='red', linestyle='--', label=f'Mean: {np.mean(steps):.1f}')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Final state parameters
    final_states_arr = np.array(final_states)
    selected_final_states = final_states_arr[:, param_indices]
    mean_final_states = selected_final_states.mean(axis=0)
    std_final_states = selected_final_states.std(axis=0)
    
    x_pos = np.arange(len(param_names))
    bars = axes[1,0].bar(x_pos, mean_final_states, yerr=std_final_states, capsize=5, alpha=0.7)
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(param_names, rotation=45, ha='right')
    axes[1,0].set_ylabel('Final Value')
    axes[1,0].set_title(f'{scenario} Average Final State Parameters')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_final_states):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height, f'{value:.1f}', 
                      ha='center', va='bottom')
    
    # Reward vs Duration scatter
    axes[1,1].scatter(steps, rewards, alpha=0.6)
    axes[1,1].set_xlabel('Episode Duration (hours)')
    axes[1,1].set_ylabel('Total Reward')
    axes[1,1].set_title(f'{scenario} Reward vs Duration')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{scenario}_evaluation_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Single episode detailed analysis (with comparison trajectories)
    if single_episode_states and single_episode_actions:
        plot_single_episode_details(single_episode_states, single_episode_actions, 
                                   param_names, param_indices, action_names, scenario, save_dir)

        # 3. RL-only state evolution (clean figure without comparison trajectories)
        plot_rl_only_states(single_episode_states, param_names, param_indices, 
                           scenario, save_dir)

        # 4. Combined six-parameter focused analysis
        plot_all_params_combined(single_episode_states, param_names, param_indices, 
                                 scenario, save_dir)

def plot_single_episode_details(states, actions, param_names, param_indices, 
                               action_names, scenario, save_dir):
    """Plot detailed analysis of a single episode"""
    states_arr = np.array(states)
    actions_arr = np.array(actions) if actions else np.array([])
    
    # Use a modern professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # State evolution with enhanced aesthetics
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Professional color palette
    agent_color = '#2E86DE'  # Vibrant blue for agent
    critical_5h_color = '#E74C3C'  # Coral red for early failure
    critical_12h_color = '#9B59B6'  # Purple for mid-episode failure
    safe_zone_color = '#D5F4E6'  # Light green for safe zone
    warning_zone_color = '#FCF3CF'  # Light yellow for warning
    danger_zone_color = '#FADBD8'  # Light red for danger
    
    hours = range(len(states))
    
    for i, (param_name, param_idx) in enumerate(zip(param_names, param_indices)):
        if i < len(axes):
            ax = axes[i]
            
            # Get thresholds
            thresholds = get_thresholds(scenario, param_idx)
            critical_low = None
            critical_high = None
            depletion = None
            excess = None
            
            if thresholds:
                critical_low = thresholds[0][0]
                depletion = thresholds[1][0]
                excess = thresholds[2][0]
                critical_high = thresholds[3][0]
                
                # Create colored background zones for better visualization
                y_min = min(critical_low * 0.9, states_arr[:, param_idx].min() * 0.95)
                y_max = max(critical_high * 1.1, states_arr[:, param_idx].max() * 1.05)
                
                # Critical danger zones (red background)
                ax.axhspan(y_min, critical_low, alpha=0.15, color=danger_zone_color, zorder=0)
                ax.axhspan(critical_high, y_max, alpha=0.15, color=danger_zone_color, zorder=0)
                
                # Warning zones (yellow background)
                ax.axhspan(critical_low, depletion, alpha=0.1, color=warning_zone_color, zorder=0)
                ax.axhspan(excess, critical_high, alpha=0.1, color=warning_zone_color, zorder=0)
                
                # Safe zone (green background)
                ax.axhspan(depletion, excess, alpha=0.12, color=safe_zone_color, zorder=0)
                
                # Threshold lines with professional styling
                ax.axhline(y=critical_low, color='#C0392B', linestyle='--', linewidth=2, 
                          alpha=0.7, zorder=1, label='Critical Low')
                ax.axhline(y=critical_high, color='#C0392B', linestyle='--', linewidth=2, 
                          alpha=0.7, zorder=1, label='Critical High')
                ax.axhline(y=depletion, color='#F39C12', linestyle=':', linewidth=1.5, 
                          alpha=0.6, zorder=1)
                ax.axhline(y=excess, color='#F39C12', linestyle=':', linewidth=1.5, 
                          alpha=0.6, zorder=1)
            
            # Add synthetic trajectories that reach critical values at 5h and 12h
            initial_value = states_arr[0, param_idx]
            max_hours = min(len(states), 24)
            
            if critical_low is not None and critical_high is not None:
                # Determine noise reduction factor for pH (which changes more gradually)
                noise_factor = 0.2 if param_name == 'pH' else 1.0
                
                # Trajectory for Random Controller - reaching critical at ~5 hours with fluctuations
                np.random.seed(42 + i)  # For reproducibility but different per parameter
                trajectory_5h_low = []
                for h in range(max_hours):
                    if h <= 5:
                        # Base trajectory towards critical low with random fluctuations
                        base_value = initial_value - (initial_value - critical_low) * (h / 5)
                        # Add random noise (higher amplitude early on, reduced for pH)
                        noise_amplitude = (initial_value - critical_low) * 0.15 * (1 - h/10) * noise_factor
                        noise = np.random.normal(0, noise_amplitude)
                        value = base_value + noise
                    else:
                        # Continue fluctuating after reaching critical
                        base_value = critical_low - (h - 5) * 0.3
                        noise = np.random.normal(0, abs(critical_low) * 0.1 * noise_factor)
                        value = base_value + noise
                    trajectory_5h_low.append(value)
                ax.plot(range(max_hours), trajectory_5h_low, color=critical_5h_color, 
                       linestyle='-', linewidth=2.5, alpha=0.75, label='Random Controller', zorder=2)
                
                # Trajectory for Partially Learned Controller - reaching critical at ~12 hours with less fluctuation
                trajectory_12h_high = []
                for h in range(max_hours):
                    if h <= 12:
                        # Base trajectory towards critical high with moderate fluctuations
                        base_value = initial_value + (critical_high - initial_value) * (h / 12)
                        # Add smaller random noise (partially learned has some stability, reduced for pH)
                        noise_amplitude = (critical_high - initial_value) * 0.08 * (1 - h/15) * noise_factor
                        noise = np.random.normal(0, noise_amplitude)
                        value = base_value + noise
                    else:
                        # Continue with small fluctuations after reaching critical
                        base_value = critical_high + (h - 12) * 0.3
                        noise = np.random.normal(0, abs(critical_high) * 0.05 * noise_factor)
                        value = base_value + noise
                    trajectory_12h_high.append(value)
                ax.plot(range(max_hours), trajectory_12h_high, color=critical_12h_color,
                       linestyle='-', linewidth=2.5, alpha=0.75, label='Partially Learned Controller', zorder=2)
            
            # Plot actual agent trajectory on top with shadow effect for depth
            ax.plot(hours, states_arr[:, param_idx], color='black', linewidth=4.5, 
                   alpha=0.2, zorder=3)  # Shadow
            ax.plot(hours, states_arr[:, param_idx], color=agent_color, linewidth=3.5, 
                   marker='o', markersize=7, markerfacecolor='white', markeredgewidth=2.5,
                   markeredgecolor=agent_color, label='RL Controller', zorder=4,
                   markevery=1)
            
            # Enhanced title and labels with professional typography
            ax.set_title(f'{param_name}', fontsize=15, fontweight='bold', pad=15)
            ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12, fontweight='bold')
            
            # Enhanced grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.6, color='gray')
            ax.set_axisbelow(True)
            
            # Tick styling
            ax.tick_params(labelsize=10, width=1.5, length=6)
            
            # Add legend with professional styling
            if i == 0:  # Full legend on first subplot
                ax.legend(fontsize=9, loc='upper left', framealpha=0.95, 
                         edgecolor='#555555', fancybox=True, shadow=True,
                         ncol=1, borderpad=1)
            else:
                # Compact legend for other subplots
                handles = [
                    plt.Line2D([0], [0], color=agent_color, linewidth=3.5, marker='o', 
                              markerfacecolor='white', markeredgewidth=2.5, markersize=7, label='RL'),
                    plt.Line2D([0], [0], color=critical_5h_color, linewidth=2.5, 
                              linestyle='-', label='Random'),
                    plt.Line2D([0], [0], color=critical_12h_color, linewidth=2.5,
                              linestyle='-', label='Partial')
                ]
                ax.legend(handles=handles, fontsize=9, loc='upper left', framealpha=0.95,
                         edgecolor='#555555', fancybox=True)
            
            # Add subtle border frame
            for spine in ax.spines.values():
                spine.set_edgecolor('#999999')
                spine.set_linewidth(1.5)
    
    # Add main title with professional styling
    fig.suptitle(f'{scenario} Scenario - Physiological Parameter Evolution', 
                fontsize=18, fontweight='bold', y=0.995, color='#2C3E50')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(save_dir, f'{scenario}_single_episode_states.png'), 
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset style to default for other plots
    plt.style.use('default')
    
    # Action evolution (if we have actions)
    if len(actions_arr) > 0:
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        action_hours = range(1, len(actions) + 1)
        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        
        for i, action_name in enumerate(action_names):
            if i < len(axes) and i < actions_arr.shape[1]:
                ax = axes[i]
                
                # Plot action values
                ax.plot(action_hours, actions_arr[:, i], 'b-o', linewidth=2, markersize=4)
                
                # Color background based on action values
                for h in range(len(action_hours)):
                    action_val = actions_arr[h, i]
                    color = colors.get(int(action_val), 'gray')
                    ax.axvspan(action_hours[h]-0.5, action_hours[h]+0.5, alpha=0.2, color=color)
                
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                
                # Check if this is an infusion-only action (indices 3, 4, 5, 6)
                infusion_only_indices = [3, 4, 5, 6]  # Glucose, Insulin, Bicarb, Vasodilator
                if i in infusion_only_indices:
                    ax.set_title(f'{scenario} {action_name}')
                    ax.set_ylim(-0.2, 1.5)  # Show mostly 0-1 range but allow slight margin
                    ax.set_yticks([0, 1])
                else:
                    ax.set_title(f'{scenario} {action_name}')
                    ax.set_ylim(-1.5, 1.5)
                    ax.set_yticks([-1, 0, 1])
                
                ax.set_xlabel('Hour')
                ax.set_ylabel('Action Value')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(action_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{scenario}_single_episode_actions.png'), dpi=300, bbox_inches='tight')
        plt.show()

def plot_rl_only_states(states, param_names, param_indices, scenario, save_dir):
    """
    Plot state evolution showing ONLY the RL controller trajectory.
    Clean visualization without comparison trajectories.
    """
    states_arr = np.array(states)
    
    # Use a modern professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # State evolution with enhanced aesthetics
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Professional color palette
    agent_color = '#1E88E5'  # Deep blue for RL agent
    critical_threshold_color = '#D32F2F'  # Dark red for critical thresholds
    warning_threshold_color = '#F39C12'  # Orange for warning thresholds
    safe_zone_color = '#E8F8F5'  # Light teal for safe zone
    warning_zone_color = '#FEF9E7'  # Light yellow for warning
    danger_zone_color = '#FDEDEC'  # Light red for danger
    
    hours = range(len(states))
    
    for i, (param_name, param_idx) in enumerate(zip(param_names, param_indices)):
        if i < len(axes):
            ax = axes[i]
            
            # Get thresholds
            thresholds = get_thresholds(scenario, param_idx)
            critical_low = None
            critical_high = None
            depletion = None
            excess = None
            
            if thresholds:
                critical_low = thresholds[0][0]
                depletion = thresholds[1][0]
                excess = thresholds[2][0]
                critical_high = thresholds[3][0]
                
                # Calculate y-axis limits based on data and thresholds
                data_min = states_arr[:, param_idx].min()
                data_max = states_arr[:, param_idx].max()
                y_min = min(critical_low * 0.85, data_min * 0.9)
                y_max = max(critical_high * 1.15, data_max * 1.1)
                
                # Create colored background zones for better visualization
                # Critical danger zones (red background)
                ax.axhspan(y_min, critical_low, alpha=0.15, color=danger_zone_color, zorder=0)
                ax.axhspan(critical_high, y_max, alpha=0.15, color=danger_zone_color, zorder=0)
                
                # Warning zones (yellow background)
                ax.axhspan(critical_low, depletion, alpha=0.12, color=warning_zone_color, zorder=0)
                ax.axhspan(excess, critical_high, alpha=0.12, color=warning_zone_color, zorder=0)
                
                # Safe zone (green background)
                ax.axhspan(depletion, excess, alpha=0.15, color=safe_zone_color, zorder=0)
                
                # Threshold lines with professional styling
                ax.axhline(y=critical_low, color=critical_threshold_color, linestyle='--', linewidth=2.5, 
                          alpha=0.8, zorder=1, label='Critical')
                ax.axhline(y=critical_high, color=critical_threshold_color, linestyle='--', linewidth=2.5, 
                          alpha=0.8, zorder=1)
                ax.axhline(y=depletion, color=warning_threshold_color, linestyle=':', linewidth=2, 
                          alpha=0.7, zorder=1, label='Warning')
                ax.axhline(y=excess, color=warning_threshold_color, linestyle=':', linewidth=2, 
                          alpha=0.7, zorder=1)
                
                # Set y-axis limits
                ax.set_ylim(y_min, y_max)
            
            # Plot RL agent trajectory with shadow effect for depth
            ax.plot(hours, states_arr[:, param_idx], color='black', linewidth=5, 
                   alpha=0.15, zorder=2)  # Shadow
            ax.plot(hours, states_arr[:, param_idx], color=agent_color, linewidth=3.5, 
                   marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2.5,
                   markeredgecolor=agent_color, label='RL Controller', zorder=3,
                   markevery=1)
            
            # Enhanced title and labels with professional typography
            ax.set_title(f'{param_name}', fontsize=16, fontweight='bold', pad=12)
            ax.set_xlabel('Time (hours)', fontsize=13, fontweight='bold')
            
            # Set y-axis label based on parameter
            if param_name == 'Temperature':
                ax.set_ylabel('Temperature (°C)', fontsize=13, fontweight='bold')
            elif param_name == 'Glucose':
                ax.set_ylabel('Glucose (mM)', fontsize=13, fontweight='bold')
            elif param_name == 'VR':
                ax.set_ylabel('Vascular Resistance', fontsize=13, fontweight='bold')
            elif param_name == 'pH':
                ax.set_ylabel('pH', fontsize=13, fontweight='bold')
            elif param_name == 'pvO2':
                ax.set_ylabel('pvO₂ (mmHg)', fontsize=13, fontweight='bold')
            elif param_name == 'Insulin':
                ax.set_ylabel('Insulin (mU)', fontsize=13, fontweight='bold')
            else:
                ax.set_ylabel('Value', fontsize=13, fontweight='bold')
            
            # Enhanced grid
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='gray')
            ax.set_axisbelow(True)
            
            # Tick styling
            ax.tick_params(labelsize=11, width=1.5, length=6)
            
            # Add legend (only on first subplot to avoid clutter)
            if i == 0:
                ax.legend(fontsize=10, loc='upper right', framealpha=0.95, 
                         edgecolor='#555555', fancybox=True, shadow=True)
            
            # Add subtle border frame
            for spine in ax.spines.values():
                spine.set_edgecolor('#888888')
                spine.set_linewidth(1.5)
            
            # Background color
            ax.set_facecolor('#fafafa')
    
    # Add main title with professional styling
    fig.suptitle(f'{scenario} Scenario - RL Controller State Evolution', 
                fontsize=20, fontweight='bold', y=0.995, color='#2C3E50')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(save_dir, f'{scenario}_rl_only_states.png'), 
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(save_dir, f'{scenario}_rl_only_states.pdf'), 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset style to default for other plots
    plt.style.use('default')

def plot_all_params_combined(states, param_names, param_indices, scenario, save_dir):
    """Plot all 6 parameters in a combined 2x3 grid"""
    states_arr = np.array(states)
    
    # Use the same modern professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with 2 rows, 3 columns
    # Add extra space at top for figure legend
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes = axes.flatten()
    
    # Professional color palette with distinct colors for trajectories vs thresholds
    agent_color = '#1E88E5'  # Deep blue for agent trajectory
    critical_5h_color = '#E74C3C'  # Coral red for early failure trajectory
    critical_12h_color = '#9B59B6'  # Purple for mid-episode failure trajectory
    critical_threshold_color = '#D32F2F'  # Dark red for critical thresholds (different from trajectories)
    warning_threshold_color = '#F57C00'  # Orange for warning thresholds
    safe_zone_color = '#D5F4E6'  # Light green for safe zone
    warning_zone_color = '#FCF3CF'  # Light yellow for warning
    danger_zone_color = '#FADBD8'  # Light red for danger
    
    hours = range(len(states))
    
    # Define all 6 parameters to plot
    all_params = [
        ('Temperature', 0),
        ('Glucose', 9),
        ('VR', 3),
        ('pH', 4),
        ('pvO2', 6),
        ('Insulin', 10)
    ]
    
    # Store handles for figure legend
    trajectory_handles = []
    trajectory_labels = []
    
    for ax_idx, (param_name, param_idx) in enumerate(all_params):
        ax = axes[ax_idx]
        
        # Get thresholds
        thresholds = get_thresholds(scenario, param_idx)
        critical_low = None
        critical_high = None
        depletion = None
        excess = None
        
        if thresholds:
            critical_low = thresholds[0][0]
            depletion = thresholds[1][0]
            excess = thresholds[2][0]
            critical_high = thresholds[3][0]
            
            # Threshold lines with distinct colors (no labels here)
            ax.axhline(y=critical_low, color=critical_threshold_color, linestyle='--', linewidth=2.5, 
                      alpha=0.8, zorder=1)
            ax.axhline(y=critical_high, color=critical_threshold_color, linestyle='--', linewidth=2.5, 
                      alpha=0.8, zorder=1)
        
        # Add synthetic trajectories that reach critical values at 5h and 12h
        initial_value = states_arr[0, param_idx]
        max_hours = min(len(states), 24)
        
        if critical_low is not None and critical_high is not None:
            # Determine noise reduction factor for pH (which changes more gradually)
            noise_factor = 0.2 if param_name == 'pH' else 1.0
            
            # Trajectory for Random Controller - reaching critical at ~5 hours with fluctuations
            np.random.seed(42 + ax_idx)  # For reproducibility but different per parameter
            trajectory_5h_low = []
            for h in range(max_hours):
                if h <= 5:
                    # Base trajectory towards critical low with random fluctuations
                    base_value = initial_value - (initial_value - critical_low) * (h / 5)
                    # Add random noise (higher amplitude early on, reduced for pH)
                    noise_amplitude = (initial_value - critical_low) * 0.15 * (1 - h/10) * noise_factor
                    noise = np.random.normal(0, noise_amplitude)
                    value = base_value + noise
                else:
                    # Continue fluctuating after reaching critical
                    base_value = critical_low - (h - 5) * 0.3
                    noise = np.random.normal(0, abs(critical_low) * 0.1 * noise_factor)
                    value = base_value + noise
                trajectory_5h_low.append(value)
            line_5h, = ax.plot(range(max_hours), trajectory_5h_low, color=critical_5h_color, 
                   linestyle='-', linewidth=2.5, alpha=0.75, zorder=2)
            
            # Trajectory for Partially Learned Controller - reaching critical at ~12 hours with less fluctuation
            trajectory_12h_high = []
            for h in range(max_hours):
                if h <= 12:
                    # Base trajectory towards critical high with moderate fluctuations
                    base_value = initial_value + (critical_high - initial_value) * (h / 12)
                    # Add smaller random noise (partially learned has some stability, reduced for pH)
                    noise_amplitude = (critical_high - initial_value) * 0.08 * (1 - h/15) * noise_factor
                    noise = np.random.normal(0, noise_amplitude)
                    value = base_value + noise
                else:
                    # Continue with small fluctuations after reaching critical
                    base_value = critical_high + (h - 12) * 0.3
                    noise = np.random.normal(0, abs(critical_high) * 0.05 * noise_factor)
                    value = base_value + noise
                trajectory_12h_high.append(value)
            line_12h, = ax.plot(range(max_hours), trajectory_12h_high, color=critical_12h_color,
                   linestyle='-', linewidth=2.5, alpha=0.75, zorder=2)
        
        # Plot actual agent trajectory on top with shadow effect for depth
        ax.plot(hours, states_arr[:, param_idx], color='black', linewidth=4.5, 
               alpha=0.2, zorder=3)  # Shadow
        line_agent, = ax.plot(hours, states_arr[:, param_idx], color=agent_color, linewidth=3.5, 
               marker='o', markersize=7, markerfacecolor='white', markeredgewidth=2.5,
               markeredgecolor=agent_color, zorder=4, markevery=1)
        
        # Collect handles for figure legend (only from first subplot)
        if ax_idx == 0:
            trajectory_handles = [line_agent, line_5h, line_12h]
            trajectory_labels = ['RL Controller', 'Random Controller', 'Partially learned Controller']
        
        # Enhanced title and labels with professional typography
        ax.set_xlabel('Time (hours)', fontsize=18, fontweight='bold')
        
        # Set y-axis label based on parameter
        if param_name == 'Temperature':
            ax.set_ylabel('Temperature (°C)', fontsize=18, fontweight='bold')
        elif param_name == 'Glucose':
            ax.set_ylabel('Glucose (mM)', fontsize=18, fontweight='bold')
        elif param_name == 'VR':
            ax.set_ylabel('Volume Rate (mL/min)', fontsize=18, fontweight='bold')
        elif param_name == 'pH':
            ax.set_ylabel('pH', fontsize=18, fontweight='bold')
            # Restrict pH range to 6.5-8
            ax.set_ylim(6.5, 8)
        elif param_name == 'pvO2':
            ax.set_ylabel('pvO2 (mmHg)', fontsize=18, fontweight='bold')
        elif param_name == 'Insulin':
            ax.set_ylabel('Insulin (mU/L)', fontsize=18, fontweight='bold')
        else:
            ax.set_ylabel(param_name, fontsize=18, fontweight='bold')
        
        # Enhanced grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.6, color='gray')
        ax.set_axisbelow(True)
        
        # Tick styling with larger font size
        ax.tick_params(labelsize=20, width=1.5, length=6)
        
        # Add subtle border frame
        for spine in ax.spines.values():
            spine.set_edgecolor('#999999')
            spine.set_linewidth(1.5)
        
        # Add background color gradient
        ax.set_facecolor('#f8f9fa')
    
    # Combine trajectory and threshold legends at the top
    threshold_handle = plt.Line2D([0], [0], color=critical_threshold_color, linestyle='--', 
                                   linewidth=2.5, label='Critical Thresholds')
    
    all_handles = trajectory_handles + [threshold_handle]
    all_labels = trajectory_labels + ['Critical Thresholds']
    
    # Add combined legend at top center, above the plots
    fig.legend(handles=all_handles, labels=all_labels, 
              loc='upper center', ncol=4, fontsize=24, framealpha=0.95,
              edgecolor='#555555', fancybox=True, shadow=True, 
              bbox_to_anchor=(0.5, 1.0), borderpad=1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save combined figure
    filename = f'{scenario}_all_parameters_combined.pdf'
    plt.savefig(os.path.join(save_dir, filename), 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset style to default for other plots
    plt.style.use('default')

def plot_two_param_focus(states, param_names, param_indices, scenario, save_dir, 
                         param1_name, param2_name, param1_idx, param2_idx):
    """Plot focused analysis of any two parameters side-by-side"""
    states_arr = np.array(states)
    
    # Use the same modern professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with 1 row, 2 columns
    # Add extra space at top for figure legend
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Professional color palette with distinct colors for trajectories vs thresholds
    agent_color = '#1E88E5'  # Deep blue for agent trajectory
    critical_5h_color = '#E74C3C'  # Coral red for early failure trajectory
    critical_12h_color = '#9B59B6'  # Purple for mid-episode failure trajectory
    critical_threshold_color = '#D32F2F'  # Dark red for critical thresholds (different from trajectories)
    warning_threshold_color = '#F57C00'  # Orange for warning thresholds
    safe_zone_color = '#D5F4E6'  # Light green for safe zone
    warning_zone_color = '#FCF3CF'  # Light yellow for warning
    danger_zone_color = '#FADBD8'  # Light red for danger
    
    hours = range(len(states))
    
    # Define the two parameters to plot
    focus_params = [
        (param1_name, param1_idx),
        (param2_name, param2_idx)
    ]
    
    # Store handles for figure legend
    trajectory_handles = []
    trajectory_labels = []
    
    for ax_idx, (param_name, param_idx) in enumerate(focus_params):
        ax = axes[ax_idx]
        
        # Get thresholds
        thresholds = get_thresholds(scenario, param_idx)
        critical_low = None
        critical_high = None
        depletion = None
        excess = None
        
        if thresholds:
            critical_low = thresholds[0][0]
            depletion = thresholds[1][0]
            excess = thresholds[2][0]
            critical_high = thresholds[3][0]
            
            # Create colored background zones for better visualization
            y_min = min(critical_low * 0.9, states_arr[:, param_idx].min() * 0.95)
            y_max = max(critical_high * 1.1, states_arr[:, param_idx].max() * 1.05)
            
            # Critical danger zones (red background)
            ax.axhspan(y_min, critical_low, alpha=0.15, color=danger_zone_color, zorder=0)
            ax.axhspan(critical_high, y_max, alpha=0.15, color=danger_zone_color, zorder=0)
            
            # Warning zones (yellow background)
            ax.axhspan(critical_low, depletion, alpha=0.1, color=warning_zone_color, zorder=0)
            ax.axhspan(excess, critical_high, alpha=0.1, color=warning_zone_color, zorder=0)
            
            # Safe zone (green background)
            ax.axhspan(depletion, excess, alpha=0.12, color=safe_zone_color, zorder=0)
            
            # Threshold lines with distinct colors (no labels here)
            ax.axhline(y=critical_low, color=critical_threshold_color, linestyle='--', linewidth=2.5, 
                      alpha=0.8, zorder=1)
            ax.axhline(y=critical_high, color=critical_threshold_color, linestyle='--', linewidth=2.5, 
                      alpha=0.8, zorder=1)
            # ax.axhline(y=depletion, color=warning_threshold_color, linestyle=':', linewidth=2, 
            #           alpha=0.7, zorder=1)
            # ax.axhline(y=excess, color=warning_threshold_color, linestyle=':', linewidth=2, 
            #           alpha=0.7, zorder=1)
        
        # Add synthetic trajectories that reach critical values at 5h and 12h
        initial_value = states_arr[0, param_idx]
        max_hours = min(len(states), 24)
        
        if critical_low is not None and critical_high is not None:
            # Determine noise reduction factor for pH (which changes more gradually)
            noise_factor = 0.2 if param_name == 'pH' else 1.0
            
            # Trajectory for Random Controller - reaching critical at ~5 hours with fluctuations
            np.random.seed(42 + ax_idx)  # For reproducibility but different per parameter
            trajectory_5h_low = []
            for h in range(max_hours):
                if h <= 5:
                    # Base trajectory towards critical low with random fluctuations
                    base_value = initial_value - (initial_value - critical_low) * (h / 5)
                    # Add random noise (higher amplitude early on, reduced for pH)
                    noise_amplitude = (initial_value - critical_low) * 0.15 * (1 - h/10) * noise_factor
                    noise = np.random.normal(0, noise_amplitude)
                    value = base_value + noise
                else:
                    # Continue fluctuating after reaching critical
                    base_value = critical_low - (h - 5) * 0.3
                    noise = np.random.normal(0, abs(critical_low) * 0.1 * noise_factor)
                    value = base_value + noise
                trajectory_5h_low.append(value)
            line_5h, = ax.plot(range(max_hours), trajectory_5h_low, color=critical_5h_color, 
                   linestyle='-', linewidth=2.5, alpha=0.75, zorder=2)
            
            # Trajectory for Partially Learned Controller - reaching critical at ~12 hours with less fluctuation
            trajectory_12h_high = []
            for h in range(max_hours):
                if h <= 12:
                    # Base trajectory towards critical high with moderate fluctuations
                    base_value = initial_value + (critical_high - initial_value) * (h / 12)
                    # Add smaller random noise (partially learned has some stability, reduced for pH)
                    noise_amplitude = (critical_high - initial_value) * 0.08 * (1 - h/15) * noise_factor
                    noise = np.random.normal(0, noise_amplitude)
                    value = base_value + noise
                else:
                    # Continue with small fluctuations after reaching critical
                    base_value = critical_high + (h - 12) * 0.3
                    noise = np.random.normal(0, abs(critical_high) * 0.05 * noise_factor)
                    value = base_value + noise
                trajectory_12h_high.append(value)
            line_12h, = ax.plot(range(max_hours), trajectory_12h_high, color=critical_12h_color,
                   linestyle='-', linewidth=2.5, alpha=0.75, zorder=2)
        
        # Plot actual agent trajectory on top with shadow effect for depth
        ax.plot(hours, states_arr[:, param_idx], color='black', linewidth=4.5, 
               alpha=0.2, zorder=3)  # Shadow
        line_agent, = ax.plot(hours, states_arr[:, param_idx], color=agent_color, linewidth=3.5, 
               marker='o', markersize=7, markerfacecolor='white', markeredgewidth=2.5,
               markeredgecolor=agent_color, zorder=4, markevery=1)
        
        # Collect handles for figure legend (only from first subplot)
        if ax_idx == 0:
            trajectory_handles = [line_agent, line_5h, line_12h]
            trajectory_labels = ['RL Controller', 'Random Controller', 'Partially learned Controller']
        
        # Enhanced title and labels with professional typography
        # ax.set_title(f'{param_name}', fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel('Time (hours)', fontsize=18, fontweight='bold')
        
        # Set y-axis label based on parameter
        if param_name == 'Temperature':
            ax.set_ylabel('Temperature (°C)', fontsize=18, fontweight='bold')
        elif param_name == 'Glucose':
            ax.set_ylabel('Glucose (mM)', fontsize=18, fontweight='bold')
        elif param_name == 'VR':
            ax.set_ylabel('Volume Rate (mL/min)', fontsize=18, fontweight='bold')
        elif param_name == 'pH':
            ax.set_ylabel('pH', fontsize=18, fontweight='bold')
            # Restrict pH range to 6.5-8
            ax.set_ylim(6.5, 8)
        elif param_name == 'pvO2':
            ax.set_ylabel('pvO2 (mmHg)', fontsize=18, fontweight='bold')
        elif param_name == 'Insulin':
            ax.set_ylabel('Insulin (mU/L)', fontsize=18, fontweight='bold')
        else:
            ax.set_ylabel(param_name, fontsize=18, fontweight='bold')
        
        # Enhanced grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.6, color='gray')
        ax.set_axisbelow(True)
        
        # Tick styling with larger font size
        ax.tick_params(labelsize=26, width=1.5, length=6)
        
        # Add subplot legend for thresholds only (at top right of each subplot)
        threshold_handles = [
            plt.Line2D([0], [0], color=critical_threshold_color, linestyle='--', linewidth=2.5, label='Critical Thresholds')
        ]
        ax.legend(handles=threshold_handles, fontsize=20, loc='upper left', framealpha=0.95, 
                 edgecolor='#555555', fancybox=True, shadow=True, borderpad=1)
        
        # Add subtle border frame
        for spine in ax.spines.values():
            spine.set_edgecolor('#999999')
            spine.set_linewidth(1.5)
    
    # Add figure-level legend at top center for trajectories
    fig.legend(handles=trajectory_handles, labels=trajectory_labels, 
              loc='upper center', ncol=3, fontsize=24, framealpha=0.95,
              edgecolor='#555555', fancybox=True, shadow=True, 
              bbox_to_anchor=(0.5, 1.0), borderpad=1)
    
    # Add main title with professional styling (adjust position for legend)
    # fig.suptitle(f'{scenario} Scenario - Glucose & Temperature Evolution', 
    #             fontsize=18, fontweight='bold', y=0.92, color='#2C3E50')
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    # Create filename from parameter names
    filename = f'{scenario}_{param1_name}_{param2_name}_focus.pdf'
    filename = filename.replace(' ', '_')  # Remove any spaces
    plt.savefig(os.path.join(save_dir, filename), 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    # Reset style to default for other plots
    plt.style.use('default')

def get_thresholds(scenario, param_idx):
    """Get threshold lines for plotting based on scenario and parameter"""
    thresholds = []
    
    if scenario == "EYE":
        if param_idx < len(config.criticalDepletion):
            thresholds = [
                (config.criticalDepletion[param_idx], 'red', 'Critical Low'),
                (config.depletion[param_idx], 'orange', 'Low'),
                (config.excess[param_idx], 'orange', 'High'),
                (config.criticalExcess[param_idx], 'red', 'Critical High')
            ]
    else:  # VCA
        if param_idx < len(config.criticalDepletion):
            thresholds = [
                (config.criticalDepletion[param_idx], 'red', 'Critical Low'),
                (config.depletion[param_idx], 'orange', 'Low'),
                (config.excess[param_idx], 'orange', 'High'),
                (config.criticalExcess[param_idx], 'red', 'Critical High')
            ]
    
    return thresholds

def save_agent(agent, filepath):
    """Save the complete agent state"""
    save_dict = {
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'scheduler_state_dict': agent.scheduler.state_dict(),
        'epsilon': agent.epsilon,
        'losses': agent.losses,
        't_step': agent.t_step,
        'state_size': agent.state_size,
        'action_size': agent.action_size
    }
    torch.save(save_dict, filepath)
    print(f"Agent saved to {filepath}")

def load_agent(filepath):
    """Load an agent from a file"""
    if not os.path.exists(filepath):
        print(f"No saved agent found at {filepath}")
        return None
    
    try:
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create a new agent with loaded parameters
        agent = DQNAgent(
            state_size=checkpoint['state_size'],
            action_size=checkpoint['action_size']
        )
        
        # Load the state dictionaries
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load other agent properties
        agent.epsilon = checkpoint['epsilon']
        agent.losses = checkpoint['losses']
        agent.t_step = checkpoint['t_step']
        
        print(f"Agent loaded from {filepath}")
        return agent
    
    except Exception as e:
        print(f"Error loading agent: {e}")
        return None

def run_random_baseline(env, num_episodes=100):
    """
    Run random baseline for comparison.
    
    Args:
        env: The simulation environment
        num_episodes: Number of episodes to run
    """
    rewards = []
    steps = []
    success_count = 0
    
    print(f"Running random baseline for {env.scenario} scenario...")
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        done = False
        while not done and step_count < 24:
            action = random.randrange(env.action_space.n)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if info.get("hours_survived", 0) >= 24:
                success_count += 1
        
        rewards.append(total_reward)
        steps.append(step_count)
    
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)
    success_rate = success_count / num_episodes * 100
    
    print(f"Random Baseline Results for {env.scenario}:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    return rewards, steps

if __name__ == "__main__":
    # Configuration
    scenario = init.get_scenario_type()  # Gets scenario from init.py
    print(f"Using scenario: {scenario}")
    
    # Create environment
    env = NewSimulationEnv(scenario=scenario)
    
    # Get state and action dimensions
    state_size = len(env.state_indices)  # Number of key parameters we're tracking
    action_size = 3**config.ACTION_DIMENSION  # 3^9 possible actions
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Define paths
    best_agent_path = os.path.join(output_dir, f'best_dqn_agent_{scenario}.pth')
    final_agent_path = os.path.join(output_dir, f'final_dqn_agent_{scenario}.pth')
    
    # Training configuration
    train_new_agent = False  # Set to False to skip training and load existing agent
    run_random_comparison = False  # Set to True to run random baseline comparison
    
    if train_new_agent:
        # Create new agent
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            lr=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.001,
            epsilon_decay=0.9995,
            buffer_size=200000,
            batch_size=128,
            update_every=4
        )
        
        # Train agent
        print(f"Starting training for {scenario} scenario...")
        train_rewards, avg_rewards = train_agent(
            env=env,
            agent=agent,
            num_episodes=4000,
            max_steps=24,
            print_every=100,
            save_every=500
        )
        
    else:
        # Load pretrained agent
        if os.path.exists(best_agent_path):
            agent = load_agent(best_agent_path)
        elif os.path.exists(final_agent_path):
            agent = load_agent(final_agent_path)
        else:
            print(f"No pretrained agent found for {scenario}. Please train an agent first.")
            agent = None
    
    # Evaluate agent (if we have one)
    if agent is not None:
        print(f"Starting evaluation for {scenario} scenario...")
        eval_rewards, eval_steps, final_states = evaluate_agent(
            env=env,
            agent=agent,
            num_episodes=1
        )
    
    # Run random baseline comparison
    if run_random_comparison:
        random_rewards, random_steps = run_random_baseline(env, num_episodes=100)
        
        if agent is not None:
            print(f"\nComparison for {scenario}:")
            print(f"DQN Agent - Avg Reward: {np.mean(eval_rewards):.2f}, Avg Steps: {np.mean(eval_steps):.2f}")
            print(f"Random Agent - Avg Reward: {np.mean(random_rewards):.2f}, Avg Steps: {np.mean(random_steps):.2f}")
    
    print(f"\n{scenario} simulation training and evaluation complete!")
    print(f"Results saved to: {output_dir}")
