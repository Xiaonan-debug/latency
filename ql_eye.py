# Revised 4.16.25 Brassil, Functional Circulation v10 - Modified to use DQN approach
# v9 changed to stochastic PFI and initialized gas flow and richness at 0.9
# v10 changed to allow simulaton after out of bounds condition to enhance learning
# v10.1 changed to call FunctionalStepSimulatorEye.py to simulate eye perfusion and to revise initial coonditions
# v11 Modified to use Deep Q-Network instead of Q-table

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
from gym import Env, spaces
from FunctionalStepSimulatorEye import SingleStep

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory for results
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(output_dir, exist_ok=True)

# Setup initial values - Eye-specific parameters
temperatureCelsius = 36
pressuremmHg = 70
perfusionFlowmLpm = 1 # Set F to 1 mL/min for eye
vascularResistance = pressuremmHg / perfusionFlowmLpm
graftMassG = .326  # Eye mass
perfusionFlowIndex = perfusionFlowmLpm / (pressuremmHg * (graftMassG / 100))
pH = 7.4
pO2mmHg = 300
pvO2 = 37
svO2 = .65
pvCO2mmHg = 30    
glucosemMolar = 8
insulinmUnits = 50
pCO2mmHg = 20
lactatemMolar = 1
hematocrit = 30

# bigState addends
bicarbMmoles = 25
gasFlowLPM = 1.5
gasRichness = .5
hours = 0

# Action and state dimensions
possibleActions = 7  # Number of action dimensions
possibleStates = 6   # Number of state dimensions

# Initialize the bigState with all the system states
initialBigState = [temperatureCelsius, pressuremmHg, perfusionFlowmLpm, perfusionFlowIndex, 
                   pH, pO2mmHg, pvO2, svO2, pvCO2mmHg, glucosemMolar, insulinmUnits, 
                   lactatemMolar, hematocrit, bicarbMmoles, gasFlowLPM, gasRichness, hours]

# Reset values
resetReward = 120

# Define the Transition tuple for experience replay
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

# Eye-specific thresholds (smaller ranges for eye)
criticalDepletion = [34, 20, 0, .1, 6.9, 70, 0, 0, 0, 2, 1, 0, 1]  # Changed from 10 to 34 for temperature
depletion = [35, 40, .1, .5, 7.1, 100, 20, .3, 20, 3, 15, 0, 10]  # Changed from 19 to 35 for temperature
excess = [37, 100, 150, 20, 7.5, 600, 500, .9, 50, 12, 45, 15, 100]  # Changed from 38 to 37 for temperature
criticalExcess = [38, 120, 3, 50, 7.7, 700, 700, 1, 60, 33, 80, 30, 100]  # Changed from 41 to 38 for temperature

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
    """Enhanced Deep Q-Network architecture for eye simulation"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

class PumpScapeEye(Env):
    """Environment for eye perfusion system simulation"""
    def __init__(self):
        super(PumpScapeEye, self).__init__()
        
        # Define continuous state space (using the 6 key parameters for eye)
        self.observation_space = spaces.Box(
            low=np.array([34, 0.1, 6.9, 0, 2, 1]),  # Changed from 10 to 34 for temperature
            high=np.array([38, 50, 7.7, 700, 33, 80]),  # Changed from 41 to 38 for temperature
            dtype=np.float32
        )
        
        # Define discrete action space (5^7 possible actions)
        self.action_space = spaces.Discrete(5**possibleActions)
        
        # Initial state
        self.state = None
        self.bigState = None
        
    def decode_state(self, bigState):
        """Extract the 6 key parameters from bigState for eye"""
        # Map from 17-element bigState to 6-element state
        # [temperature, PFI, pH, pvO2, glucose, insulin]
        return np.array([
            bigState[0],      # temperature
            bigState[3],      # PFI
            bigState[4],      # pH
            bigState[6],      # pvO2
            bigState[9],      # glucose
            bigState[10]      # insulin      
        ], dtype=np.float32)
    
    def decode_action(self, action_idx):
        """Convert action index to action components"""
        action_vector = []
        temp = action_idx

        action_mapping = {0: -1, 1: -0.5, 2: 0, 3: 0.5, 4: 1}
        
        for _ in range(possibleActions):
            component = temp % 5  # Changed from 3 to 5
            action_vector.append(action_mapping[component])
            temp = temp // 5  # Changed from 3 to 5
            
        return action_vector
        
    def step(self, action_idx, train=True):
        # Decode the action index to action components
        action_vector = self.decode_action(action_idx)
        
        # Prepare the action combo for SingleStep
        action_combo = [action_idx, self.bigState]
        
        # Call the simulator to get the next state and reward
        answer = SingleStep(action_combo)
        
        # Unpack the results
        self.bigState = answer[0]  # Updated bigState
        score_vector = answer[1]   # Score vector
        simulator_reward = answer[2]  # Simulator reward
        
        # Extract the 6-element state from bigState
        self.state = self.decode_state(self.bigState)

        hours_survived = self.bigState[16]
        
        # Check if episode is done (24 hours reached or critical values)
        done = False
        if hours_survived >= 24:  # Hours exceed 24
            done = True
        
        # Check if any critical values were reached
        for score in score_vector:
            if abs(score) >= 2:  # Critical values have score -2 or 2
                done = True
                break
                
        # Calculate reward for eye simulation
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
                
        return self.state, reward, done, {}

    def reset(self):
        # Reset to initial conditions with realistic randomness for eye
        self.bigState = initialBigState.copy()
        
        # Add realistic variations for eye simulation
        # Vascular resistance for eye
        vascularResistance = random.uniform(60, 80)
        perfusionFlowmLpm = pressuremmHg / vascularResistance
        perfusionFlowIndex = perfusionFlowmLpm / (pressuremmHg * (graftMassG / 100))
        
        # Temperature variation for eye - constrained to 34-38°C range
        temp_variation = random.uniform(-1, 1)
        self.bigState[0] = max(34, min(38, 36 + temp_variation))  # Ensure temperature stays within 34-38°C
        
        # pH variation (±0.1)
        self.bigState[4] = 7.4 + random.uniform(-0.1, 0.1)
        
        # Glucose variation for eye (±1 mM)
        self.bigState[9] = 8 + random.uniform(-1, 1)
        
        # Insulin variation (±10 mU)
        self.bigState[10] = 50 + random.uniform(-10, 10)
        
        # Hematocrit variation for eye (±2%)
        self.bigState[12] = 30 + random.uniform(-2, 2)
        
        # Update flow parameters
        self.bigState[2] = perfusionFlowmLpm
        self.bigState[3] = perfusionFlowIndex
        self.bigState[16] = 0
        
        self.state = self.decode_state(self.bigState)
        return self.state

class DQNAgent:
    """Deep Q-Network agent for eye simulation"""
    def __init__(self, state_size, action_size, 
                 lr=1e-4, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64,
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
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
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
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(device)
        
        # Compute current Q values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next state values
        next_state_values = torch.zeros(self.batch_size, 1).to(device)
        with torch.no_grad():
            next_state_values[non_terminal_mask] = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        
        # Compute expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.losses.append(loss.item())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
    
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
                
        # Update target network periodically
        if self.t_step % (self.update_every * 10) == 0:
            self.update_target_network()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

def train_agent(env, agent, num_episodes=10000, max_steps=24, 
                print_every=100, save_every=1000, save_dir=output_dir):
    """Train the DQN agent for eye simulation"""
    rewards = []
    avg_rewards = []
    best_avg_reward = -float('inf')
    episode_durations = []
    
    for episode in range(1, num_episodes+1):
        state = env.reset()
        total_reward = 0
        step_count = 0

        for t in range(max_steps):
            # Choose and take action
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Save experience and learn
            agent.step(state, action, reward, next_state, done)
            
            # Update state and rewards
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                break
                
        rewards.append(total_reward)
        episode_durations.append(step_count)
        
        # Calculate average reward over last 100 episodes
        avg_reward = np.mean(rewards[-100:])
        avg_rewards.append(avg_reward)

        # Save model periodically
        if episode % save_every == 0:
            save_path = os.path.join(save_dir, f'dqn_eye_agent_episode_{episode}.pth')
            save_agent(agent, save_path)
            
        # Save best model if improved
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_model_path = os.path.join(save_dir, 'best_dqn_eye_agent.pth')
            save_agent(agent, best_model_path)
        
        # Print progress
        if episode % print_every == 0:
            print(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'final_dqn_eye_agent.pth')
    save_agent(agent, final_model_path)
    
    # Plot training progress
    plt.figure(figsize=(12, 8))
    plt.plot(rewards, alpha=0.3)
    plt.plot(avg_rewards, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Eye Simulation Training Progress')
    plt.savefig(os.path.join(output_dir, 'eye_training_progress.png'))
    plt.show()
    
    # Plot loss history
    plt.figure(figsize=(12, 8))
    plt.plot(agent.losses)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Eye Simulation Training Loss')
    plt.savefig(os.path.join(output_dir, 'eye_training_loss.png'))
    plt.show()

    return rewards, avg_rewards

def evaluate_agent(env, agent=None, agent_path=None, num_episodes=100):
    """Evaluate the trained agent for eye simulation"""
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
    action_counts = np.zeros(5**possibleActions)
    final_bigStates = []
    max_steps = 24  # Max steps for evaluation

    all_state_values = [[] for _ in range(max_steps)]
    all_actions_idx = [[] for _ in range(max_steps)]
    all_actions_components = [[] for _ in range(max_steps)]

    # For tracking a single episode in detail
    single_episode_states = []
    single_episode_actions = []
    episode_to_track = 0  # Track the first episode by default
    
    # Set to evaluation mode
    agent.policy_net.eval()
    agent.epsilon = 0.0  # No exploration during evaluation
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0

        # Lists to store this episode's data
        episode_states = [env.bigState.copy()]  # Initial state
        episode_actions = []

        done = False
        while not done and step_count < 24:  # Max 24 hours
            action = agent.choose_action(state)
            action_idx = action
            action_components = env.decode_action(action_idx)

            # Enforce non-negative actions for infusion-only components (indices 3-6)
            constrained_components = action_components.copy()
            for i in range(3, len(constrained_components)):
                if constrained_components[i] < 0:
                    constrained_components[i] = 0

            episode_actions.append(constrained_components)
            all_actions_idx[step_count].append(action_idx)
            all_actions_components[step_count].append(constrained_components)

            action_counts[action] += 1
            next_state, reward, done, _ = env.step(action, train=False)

            episode_states.append(env.bigState.copy())
            all_state_values[step_count].append(env.bigState.copy())

            state = next_state
            total_reward += reward
            step_count += 1

        # Save a specific episode for detailed visualization
        if episode == episode_to_track or total_reward > max(rewards, default=0):
            single_episode_states = episode_states
            single_episode_actions = episode_actions
            episode_to_track = episode
            
        rewards.append(total_reward)
        steps.append(step_count)
        final_bigStates.append(env.bigState.copy())
    
    # Calculate mean and std state values
    mean_state_values = []
    std_state_values = []
    
    for hour in range(max_steps):
        if all_state_values[hour]:
            hour_data = np.array(all_state_values[hour])
            mean_state_values.append(np.mean(hour_data, axis=0))
            std_state_values.append(np.std(hour_data, axis=0))
        else:
            mean_state_values.append(np.full(len(initialBigState), np.nan))
            std_state_values.append(np.full(len(initialBigState), np.nan))
    
    mean_state_values = np.array(mean_state_values)
    std_state_values = np.array(std_state_values)

    # Calculate action component means
    action_component_means = []
    
    for hour in range(max_steps):
        if all_actions_components[hour]:
            hour_actions = np.array(all_actions_components[hour])
            action_component_means.append(np.mean(hour_actions, axis=0))
        else:
            action_component_means.append(np.full(possibleActions, np.nan))
    
    action_component_means = np.array(action_component_means)
        
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)
    
    print(f"Eye Simulation Evaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    
    # Eye-specific parameter names and indices
    param_names = ["Temperature (C)", "PFI", "pH", "pvO2", 
                   "Glucose (mM)", "Insulin (mU)", "Lactate (mM)", 
                   "Hematocrit", "Bicarb (mmoles)"]
    param_indices = [0, 3, 4, 6, 9, 10, 11, 12, 13]
    
    action_names = ["Temperature", "Gas Flow", "Gas Richness", 
                   "Glucose", "Insulin", "Bicarb", "Vasodilator"]

    # Plot visualizations
    plot_eye_state_evolution(mean_state_values, std_state_values, param_names, max_steps, param_indices)
    plot_eye_action_evolution(action_component_means, action_names, max_steps)
    plot_single_episode_eye_state_evolution(single_episode_states, param_names, param_indices)
    plot_single_episode_eye_action_evolution(single_episode_actions, action_names)
    
    # Final state analysis
    final_bigStates_arr = np.array(final_bigStates)
    selected_mean_values = final_bigStates_arr[:, param_indices].mean(axis=0)
    selected_std_values = final_bigStates_arr[:, param_indices].std(axis=0)
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(selected_mean_values)), selected_mean_values, yerr=selected_std_values, capsize=5)
    plt.xticks(range(len(selected_mean_values)), param_names, rotation=45, ha="right")
    plt.ylabel("Average Final Value")
    plt.title("Average Final Eye System State Parameters After Evaluation (DQN-Eye)")
    
    for bar, value in zip(bars, selected_mean_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_eye_state_parameters.png'))
    plt.show()
    
    return rewards, steps, action_counts

def plot_eye_state_evolution(mean_values, std_values, param_names, max_steps, selected_indices):
    """Plot the evolution of selected state parameters over time for eye simulation"""
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    hours = range(1, max_steps + 1)
    
    for i, (param_name, param_idx) in enumerate(zip(param_names, selected_indices)):
        if i < len(axs):
            ax = axs[i]
            
            ax.plot(hours, mean_values[:, param_idx], 'b-', label='Mean')
            ax.fill_between(
                hours, 
                mean_values[:, param_idx] - std_values[:, param_idx],
                mean_values[:, param_idx] + std_values[:, param_idx],
                alpha=0.2, color='b', label='Std Dev'
            )
            
            ax.set_title(f'Eye {param_name}')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.7)
    
    for i in range(len(param_names), len(axs)):
        axs[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eye_state_evolution_over_time.png'))
    plt.show()

def plot_eye_action_evolution(action_means, action_names, max_steps):
    """Plot the evolution of actions taken over time for eye simulation"""
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    axs = axs.flatten()
    
    hours = range(1, max_steps + 1)
    colors = {-1: 'red', 0: 'gray', 1: 'green'}
    
    for i, action_name in enumerate(action_names):
        if i < len(axs):
            ax = axs[i]
            
            action_data = action_means[:, i]
            ax.plot(hours, action_data, 'b-o', label='Mean Action Value')
            
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            if i >= 3:  # Infusion-only actions
                ax.set_ylim(-0.1, 1.1)
                ax.set_title(f'Eye Action: {action_name} (≥0 only)')
                ax.set_ylabel('Mean Action Value (0 to 1)')
            else:  # Bidirectional actions
                ax.set_ylim(-1.1, 1.1)
                ax.set_title(f'Eye Action: {action_name}')
                ax.set_ylabel('Mean Action Value (-1 to 1)')
            
            ax.set_xlabel('Hour')
            ax.grid(True, linestyle='--', alpha=0.7)
    
    for i in range(len(action_names), len(axs)):
        axs[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eye_action_evolution_over_time.png'))
    plt.show()

def plot_single_episode_eye_state_evolution(states, param_names, selected_indices):
    """Plot the evolution of selected state parameters for a single eye simulation episode"""
    states_array = np.array(states)
    
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    hours = range(len(states))
    
    for i, (param_name, param_idx) in enumerate(zip(param_names, selected_indices)):
        if i < len(axs):
            ax = axs[i]
            
            ax.plot(hours, states_array[:, param_idx], 'b-o', label='Value')
            
            # Add threshold lines for eye-specific values
            if param_idx < 13:
                ax.axhline(y=criticalDepletion[param_idx], color='r', linestyle='--', alpha=0.5, label='Critical Low')
                ax.axhline(y=depletion[param_idx], color='orange', linestyle='--', alpha=0.5, label='Low')
                ax.axhline(y=excess[param_idx], color='orange', linestyle='--', alpha=0.5, label='High')
                ax.axhline(y=criticalExcess[param_idx], color='r', linestyle='--', alpha=0.5, label='Critical High')
            
            ax.set_title(f'Eye {param_name}')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            if i == 0:
                ax.legend()
    
    for i in range(len(param_names), len(axs)):
        axs[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'single_episode_eye_state_evolution.png'))
    plt.show()

def plot_single_episode_eye_action_evolution(actions, action_names):
    """Plot the evolution of actions for a single eye simulation episode"""
    actions_array = np.array(actions)
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    axs = axs.flatten()
    
    hours = range(1, len(actions) + 1)
    colors = {-1: 'red', 0: 'gray', 1: 'green'}
    action_labels = {-1: 'Decrease (-1)', 0: 'No Change (0)', 1: 'Increase (+1)'}
    
    for i, action_name in enumerate(action_names):
        if i < len(axs):
            ax = axs[i]
            
            ax.plot(hours, actions_array[:, i], 'b-o', label='Action Value')
            
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            # Color background based on action value
            for h in range(len(hours)):
                action_val = actions_array[h, i]
                color = colors.get(action_val, 'gray')
                ax.axvspan(hours[h]-0.5, hours[h]+0.5, alpha=0.2, color=color)
            
            if i >= 3:  # Infusion-only actions
                ax.set_ylim(-0.1, 1.1)
                ax.set_title(f'Eye Action: {action_name} (≥0 only)')
                ax.set_ylabel('Action Value (0 to 1)')
            else:  # Bidirectional actions
                ax.set_ylim(-1.1, 1.1)
                ax.set_title(f'Eye Action: {action_name}')
                ax.set_ylabel('Action Value (-1 to 1)')
            
            ax.set_xlabel('Hour')
            ax.grid(True, linestyle='--', alpha=0.7)
    
    for i in range(len(action_names), len(axs)):
        axs[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'single_episode_eye_action_evolution.png'))
    plt.show()

def save_agent(agent, filepath):
    """Save the complete agent state"""
    save_dict = {
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'losses': agent.losses,
        't_step': agent.t_step
    }
    torch.save(save_dict, filepath)
    print(f"Eye agent saved to {filepath}")

def load_agent(filepath, agent=None, state_size=6, action_size=5**7):
    """Load an agent from a file"""
    if not os.path.exists(filepath):
        print(f"No saved agent found at {filepath}")
        return None
        
    checkpoint = torch.load(filepath)
    
    if agent is None:
        # Create a new agent if none provided
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size
        )
    
    # Load the state dictionaries
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load other agent properties
    agent.epsilon = checkpoint['epsilon']
    agent.losses = checkpoint['losses']
    agent.t_step = checkpoint['t_step']
    
    print(f"Eye agent loaded from {filepath}")
    return agent

if __name__ == "__main__":
    # Create environment
    env = PumpScapeEye()
    
    # Get state and action dimensions
    state_size = 6  # The 6 key parameters we're tracking for eye
    action_size = 5**possibleActions  # 5^7 possible actions
    
    # Define paths
    agent_save_path = os.path.join(output_dir, 'final_dqn_eye_agent.pth')
    best_agent_path = os.path.join(output_dir, 'best_dqn_eye_agent.pth')
    
    # Check if we want to train or load a pretrained agent
    train_new_agent = False # Set to False to skip training
    
    if train_new_agent:
        # Create new agent for eye simulation
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            lr=5e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.0001,
            epsilon_decay=0.995,
            buffer_size=100000,
            batch_size=64,
            update_every=4
        )
        
        # Train agent
        print("Starting eye simulation training...")
        train_rewards, avg_rewards = train_agent(
            env=env,
            agent=agent,
            num_episodes=1,
            max_steps=24,
            print_every=100,
            save_every=1000
        )
    else:
        # Load pretrained agent
        if os.path.exists(best_agent_path):
            agent = load_agent(best_agent_path)
        elif os.path.exists(agent_save_path):
            agent = load_agent(agent_save_path)
        else:
            print("No pretrained agent found. Please train an agent first.")
            agent = None
    
    # Evaluate agent (if we have one)
    if agent is not None:
        print("Starting eye simulation evaluation...")
        eval_rewards, eval_steps, action_counts = evaluate_agent(
            env=env,
            agent=agent,
            num_episodes=1
        )
        
        print("Eye simulation training and evaluation complete!")








