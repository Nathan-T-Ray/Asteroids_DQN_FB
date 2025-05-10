import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def push(self, transition, priority):
        self.buffer.append(transition)
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Small constant to avoid zero

class DQN(nn.Module):
    def __init__(self, input_size, num_actions):
        super(DQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.value_stream = nn.Linear(256, 1)
        self.advantage_stream = nn.Linear(256, num_actions)

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class QLearningAgent:
    def __init__(self, input_size=209, num_actions=5, learning_rate=0.0001, discount_factor=0.99):
        self.input_size = input_size
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_size, num_actions).to(self.device)
        self.target_model = DQN(input_size, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.memory = PrioritizedReplayBuffer(capacity=100000)
        self.temp_buffer = deque(maxlen=3)  # For 3-step learning
        self.n_step = 3
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_steps = 100000
        self.epsilon_decay = (self.epsilon_min / 1.0) ** (1.0 / self.epsilon_decay_steps) #CHECKME!
        self.beta = 0.4
        self.beta_increment = (1.0 - 0.4) / 100000
        self.update_freq = 1000
        self.step_count = 0
        self.total_reward = 0.0
        self.reward_count = 0
        self.batch_size = 128
        self.best_score = -float('inf') # CHECKME necessary if loss is capped?
        self.writer = SummaryWriter() # CHECKME: TensorBoard logging (ai)
        self.model_path = 'dqn_asteroids_best.pth' 

        #checkpoint overhaul ^ 
        #trying tensorboard 

        self.csv_path = 'game_stats.csv'
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
        #NOTE: *not* visible, files were checkpoints

        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
            print(f"Loaded model from {self.model_path}")
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.5)

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, batch):
        for state, action, reward, next_state in batch:
            self.temp_buffer.append((state, action, reward, next_state))
            if len(self.temp_buffer) == self.n_step:
                n_step_state, n_step_action, n_step_reward, n_step_next_state = self.temp_buffer[0]
                n_step_reward = sum([t[2] * (self.discount_factor ** i) for i, t in enumerate(self.temp_buffer)]) # CHECKME: n-step reward
                self.memory.add(n_step_state, n_step_action, n_step_reward, n_step_next_state)
        self.step_count += 1

        batch_size = min(self.batch_size, len(self.memory.buffer))
        if batch_size < 1:
            return

        minibatch, indices, weights = self.memory.sample(batch_size, self.beta)
        states, actions, rewards, next_states = zip(*minibatch)

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        rewards = torch.clamp(rewards, -10.0, 10.0)

        q_values = self.model(states)
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1)
            next_q_values = self.target_model(next_states)
            target_q = rewards + (self.discount_factor ** self.n_step) * next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        q_values_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = (q_values_selected - target_q).abs().detach().cpu().numpy()
        loss = (weights * nn.MSELoss(reduction='none')(q_values_selected, target_q)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)

        if self.step_count % self.update_freq == 0:
            self.update_target_model()

        if self.step_count % 1000 == 0: # CHECKME: Log steps
            avg_reward = self.total_reward / max(self.reward_count, 1)
            self.writer.add_scalar('Loss', loss.item(), self.step_count)
            self.writer.add_scalar('Avg_Reward', avg_reward, self.step_count)
            self.writer.add_scalar('Epsilon', self.epsilon, self.step_count)
            self.writer.add_scalar('Q_Value_Mean', q_values.mean().item(), self.step_count)
            print(f"Step {self.step_count}: Avg Reward={avg_reward:.3f}, Epsilon={self.epsilon:.3f}, Q-Value Mean={q_values.mean().item():.3f}, Loss={loss.item():.3f}")
            self.total_reward = 0.0
            self.reward_count = 0

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.beta = min(1.0, self.beta + self.beta_increment)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad(): 
            q_values = self.model(state)
        return q_values.argmax().item()

    def add_reward(self, reward):
        self.total_reward += reward
        self.reward_count += 1

    def check_and_save_model(self, score):
        if score > self.best_score:
            self.best_score = score
            self.save_model()
            print(f"New high score: {score}, model saved to {self.model_path}")

    def log_game_stats(self, score, lives, rocks_destroyed, shots_fired):
        shot_accuracy = rocks_destroyed / max(shots_fired, 1)
        data = {
            'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Score': [score],
            'Lives': [lives],
            'Rocks_Destroyed': [rocks_destroyed],
            'Shots_Fired': [shots_fired],
            'Shot_Accuracy': [shot_accuracy]
        }
        df = pd.DataFrame(data)
        if os.path.exists(self.csv_path):
            df.to_csv(self.csv_path, mode='a', header=False, index=False) #mode='a' for append
        else:
            df.to_csv(self.csv_path, mode='w', header=True, index=False)
        print(f"Game stats logged to {self.csv_path}")

    def save_model(self, path=None):
        if path is None:
            path = self.model_path
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='dqn_asteroids_best.pth'):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.update_target_model()

    def __del__(self):
        self.writer.close()