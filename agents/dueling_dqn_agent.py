"""
Dueling Deep Q-Network (Dueling DQN) Agent cho bài toán Inventory Management
=============================================================================
Dueling Network phân tách giá trị của State (V) và Lợi thế của Action (A).
Giúp việc học ổn định hơn khi nhiều trạng thái không yêu cầu chọn action phức tạp.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque, namedtuple
from typing import Optional, Tuple

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class DuelingQNetwork(nn.Module):
    """
    Dueling Neural Network.
    Tách luồng xử lý thành 2 nhánh:
    1. Nhánh Value (V): Đánh giá xem State hiện tại tốt hay xấu.
    2. Nhánh Advantage (A): Đánh giá Action này tốt hơn các action khác bao nhiêu.
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
    """
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dims: Tuple[int, ...] = (128, 64)):
        super().__init__()
        
        # Feature extraction (Lớp dùng chung)
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
            
        self.feature_layer = nn.Sequential(*layers)
        
        # Nhánh Tính Value (V)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Nhánh Tính Advantage (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Công thức Dueling DQN: Q = V + (A - mean(A))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DuelingDQNAgent:
    """
    Agent chạy trên thuật toán Dueling DQN
    """
    def __init__(
        self,
        state_dim: int = 4,
        n_actions: int = 11,
        hidden_dims: Tuple[int, ...] = (128, 64),
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 50000,
        target_update_freq: int = 10,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.n_actions = n_actions
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Khởi tạo 2 mạng Q với kiến trúc Dueling
        self.q_network = DuelingQNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.training_rewards = []
        self.training_costs = []
        self.training_losses = []
        self.epsilon_history = []
        self.episode_count = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()
                
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Calculate Current Q
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate Target Q (Dùng standard DQN target update logic - có thể nâng cấp thêm Double DQN)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
            
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
        self.episode_count += 1
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network()

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "training_rewards": self.training_rewards,
            "training_costs": self.training_costs,
            "training_losses": self.training_losses,
        }, filepath)
        print(f"✅ Dueling DQN Model saved to {filepath}")

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.episode_count = checkpoint.get("episode_count", 0)
        self.training_rewards = checkpoint.get("training_rewards", [])
        self.training_costs = checkpoint.get("training_costs", [])
        self.training_losses = checkpoint.get("training_losses", [])
        print(f"✅ Dueling DQN Model loaded from {filepath}")
