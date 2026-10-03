"""
Q-Learning Agent cho bài toán Inventory Management
====================================================
Tabular Q-Learning với state discretization.
"""

import numpy as np
import pickle
import os
from typing import Optional, Tuple
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning Agent cho Inventory Management.
    
    State được discretize thành bins để sử dụng Q-table.
    """
    
    def __init__(
        self,
        n_actions: int,
        n_bins: Tuple[int, ...] = (20, 7, 5, 10),  # bins cho mỗi state dimension
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: defaultdict để tự động khởi tạo
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # Statistics
        self.training_rewards = []
        self.training_costs = []
        self.epsilon_history = []
    
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Chuyển continuous state thành discrete state."""
        discrete = []
        for i, (val, n_bin) in enumerate(zip(state, self.n_bins)):
            # State đã được normalize về [low, high], ta chia thành n_bin bins
            bin_idx = int(np.clip(val * n_bin, 0, n_bin - 1))
            discrete.append(bin_idx)
        return tuple(discrete)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Chọn action theo epsilon-greedy policy.
        
        Args:
            state: Observation từ environment
            training: True nếu đang training (explore), False nếu evaluate (exploit)
        """
        discrete_state = self._discretize_state(state)
        
        if training and np.random.random() < self.epsilon:
            # Explore: chọn ngẫu nhiên
            return np.random.randint(self.n_actions)
        else:
            # Exploit: chọn action có Q-value cao nhất
            return int(np.argmax(self.q_table[discrete_state]))
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Cập nhật Q-table theo Q-Learning update rule:
        Q(s,a) = Q(s,a) + lr * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        s = self._discretize_state(state)
        s_next = self._discretize_state(next_state)
        
        # Q-Learning update
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[s_next])
        
        self.q_table[s][action] += self.lr * (target - self.q_table[s][action])
    
    def decay_epsilon(self):
        """Giảm epsilon sau mỗi episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
    
    def save(self, filepath: str):
        """Lưu Q-table và parameters."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        data = {
            "q_table": dict(self.q_table),
            "n_actions": self.n_actions,
            "n_bins": self.n_bins,
            "epsilon": self.epsilon,
            "training_rewards": self.training_rewards,
            "training_costs": self.training_costs,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load Q-table từ file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data["q_table"])
        self.epsilon = data.get("epsilon", self.epsilon_end)
        self.training_rewards = data.get("training_rewards", [])
        self.training_costs = data.get("training_costs", [])
        print(f"✅ Model loaded from {filepath}")
