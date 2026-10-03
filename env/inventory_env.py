"""
Inventory Management with Reinforcement Learning
==================================================
Custom Gymnasium Environment for the Inventory/Warehouse Replenishment Problem.

Bài toán: Một cửa hàng/kho hàng cần quyết định mỗi ngày nhập bao nhiêu hàng
để tối thiểu hóa tổng chi phí (chi phí tồn kho + chi phí thiếu hàng + chi phí đặt hàng).

State:  (inventory_level, day_of_week, demand_trend)
Action: số lượng hàng đặt mua (discrete: 0, 10, 20, ..., max_order)
Reward: negative total cost (vì ta muốn minimize cost => maximize negative cost)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class InventoryEnv(gym.Env):
    """
    Reinforcement Learning Environment cho bài toán Quản lý Kho hàng.
    
    Mục tiêu: Agent học cách đặt hàng tối ưu để minimize tổng chi phí:
        - Holding cost: chi phí lưu trữ hàng tồn kho
        - Stockout cost: chi phí khi hết hàng (mất doanh thu, uy tín)  
        - Ordering cost: chi phí cố định mỗi lần đặt hàng + chi phí mỗi đơn vị
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        max_inventory: int = 200,        # Sức chứa tối đa kho
        max_order: int = 100,             # Số lượng đặt hàng tối đa mỗi lần
        order_step: int = 10,             # Bước đặt hàng (0, 10, 20, ...)
        holding_cost: float = 1.0,        # Chi phí tồn kho / đơn vị / ngày
        stockout_cost: float = 5.0,       # Chi phí thiếu hàng / đơn vị
        fixed_order_cost: float = 20.0,   # Chi phí cố định mỗi lần đặt hàng
        unit_order_cost: float = 2.0,     # Chi phí / đơn vị đặt hàng
        lead_time: int = 1,               # Thời gian giao hàng (ngày)
        episode_length: int = 365,        # Số ngày trong 1 episode
        demand_data: Optional[np.ndarray] = None,  # Demand data bên ngoài
        demand_type: str = "seasonal",    # Loại demand: "poisson", "seasonal", "real"
        mean_demand: float = 30.0,        # Nhu cầu trung bình / ngày
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.max_inventory = max_inventory
        self.max_order = max_order
        self.order_step = order_step
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.fixed_order_cost = fixed_order_cost
        self.unit_order_cost = unit_order_cost
        self.lead_time = lead_time
        self.episode_length = episode_length
        self.demand_data = demand_data
        self.demand_type = demand_type
        self.mean_demand = mean_demand
        self.render_mode = render_mode
        
        # Số lượng action rời rạc: 0, order_step, 2*order_step, ..., max_order
        self.n_actions = (max_order // order_step) + 1
        
        # Action space: chọn 1 trong n_actions mức đặt hàng
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Observation space: [inventory_level, day_of_week, pending_orders, last_demand]
        # Normalize về [0, 1]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        
        # Internal state
        self.inventory = 0
        self.day = 0
        self.pending_orders = []  # Danh sách đơn hàng đang vận chuyển
        self.last_demand = 0
        self.total_cost = 0
        self.history = []
        
    def _get_demand(self, day: int) -> int:
        """Lấy nhu cầu hàng hóa cho ngày hiện tại."""
        if self.demand_type == "real" and self.demand_data is not None:
            # Sử dụng data thật
            idx = day % len(self.demand_data)
            return int(self.demand_data[idx])
        elif self.demand_type == "seasonal":
            # Nhu cầu theo mùa (có tính chu kỳ tuần và mùa)
            base = self.mean_demand
            # Xu hướng theo mùa (365 ngày)
            seasonal = 10 * np.sin(2 * np.pi * day / 365)
            # Xu hướng theo tuần (cuối tuần demand cao hơn)
            weekly = 5 * np.sin(2 * np.pi * day / 7)
            # Random noise
            noise = np.random.normal(0, 5)
            demand = max(0, int(base + seasonal + weekly + noise))
            return demand
        else:
            # Poisson demand
            return np.random.poisson(self.mean_demand)
    
    def _get_obs(self) -> np.ndarray:
        """Tạo observation vector (normalized)."""
        inv_norm = np.clip(self.inventory / self.max_inventory, -1.0, 1.0)
        day_norm = (self.day % 7) / 6.0
        pending_norm = sum(qty for _, qty in self.pending_orders) / self.max_order if self.pending_orders else 0.0
        demand_norm = np.clip(self.last_demand / (2 * self.mean_demand), 0.0, 1.0)
        
        return np.array([inv_norm, day_norm, pending_norm, demand_norm], dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment về trạng thái ban đầu."""
        super().reset(seed=seed)
        
        self.inventory = self.max_inventory // 2  # Bắt đầu với kho nửa đầy
        self.day = 0
        self.pending_orders = []
        self.last_demand = 0
        self.total_cost = 0
        self.history = []
        
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Thực hiện 1 bước (1 ngày):
        1. Agent quyết định đặt hàng (action)
        2. Hàng đã đặt trước đó được giao (nếu hết lead time)
        3. Demand xảy ra, trừ inventory
        4. Tính chi phí
        """
        # 1. Đặt hàng
        order_qty = action * self.order_step
        order_cost = 0.0
        if order_qty > 0:
            order_cost = self.fixed_order_cost + self.unit_order_cost * order_qty
            self.pending_orders.append((self.day + self.lead_time, order_qty))
        
        # 2. Nhận hàng (hàng đã đến)
        received = 0
        remaining_orders = []
        for delivery_day, qty in self.pending_orders:
            if delivery_day <= self.day:
                received += qty
            else:
                remaining_orders.append((delivery_day, qty))
        self.pending_orders = remaining_orders
        
        self.inventory = min(self.inventory + received, self.max_inventory)
        
        # 3. Demand xảy ra
        demand = self._get_demand(self.day)
        self.last_demand = demand
        
        # Tính stockout
        stockout = max(0, demand - self.inventory)
        self.inventory = max(0, self.inventory - demand)
        
        # 4. Tính chi phí
        holding = self.holding_cost * self.inventory
        stockout_penalty = self.stockout_cost * stockout
        daily_cost = holding + stockout_penalty + order_cost
        self.total_cost += daily_cost
        
        # Reward = negative cost (muốn maximize reward = minimize cost)
        reward = -daily_cost
        
        # Lưu lịch sử
        self.history.append({
            "day": self.day,
            "inventory_before": self.inventory + demand - received,
            "order_qty": order_qty,
            "received": received,
            "demand": demand,
            "stockout": stockout,
            "inventory_after": self.inventory,
            "holding_cost": holding,
            "stockout_cost": stockout_penalty,
            "order_cost": order_cost,
            "daily_cost": daily_cost,
            "total_cost": self.total_cost,
            "reward": reward,
        })
        
        self.day += 1
        terminated = self.day >= self.episode_length
        truncated = False
        
        info = {
            "daily_cost": daily_cost,
            "total_cost": self.total_cost,
            "stockout": stockout,
            "inventory": self.inventory,
            "demand": demand,
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def get_history_df(self) -> pd.DataFrame:
        """Trả về lịch sử dưới dạng DataFrame."""
        return pd.DataFrame(self.history)
    
    def render(self):
        """Hiển thị trạng thái hiện tại."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            status = (
                f"Day {self.day:3d} | "
                f"Inventory: {self.inventory:4d} | "
                f"Last Demand: {self.last_demand:3d} | "
                f"Pending: {sum(q for _, q in self.pending_orders):3d} | "
                f"Total Cost: {self.total_cost:,.0f}"
            )
            if self.render_mode == "human":
                print(status)
            return status


# Đăng ký environment với Gymnasium
gym.register(
    id="InventoryManagement-v0",
    entry_point="env.inventory_env:InventoryEnv",
)
