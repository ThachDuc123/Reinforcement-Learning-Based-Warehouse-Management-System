"""
Baseline Policies để so sánh với RL Agent
==========================================
- Random Policy: đặt hàng ngẫu nhiên
- (s, S) Policy: đặt hàng khi inventory < s, đặt lên mức S
- Fixed Order Policy: đặt số lượng cố định mỗi ngày
"""

import numpy as np


class RandomPolicy:
    """Đặt hàng ngẫu nhiên."""
    
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.name = "Random Policy"
    
    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        return np.random.randint(self.n_actions)


class FixedOrderPolicy:
    """Đặt cùng một lượng hàng mỗi ngày."""
    
    def __init__(self, n_actions: int, fixed_action: int = 3):
        self.n_actions = n_actions
        self.fixed_action = fixed_action  # action index (vd: 3 = 30 units nếu step=10)
        self.name = f"Fixed Order (action={fixed_action})"
    
    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        return self.fixed_action


class SsPolicyAgent:
    """
    (s, S) Policy - chính sách tồn kho cổ điển:
    - Khi inventory <= s (reorder point): đặt hàng lên mức S (order-up-to level)
    - Khi inventory > s: không đặt hàng
    
    Đây là baseline mạnh trong inventory management.
    """
    
    def __init__(
        self,
        n_actions: int,
        order_step: int = 10,
        max_inventory: int = 200,
        reorder_point: float = 0.3,   # s = 30% max_inventory
        order_up_to: float = 0.8,     # S = 80% max_inventory
    ):
        self.n_actions = n_actions
        self.order_step = order_step
        self.max_inventory = max_inventory
        self.s = int(reorder_point * max_inventory)  # Reorder point
        self.S = int(order_up_to * max_inventory)     # Order-up-to level
        self.name = f"(s,S) Policy (s={self.s}, S={self.S})"
    
    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        """
        State[0] = inventory_level (normalized to [-1, 1])
        """
        inv_level = int(state[0] * self.max_inventory)
        
        if inv_level <= self.s:
            # Cần đặt hàng: đặt đủ lên mức S
            order_qty = self.S - inv_level
            action = min(order_qty // self.order_step, self.n_actions - 1)
            return max(1, action)
        else:
            return 0  # Không đặt hàng


class EOQPolicy:
    """
    Economic Order Quantity (EOQ) Policy:
    - Tính EOQ dựa trên demand trung bình
    - Đặt hàng khi inventory gần hết
    """
    
    def __init__(
        self,
        n_actions: int,
        order_step: int = 10,
        max_inventory: int = 200,
        mean_demand: float = 30.0,
        fixed_order_cost: float = 20.0,
        holding_cost: float = 1.0,
    ):
        self.n_actions = n_actions
        self.order_step = order_step
        self.max_inventory = max_inventory
        
        # EOQ formula: sqrt(2 * D * K / h)
        # D = annual demand, K = fixed ordering cost, h = holding cost
        annual_demand = mean_demand * 365
        self.eoq = np.sqrt(2 * annual_demand * fixed_order_cost / holding_cost)
        self.eoq_action = min(int(self.eoq / order_step), n_actions - 1)
        
        # Reorder point = lead time demand + safety stock
        self.reorder_point = int(mean_demand * 2)  # 2 days lead time coverage
        
        self.name = f"EOQ Policy (EOQ={self.eoq:.0f}, ROP={self.reorder_point})"
    
    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        inv_level = int(state[0] * self.max_inventory)
        
        if inv_level <= self.reorder_point:
            return max(1, self.eoq_action)
        else:
            return 0
