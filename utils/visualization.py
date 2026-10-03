"""
Visualization module cho Inventory Management RL
==================================================
Các biểu đồ phân tích kết quả training và evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12
import seaborn as sns
from typing import List, Dict, Optional


def set_style():
    """Thiết lập style cho biểu đồ."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


def plot_demand_analysis(demand_df: pd.DataFrame, save_path: Optional[str] = None):
    """Phân tích và visualize demand data."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("📦 Demand Data Analysis", fontsize=16, fontweight='bold')
    
    # 1. Demand over time
    ax = axes[0, 0]
    ax.plot(demand_df["date"], demand_df["demand"], alpha=0.5, linewidth=0.8, color='steelblue')
    # Moving average
    ma = demand_df["demand"].rolling(window=30).mean()
    ax.plot(demand_df["date"], ma, color='red', linewidth=2, label='30-day MA')
    ax.set_title("Daily Demand Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (units)")
    ax.legend()
    
    # 2. Demand distribution
    ax = axes[0, 1]
    ax.hist(demand_df["demand"], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(demand_df["demand"].mean(), color='red', linestyle='--', label=f'Mean: {demand_df["demand"].mean():.1f}')
    ax.set_title("Demand Distribution")
    ax.set_xlabel("Demand (units)")
    ax.set_ylabel("Frequency")
    ax.legend()
    
    # 3. Demand by day of week
    ax = axes[1, 0]
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily_avg = demand_df.groupby("day_of_week")["demand"].mean()
    bars = ax.bar(range(7), daily_avg.values, color=sns.color_palette("husl", 7), edgecolor='black')
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names)
    ax.set_title("Average Demand by Day of Week")
    ax.set_ylabel("Avg Demand")
    
    # 4. Monthly demand
    ax = axes[1, 1]
    monthly_avg = demand_df.groupby("month")["demand"].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    bars = ax.bar(range(1, 13), monthly_avg.values, color=sns.color_palette("coolwarm", 12), edgecolor='black')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months, rotation=45)
    ax.set_title("Average Demand by Month")
    ax.set_ylabel("Avg Demand")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_curves(
    agents_data: Dict[str, dict],
    save_path: Optional[str] = None,
):
    """
    Vẽ training curves cho nhiều agents.
    
    agents_data: {"Q-Learning": {"rewards": [...], "costs": [...]}, "DQN": {...}}
    """
    set_style()
    n_agents = len(agents_data)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("📈 Training Progress", fontsize=16, fontweight='bold')
    
    colors = sns.color_palette("husl", n_agents)
    
    for idx, (name, data) in enumerate(agents_data.items()):
        color = colors[idx]
        
        # 1. Episode Rewards
        ax = axes[0]
        rewards = data["rewards"]
        ax.plot(rewards, alpha=0.2, color=color)
        # Smoothed
        window = min(50, len(rewards) // 5)
        if window > 1:
            smoothed = pd.Series(rewards).rolling(window=window).mean()
            ax.plot(smoothed, color=color, linewidth=2, label=f"{name}")
        ax.set_title("Episode Total Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        
        # 2. Total Cost
        ax = axes[1]
        costs = data["costs"]
        ax.plot(costs, alpha=0.2, color=color)
        if window > 1:
            smoothed = pd.Series(costs).rolling(window=window).mean()
            ax.plot(smoothed, color=color, linewidth=2, label=f"{name}")
        ax.set_title("Episode Total Cost")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Cost")
        ax.legend()
        
        # 3. Epsilon decay
        ax = axes[2]
        if "epsilon_history" in data:
            ax.plot(data["epsilon_history"], color=color, linewidth=2, label=f"{name}")
    
    axes[2].set_title("Epsilon Decay (Exploration)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_evaluation_comparison(
    eval_results: Dict[str, dict],
    save_path: Optional[str] = None,
):
    """So sánh kết quả evaluation giữa các agents/policies."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("🏆 Agent Comparison (Evaluation)", fontsize=16, fontweight='bold')
    
    names = list(eval_results.keys())
    colors = sns.color_palette("husl", len(names))
    
    # 1. Average Cost
    ax = axes[0]
    costs = [eval_results[n]["mean_cost"] for n in names]
    cost_stds = [eval_results[n].get("std_cost", 0) for n in names]
    bars = ax.bar(names, costs, yerr=cost_stds, color=colors, edgecolor='black', capsize=5)
    ax.set_title("Average Total Cost (Lower is Better)")
    ax.set_ylabel("Total Cost")
    ax.tick_params(axis='x', rotation=30)
    # Highlight best
    best_idx = np.argmin(costs)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # 2. Average Stockout
    ax = axes[1]
    stockouts = [eval_results[n]["mean_stockout"] for n in names]
    bars = ax.bar(names, stockouts, color=colors, edgecolor='black')
    ax.set_title("Total Stockout Units (Lower is Better)")
    ax.set_ylabel("Stockout Units")
    ax.tick_params(axis='x', rotation=30)
    
    # 3. Average Reward
    ax = axes[2]
    rewards = [eval_results[n]["mean_reward"] for n in names]
    reward_stds = [eval_results[n].get("std_reward", 0) for n in names]
    bars = ax.bar(names, rewards, yerr=reward_stds, color=colors, edgecolor='black', capsize=5)
    ax.set_title("Average Total Reward (Higher is Better)")
    ax.set_ylabel("Total Reward")
    ax.tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_inventory_simulation(
    history_df: pd.DataFrame,
    title: str = "Inventory Simulation",
    save_path: Optional[str] = None,
):
    """Vẽ chi tiết 1 episode simulation."""
    set_style()
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f"📊 {title}", fontsize=16, fontweight='bold')
    
    days = history_df["day"]
    
    # 1. Inventory Level
    ax = axes[0]
    ax.fill_between(days, history_df["inventory_after"], alpha=0.3, color='steelblue')
    ax.plot(days, history_df["inventory_after"], color='steelblue', linewidth=1.5, label='Inventory')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Stockout Level')
    ax.set_ylabel("Inventory Level")
    ax.set_title("Inventory Level Over Time")
    ax.legend()
    
    # 2. Demand vs Order
    ax = axes[1]
    ax.bar(days, history_df["demand"], alpha=0.6, color='coral', label='Demand', width=1)
    ax.bar(days, history_df["order_qty"], alpha=0.6, color='green', label='Order Qty', width=0.5)
    ax.set_ylabel("Units")
    ax.set_title("Demand vs Order Quantity")
    ax.legend()
    
    # 3. Daily Costs breakdown
    ax = axes[2]
    ax.stackplot(
        days,
        history_df["holding_cost"],
        history_df["stockout_cost"],
        history_df["order_cost"],
        labels=["Holding Cost", "Stockout Cost", "Order Cost"],
        alpha=0.7,
        colors=['steelblue', 'coral', 'forestgreen'],
    )
    ax.set_ylabel("Cost")
    ax.set_title("Daily Cost Breakdown")
    ax.legend(loc='upper left')
    
    # 4. Cumulative Cost
    ax = axes[3]
    ax.plot(days, history_df["total_cost"], color='darkred', linewidth=2)
    ax.fill_between(days, history_df["total_cost"], alpha=0.2, color='red')
    ax.set_ylabel("Cumulative Cost")
    ax.set_xlabel("Day")
    ax.set_title("Cumulative Total Cost")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_cost_breakdown_pie(
    history_df: pd.DataFrame,
    title: str = "Cost Breakdown",
    save_path: Optional[str] = None,
):
    """Biểu đồ tròn phân tích chi phí."""
    set_style()
    
    total_holding = history_df["holding_cost"].sum()
    total_stockout = history_df["stockout_cost"].sum()
    total_order = history_df["order_cost"].sum()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sizes = [total_holding, total_stockout, total_order]
    labels = [
        f"Holding Cost\n{total_holding:,.0f} ({total_holding/sum(sizes)*100:.1f}%)",
        f"Stockout Cost\n{total_stockout:,.0f} ({total_stockout/sum(sizes)*100:.1f}%)",
        f"Order Cost\n{total_order:,.0f} ({total_order/sum(sizes)*100:.1f}%)",
    ]
    colors = ['steelblue', 'coral', 'forestgreen']
    explode = (0, 0.05, 0)
    
    ax.pie(sizes, labels=labels, colors=colors, explode=explode,
           autopct='', startangle=90, textprops={'fontsize': 11})
    ax.set_title(f"🥧 {title}\nTotal: {sum(sizes):,.0f}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
