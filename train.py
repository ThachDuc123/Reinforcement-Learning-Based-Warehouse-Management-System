"""
Training Script cho Inventory Management RL
=============================================
Huấn luyện Q-Learning và DQN agents.
"""

import sys
import os
import numpy as np
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.inventory_env import InventoryEnv
from agents.q_learning import QLearningAgent
from agents.dqn_agent import DQNAgent
from data.data_generator import generate_seasonal_demand


def train_agent(
    agent,
    env: InventoryEnv,
    n_episodes: int = 500,
    verbose: bool = True,
) -> dict:
    """
    Huấn luyện agent trên environment.
    
    Args:
        agent: QLearningAgent hoặc DQNAgent
        env: InventoryEnv
        n_episodes: Số episodes huấn luyện
        verbose: In tiến trình
        
    Returns:
        Dictionary chứa training statistics
    """
    is_dqn = hasattr(agent, "store_experience")
    
    all_rewards = []
    all_costs = []
    best_reward = -float("inf")
    
    pbar = tqdm(range(n_episodes), desc="Training", disable=not verbose)
    
    for episode in pbar:
        state, _ = env.reset(seed=episode)
        episode_reward = 0
        episode_losses = []
        
        done = False
        while not done:
            # Chọn action
            action = agent.select_action(state, training=True)
            
            # Thực hiện action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Cập nhật agent
            if is_dqn:
                agent.store_experience(state, action, reward, next_state, done)
                loss = agent.update()
                if loss is not None:
                    episode_losses.append(loss)
            else:
                agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Record stats
        total_cost = info["total_cost"]
        all_rewards.append(episode_reward)
        all_costs.append(total_cost)
        agent.training_rewards.append(episode_reward)
        agent.training_costs.append(total_cost)
        
        if is_dqn and episode_losses:
            avg_loss = np.mean(episode_losses)
            agent.training_losses.append(avg_loss)
        
        # Track best
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Update progress bar
        if episode % 10 == 0:
            avg_reward = np.mean(all_rewards[-50:])
            avg_cost = np.mean(all_costs[-50:])
            pbar.set_postfix({
                "avg_reward": f"{avg_reward:.0f}",
                "avg_cost": f"{avg_cost:,.0f}",
                "epsilon": f"{agent.epsilon:.3f}",
                "best": f"{best_reward:.0f}",
            })
    
    return {
        "rewards": all_rewards,
        "costs": all_costs,
        "best_reward": best_reward,
        "final_avg_reward": np.mean(all_rewards[-50:]),
        "final_avg_cost": np.mean(all_costs[-50:]),
    }


def evaluate_agent(
    agent,
    env: InventoryEnv,
    n_episodes: int = 10,
    seed: int = 1000,
) -> dict:
    """
    Đánh giá agent (không explore, chỉ exploit).
    
    Returns:
        Dictionary chứa evaluation metrics
    """
    all_rewards = []
    all_costs = []
    all_stockouts = []
    all_histories = []
    
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        episode_reward = 0
        total_stockout = 0
        
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            total_stockout += info["stockout"]
        
        all_rewards.append(episode_reward)
        all_costs.append(info["total_cost"])
        all_stockouts.append(total_stockout)
        all_histories.append(env.get_history_df())
    
    return {
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "mean_cost": np.mean(all_costs),
        "std_cost": np.std(all_costs),
        "mean_stockout": np.mean(all_stockouts),
        "histories": all_histories,
    }


def train_dqn(env, agent_type="DQN", episodes=500, max_steps=100, batch_size=32):
    """
    Train a Deep Q-Network Agent or Dueling DQN Agent.
    """
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    if agent_type == "DuelingDQN":
        from agents.dueling_dqn_agent import DuelingDQNAgent
        agent = DuelingDQNAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            batch_size=batch_size,
            epsilon_decay=0.995,
            learning_rate=0.001
        )
        print("Training Dueling DQN Agent...")
    else:
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            batch_size=batch_size,
            epsilon_decay=0.995,
            learning_rate=0.001
        )
        print("Training Standard DQN Agent...")

    # Training loop
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_loss = 0
        loss_steps = 0
        
        done = False
        while not done:
            # Chọn action
            action = agent.select_action(state, training=True)
            
            # Thực hiện action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Tính toán reward và cập nhật thông tin chi phí
            episode_reward += reward
            if info["stockout"] > 0:
                episode_cost += info["stockout"] * env.stockout_cost
            if action > 0:
                episode_cost += env.fixed_order_cost + action * env.unit_order_cost
            
            # Store and Update
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                episode_loss += loss
                loss_steps += 1
            
            state = next_state
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Log stats
        if episode % 10 == 0 or episode == episodes - 1:
            avg_reward = episode_reward / (max_steps if max_steps > 0 else 1)
            avg_cost = episode_cost / (max_steps if max_steps > 0 else 1)
            avg_loss = episode_loss / (loss_steps if loss_steps > 0 else 1)
            print(f"Episode {episode}/{episodes} - "
                  f"Reward: {avg_reward:.2f}, Cost: {avg_cost:.2f}, Loss: {avg_loss:.4f}, "
                  f"Exploration: {agent.epsilon:.4f}")
    
    print("Training finished!")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train RL Agents for Inventory Management")
    parser.add_argument("--agent", type=str, default="Q-Learning", choices=["Q-Learning", "DQN", "DuelingDQN"],
                        help="Tên agent cần train")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save", action="store_true",
                        help="Lưu model sau khi huấn luyện")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  INVENTORY MANAGEMENT WITH REINFORCEMENT LEARNING")
    print("=" * 60)
    
    # 1. Generate demand data
    print("\n📦 Generating demand data...")
    demand_df = generate_seasonal_demand(n_days=730, seed=42)
    demand_array = demand_df["demand"].values
    
    # 2. Tạo environment
    env = InventoryEnv(
        max_inventory=200,
        max_order=100,
        order_step=10,
        holding_cost=1.0,
        stockout_cost=5.0,
        fixed_order_cost=20.0,
        unit_order_cost=2.0,
        lead_time=1,
        episode_length=365,
        demand_data=demand_array,
        demand_type="real",
    )
    
    n_actions = env.n_actions
    print(f"   Actions: {n_actions} (0, 10, 20, ..., 100)")
    print(f"   Episode length: 365 days")
    
    if args.agent == "Q-Learning":
        agent = QLearningAgent(
            n_actions=n_actions,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
        )
        print("\n🧠 Training Q-Learning Agent...")
        q_stats = train_agent(agent, env, n_episodes=args.episodes)
        # TODO: Cần implement save model cho Q-Learning
    elif args.agent in ["DQN", "DuelingDQN"]:
        agent = train_dqn(env, agent_type=args.agent, episodes=args.episodes, max_steps=args.max_steps, batch_size=args.batch_size)
        if args.save:
            save_path = f"models/{args.agent.lower()}_model.pth"
            agent.save(save_path)
            print(f"Model saved to {save_path}")
    
    # 5. Evaluate
    print("\n📊 Evaluating agents...")
    q_eval = evaluate_agent(agent, env, n_episodes=10)
    
    print(f"\n{'='*50}")
    print(f"{'Agent':<20} {'Avg Cost':>12} {'Avg Stockout':>15}")
    print(f"{'='*50}")
    print(f"{args.agent:<20} {q_eval['mean_cost']:>12,.0f} {q_eval['mean_stockout']:>15,.0f}")
    print(f"{'='*50}")
    
    print("\n✅ Training complete! Check 'models/' for saved models.")


if __name__ == "__main__":
    main()
