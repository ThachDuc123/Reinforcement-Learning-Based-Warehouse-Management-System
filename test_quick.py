"""Quick test to verify all components work."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.inventory_env import InventoryEnv
from agents.q_learning import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.baselines import RandomPolicy, SsPolicyAgent
from train import train_agent, evaluate_agent
from data.data_generator import generate_seasonal_demand
import numpy as np

print("=" * 50)
print("  QUICK TEST - All Components")
print("=" * 50)

# 1. Data
print("\n1. Testing data generation...")
df = generate_seasonal_demand(n_days=100, seed=42)
print(f"   ✅ Generated {len(df)} rows, mean demand: {df['demand'].mean():.1f}")

# 2. Environment
print("\n2. Testing environment...")
env = InventoryEnv(episode_length=30, demand_data=df['demand'].values, demand_type='real')
state, _ = env.reset(seed=42)
print(f"   ✅ State: {state}, Actions: {env.n_actions}")

# 3. Q-Learning
print("\n3. Testing Q-Learning (10 episodes)...")
q_agent = QLearningAgent(n_actions=env.n_actions)
q_stats = train_agent(q_agent, env, n_episodes=10, verbose=False)
q_eval = evaluate_agent(q_agent, env, n_episodes=3)
print(f"   ✅ Avg Cost: {q_eval['mean_cost']:,.0f}, Stockout: {q_eval['mean_stockout']:,.0f}")

# 4. DQN
print("\n4. Testing DQN (10 episodes)...")
dqn_agent = DQNAgent(state_dim=4, n_actions=env.n_actions)
dqn_stats = train_agent(dqn_agent, env, n_episodes=10, verbose=False)
dqn_eval = evaluate_agent(dqn_agent, env, n_episodes=3)
print(f"   ✅ Avg Cost: {dqn_eval['mean_cost']:,.0f}, Stockout: {dqn_eval['mean_stockout']:,.0f}")

# 5. Baselines
print("\n5. Testing baselines...")
random_pol = RandomPolicy(env.n_actions)
ss_pol = SsPolicyAgent(env.n_actions)
r_eval = evaluate_agent(random_pol, env, n_episodes=3)
s_eval = evaluate_agent(ss_pol, env, n_episodes=3)
print(f"   ✅ Random Cost: {r_eval['mean_cost']:,.0f}")
print(f"   ✅ (s,S) Cost: {s_eval['mean_cost']:,.0f}")

print("\n" + "=" * 50)
print("  ✅ ALL TESTS PASSED!")
print("=" * 50)
