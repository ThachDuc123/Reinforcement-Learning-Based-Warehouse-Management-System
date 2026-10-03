# 🏭 Inventory Management with Reinforcement Learning

### Intelligent inventory replenishment using Q-Learning and Deep Q-Networks

<p align="center">

**Demand → Inventory State → RL Agent → Order Decision → Cost Optimization**

</p>

---

## 🚀 Overview

How much inventory should a warehouse order today?

Ordering too much creates **holding costs** and ties up capital.

Ordering too little creates **stockouts**, which can lead to lost sales.

Ordering too frequently increases **ordering costs**.

This project formulates inventory replenishment as a **Reinforcement Learning (RL)** problem.

Instead of manually specifying a fixed ordering rule, an RL agent learns to make daily replenishment decisions based on the current warehouse state.

```text
                    ┌─────────────────────┐
                    │   Warehouse State   │
                    │                     │
                    │ • Inventory        │
                    │ • Pending orders   │
                    │ • Day of week      │
                    │ • Recent demand    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │    RL Agent         │
                    │                     │
                    │ Q-Learning / DQN    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │   Order Decision    │
                    │                     │
                    │ 0, 10, 20, ..., N  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Warehouse Dynamics  │
                    │                     │
                    │ Demand + Lead Time  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │      Reward         │
                    │                     │
                    │ Holding + Stockout  │
                    │ + Ordering Cost     │
                    └─────────────────────┘
```

The objective is:

> **Learn an ordering policy that minimizes the long-term total inventory cost while maintaining sufficient stock to satisfy demand.**

---

# 🎯 The Problem

Traditional inventory systems often rely on manually defined policies such as:

* Fixed order quantities
* Random ordering
* `(s, S)` inventory policy
* EOQ — Economic Order Quantity

These approaches are useful baselines, but their decisions are determined by predefined rules.

This project asks a different question:

> **Can an agent learn when and how much to order by interacting with a simulated warehouse environment?**

The agent receives the current inventory state, chooses an order quantity, observes the resulting demand and inventory changes, and receives a reward based on the resulting cost.

---

# 🧠 Reinforcement Learning Formulation

The inventory problem is modeled as a **Markov Decision Process (MDP)**.

```text
State
  ↓
Agent
  ↓
Action
  ↓
Environment
  ↓
Reward
  ↓
Next State
  ↺
```

---

## 1. State Space

At each day \(t\), the agent observes a 4-dimensional state:

$$
s_t =
[
I_t,\;
DOW_t,\;
P_t,\;
D_{t-1}
]
$$

where:

| State                  | Meaning                                        |
| ---------------------- | ---------------------------------------------- |
| `inventory_level_norm` | Current inventory level                        |
| `day_of_week_norm`     | Current day of the week                        |
| `pending_orders_norm`  | Quantity already ordered but not yet delivered |
| `last_demand_norm`     | Demand from the previous day                   |

The state allows the agent to answer questions such as:

```text
How much inventory do I currently have?
        +
Is an order already on the way?
        +
What day of the week is it?
        +
What was demand yesterday?
        ↓
Should I order more inventory?
```

The input variables are normalized before being provided to the learning algorithm.

---

# 🎮 2. Action Space

The agent chooses a **discrete order quantity**.

$$
a_t \in \{0,1,2,\ldots,N-1\}
$$

The actual order quantity is:

$$
q_t = a_t \times order\_step
$$

For example, with:

```text
max_order = 100
order_step = 10
```

the agent can choose:

```text
0
10
20
30
...
100
```

This turns the business question:

> "How much should the warehouse order today?"

into a reinforcement learning action.

---

# 🔄 3. Environment Transition

Each RL step represents one day of warehouse operation.

The environment follows this sequence:

```text
1. Agent chooses order quantity
             ↓
2. New order enters the pending-order queue
             ↓
3. Orders whose lead time has finished arrive
             ↓
4. Customer demand occurs
             ↓
5. Calculate stockout
             ↓
6. Update remaining inventory
             ↓
7. Calculate reward
             ↓
8. Move to the next day
```

Stockout is calculated as:

$$
SO_t = \max(0,d_t-I_t)
$$

Remaining inventory is:

$$
I_{t+1}=\max(0,I_t-d_t)
$$

This allows the agent to experience the consequences of ordering too much or too little.

---

# 💰 4. Reward Function

The reward is designed to represent the **negative total inventory cost**.

$$
r_t =
-\left[
c_hI_t
+c_sSO_t
+\mathbb{1}_{(q_t>0)}c_f
+c_uq_t
\right]
$$

where:

| Symbol   | Meaning              |
| -------- | -------------------- |
| \(I_t\)  | End-of-day inventory |
| \(SO_t\) | Stockout quantity    |
| \(q_t\)  | Order quantity       |
| \(c_h\)  | Holding cost         |
| \(c_s\)  | Stockout penalty     |
| \(c_f\)  | Fixed ordering cost  |
| \(c_u\)  | Unit ordering cost   |

Therefore:

$$
\max \sum_t r_t
\quad\Longleftrightarrow\quad
\min \sum_t Cost_t
$$

In other words:

> **The RL agent is rewarded for making inventory decisions that reduce the long-term total cost.**

---

# 🤖 Algorithms

The project implements two reinforcement learning approaches.

## Q-Learning

**Q-Learning** is the tabular RL baseline.

```text
State
  ↓
State discretization
  ↓
Q-Table
  ↓
Best action
  ↓
Order quantity
```

It learns the expected value of taking an action in a particular inventory state.

---

## Deep Q-Network

The project also implements **DQN**.

Instead of storing every state-action value in a table, DQN uses a neural network to approximate the Q-function.

```text
State
  ↓
Neural Network
  ↓
Q-values for actions
  ↓
Selected order quantity
```

The implementation includes the standard DQN components:

* Neural network
* Experience replay
* Target network

---

# ⚖️ Baseline Policies

RL policies are evaluated alongside conventional inventory strategies.

| Method          | Type      | Decision mechanism                    |
| --------------- | --------- | ------------------------------------- |
| Random Policy   | Baseline  | Random order quantity                 |
| Fixed Order     | Baseline  | Fixed quantity                        |
| `(s, S)` Policy | Classical | Reorder based on inventory thresholds |
| EOQ             | Classical | Economic Order Quantity               |
| Q-Learning      | RL        | Learned Q-table                       |
| DQN             | Deep RL   | Neural-network Q-function             |

This comparison makes it possible to evaluate whether the learned policies behave differently from simple inventory rules.

---

# 📊 Dataset

The project currently uses **synthetic demand data** for training and evaluation.

The generated dataset contains:

* **730 days** of demand
* Weekly variation
* Seasonal variation
* Sales/event effects

Additional data files include:

```text
data/
├── demand_data.csv
├── product_catalog.csv
└── supplier_info.csv
```

The synthetic dataset allows the RL environment to be controlled and reproduced while experimenting with different inventory conditions.

---

# 📈 Analysis & Visualization

The project includes visualization utilities and saved figures.

Examples include:

### Demand analysis

```text
data
 ↓
Demand distribution
 ↓
Weekly / seasonal patterns
 ↓
Inventory simulation
```

### Training curves

Training behavior can be inspected through the generated training curves.

### Evaluation comparison

Different RL agents and baseline policies can be compared using the generated evaluation figures.

Figures are stored under:

```text
figures/
```

---

# 🗂️ Project Structure

```text
Reinforcement-Learning-Based-Warehouse-Management-System/
│
├── agents/
│   ├── __init__.py
│   ├── q_learning.py
│   ├── dqn_agent.py
│   └── baselines.py
│
├── env/
│   ├── __init__.py
│   └── inventory_env.py
│
├── data/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── demand_data.csv
│   ├── product_catalog.csv
│   └── supplier_info.csv
│
├── models/
│   ├── q_learning_model.pkl
│   └── dqn_model.pth
│
├── figures/
│   ├── demand_analysis.png
│   ├── training_curves.png
│   ├── evaluation_comparison.png
│   └── ...
│
├── utils/
│   ├── __init__.py
│   └── visualization.py
│
├── main_notebook.ipynb
├── train.py
├── test_quick.py
├── requirements.txt
└── README.md
```

---

# ⚡ Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/ThachDuc123/Reinforcement-Learning-Based-Warehouse-Management-System.git

cd Reinforcement-Learning-Based-Warehouse-Management-System
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the notebook

The easiest way to explore the complete pipeline is:

```text
main_notebook.ipynb
```

Open it using Jupyter Notebook or VS Code and execute the cells sequentially.

---

## 4. Train the agents

Alternatively:

```bash
python train.py
```

This runs the training pipeline defined by the project.

---

## 5. Quick test

```bash
python test_quick.py
```

This can be used to quickly verify that the environment and core components are working.

---

# 🔬 Experimental Pipeline

The complete experiment can be summarized as:

```text
                  Synthetic Demand
                         │
                         ↓
                ┌─────────────────┐
                │ Inventory Env   │
                │   Gymnasium     │
                └────────┬────────┘
                         │
                         ↓
                  State sₜ
                         │
          ┌──────────────┴──────────────┐
          ↓                             ↓
     Q-Learning                        DQN
          │                             │
          └──────────────┬──────────────┘
                         ↓
                  Order Decision
                         │
                         ↓
                 Warehouse Update
                         │
                         ↓
                       Cost
                         │
                         ↓
                      Reward
                         │
                         ↺
```

---

# 💡 Why Reinforcement Learning?

Inventory management is inherently sequential.

Today's ordering decision affects:

* Tomorrow's inventory
* Future stockouts
* Future holding costs
* Pending orders
* Future ordering decisions

Therefore, optimizing one day independently is not necessarily enough.

RL provides a natural framework because the agent learns a **policy over a sequence of decisions**.

The objective is not simply:

> "Choose the cheapest order today."

Instead:

> **"Learn a strategy that performs well over the entire inventory horizon."**

---

# 🏗️ Future Improvements

Several extensions can make the system closer to a real-world inventory optimization problem.

## Continuous Order Quantities

The current formulation uses discrete actions.

A future version could use **PPO with continuous actions**, allowing the agent to directly output a continuous order quantity.

```text
Current:
0 → 10 → 20 → 30 → ... → 100

Future:
0.0 → 7.3 → 18.6 → 24.1 → ... 
```

---

## Better DQN Architectures

Possible improvements include:

* Double DQN
* Dueling DQN
* Prioritized Experience Replay

These can be investigated to improve learning stability and Q-value estimation.

---

## Demand History

The current state includes the previous day's demand.

A future model could provide a longer demand history:

```text
Demand:
D(t-14) ... D(t-7) ... D(t-1)
                    ↓
                   LSTM
                    ↓
                  RL Agent
```

This could allow the agent to learn longer-term demand patterns.

---

## Real-World Data

The current experiments use synthetic demand data.

A future version could incorporate real transaction or sales data containing:

* Historical demand
* Product information
* Supplier lead times
* Ordering costs
* Inventory levels
* Seasonal effects

This would allow the simulation environment to represent a more realistic warehouse operation.

---

# 🏢 Potential Business Applications

The same RL formulation can potentially be extended to:

* Inventory replenishment
* Retail stock management
* Warehouse purchasing
* Multi-product inventory control
* Supplier lead-time optimization
* Seasonal demand management
* Automated replenishment systems

The key idea is to replace manually fixed replenishment rules with a policy that learns from the interaction between **inventory, demand, lead time and cost**.

---

# 📚 Project Context

**Course:** REL301m — Reinforcement Learning

The project demonstrates how a classical inventory-management problem can be formulated as an RL environment and solved using both **tabular and deep reinforcement learning** approaches.

---

# 🧰 Tech Stack

```text
Python
│
├── Gymnasium
├── NumPy
├── Pandas
├── PyTorch
├── Matplotlib
└── Reinforcement Learning
    ├── Q-Learning
    └── DQN
```

---

# 📌 Key Takeaways

This project demonstrates a complete RL workflow:

```text
Real-world problem
       ↓
MDP formulation
       ↓
State / Action / Reward design
       ↓
Custom Gymnasium environment
       ↓
Q-Learning + DQN
       ↓
Classical baselines
       ↓
Training
       ↓
Evaluation
       ↓
Visualization
```

The central idea is simple:

> **The agent learns when and how much to order by experiencing the long-term consequences of its inventory decisions.**

---

<p align="center">

### 🏭 Inventory Management × 🤖 Reinforcement Learning

**Learning smarter replenishment decisions through sequential decision making.**

</p>
