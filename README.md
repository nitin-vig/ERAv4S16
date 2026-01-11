# ERAv4 Session 16 - Reinforcement Learning

This repository contains reinforcement learning exercises and implementations.

---

## 📋 Table of Contents

- [Q1: GridWorld with Value Iteration](#q1-gridworld-with-value-iteration)
- [Q2: TBD](#q2-tbd)
- [Installation](#installation)
- [Usage](#usage)

---

## Q1: GridWorld with Value Iteration

### Overview

A 4x4 GridWorld environment implementation for reinforcement learning, featuring:
- **Environment**: 4x4 grid (16 states)
- **Start State**: Top-left corner (state 0)
- **Goal State**: Bottom-right corner (state 15)
- **Actions**: 4 possible actions - UP, DOWN, LEFT, RIGHT
- **Rewards**: 
  - `-1` for each step
  - `0` for reaching the terminal state (goal)
- **No obstacles**: Agent can move freely within grid boundaries

### State Representation

States are numbered 0-15 in row-major order:

```
 0   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14  15
```

### Features

#### 1. **GridWorld Environment**
- Deterministic state transitions
- Boundary handling (agent stays in place when hitting walls)
- Terminal state detection
- Reward function implementation

#### 2. **Value Iteration Algorithm**
Implements the Bellman optimality equation:

```
V(s) = max_a [ R(s,a) + γ * Σ P(s'|s,a) * V(s') ]
```

**Algorithm Steps:**
1. Initialize V(s) = 0 for all states
2. Repeat until convergence:
   - For each non-terminal state:
     - Calculate action values for all actions
     - Update V(s) with maximum action value
     - Track maximum change (delta)
   - Check convergence: if delta < θ, stop
3. Extract optimal policy from converged value function

**Parameters:**
- **Discount Factor (γ)**: `1.0` (no discounting)
- **Convergence Threshold (θ)**: `1e-4`
- **Max Iterations**: `1000`

#### 3. **Policy Extraction**
After value iteration converges, the optimal policy is extracted by selecting the action that maximizes the expected return for each state.

#### 4. **Visualization**
- **Grid Visualization**: Shows state numbers, values, and policy arrows
- **Value Heatmap**: Color-coded state values
- **Policy Arrows**: Visual representation of optimal actions
- **Convergence Plot**: Tracks delta (max change) over iterations

### File Structure

```
q1/
├── GridWorld.py          # Main implementation
├── requirements.txt      # Python dependencies
└── outputs/             # Generated visualizations (created on run)
    ├── gridworld_initial.png
    ├── gridworld_optimal.png
    ├── convergence_history.png
    └── gridworld_iteration_*.png
```

### Implementation Details

**Class: `GridWorld`**

**Key Methods:**
- `__init__()`: Initialize environment, value function, and parameters
- `state_to_position(state)`: Convert state number to (row, col)
- `position_to_state(row, col)`: Convert (row, col) to state number
- `get_next_state(state, action)`: Compute next state given action
- `get_reward(state, action, next_state)`: Return reward for transition
- `step(action)`: Execute action and return (next_state, reward, done)
- `reset()`: Reset environment to start state
- `value_iteration(max_iterations, verbose)`: Run value iteration algorithm
- `extract_policy()`: Extract optimal policy from value function
- `visualize_grid(values, policy, title)`: Visualize grid with values/policy
- `print_grid_info()`: Print environment information

### Results

**Optimal Value Function:**
The value function represents the expected cumulative reward from each state following the optimal policy. With γ=1 and step reward=-1, the values represent the negative of the optimal number of steps to reach the goal.

**Optimal Policy:**
The optimal policy shows the best action to take from each state to minimize the number of steps to the goal.

Example output:
```
Optimal Value Function:
============================================================
  -6.00    -5.00    -4.00    -3.00
  -5.00    -4.00    -3.00    -2.00
  -4.00    -3.00    -2.00    -1.00
  -3.00    -2.00    -1.00     0.00
============================================================

Optimal Policy:
============================================================
 RIGHT   RIGHT   RIGHT    DOWN
  DOWN    DOWN    DOWN    DOWN
  DOWN    DOWN    DOWN    DOWN
 RIGHT   RIGHT   RIGHT    GOAL
============================================================
```

### Running Q1

```bash
cd q1
python GridWorld.py
```

**Output:**
- Prints environment information
- Runs value iteration with iteration-by-iteration progress
- Displays optimal value function and policy
- Saves visualization images

---

## Q2: TBD

*Details to be added*

---

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ERAv4S16
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies for Q1:
```bash
cd q1
pip install -r requirements.txt
```

### Dependencies (Q1)

- `numpy>=1.24.0` - Numerical operations and array handling
- `matplotlib>=3.7.0` - Visualization and plotting
- `seaborn>=0.12.0` - Enhanced color palettes

---

## Usage

### Q1: GridWorld

**Basic Usage:**
```python
from GridWorld import GridWorld

# Create environment
env = GridWorld()

# Run value iteration
V_optimal, num_iterations, delta_history = env.value_iteration()

# Extract optimal policy
optimal_policy = env.extract_policy()

# Visualize
env.visualize_grid(values=V_optimal, policy=optimal_policy)
```

**Custom Parameters:**
```python
env = GridWorld()
env.gamma = 0.9  # Change discount factor
env.theta = 1e-6  # Change convergence threshold

# Run with custom settings
V_optimal, num_iterations, delta_history = env.value_iteration(
    max_iterations=500,
    verbose=True
)
```

**Interactive Episode:**
```python
env = GridWorld()
state = env.reset()

while True:
    action = int(input("Enter action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT): "))
    next_state, reward, done = env.step(action)
    print(f"State: {next_state}, Reward: {reward}")
    
    if done:
        print("Goal reached!")
        break
```

---

## Key Concepts

### Reinforcement Learning Fundamentals

1. **Markov Decision Process (MDP)**
   - States (S): Grid positions
   - Actions (A): Movement directions
   - Transition function (P): Deterministic state transitions
   - Reward function (R): Step penalties and goal reward

2. **Value Function**
   - V(s): Expected cumulative reward from state s
   - Represents "goodness" of being in a state

3. **Policy**
   - π(s): Action to take in state s
   - Optimal policy maximizes expected return

4. **Bellman Equation**
   - Recursive relationship for value functions
   - Foundation for dynamic programming algorithms

### Value Iteration

Value iteration is a dynamic programming algorithm that:
- Iteratively improves value estimates
- Converges to optimal value function V*
- Guarantees convergence for finite MDPs
- Time complexity: O(|S|² |A| per iteration)

---

## License

*Add license information here*

## Author

*Add author information here*

## Acknowledgments

- Part of ERA V4 (Extensive & Rigorous AI) curriculum
- Session 16: Reinforcement Learning
