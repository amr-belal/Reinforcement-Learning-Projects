# 🧩 Gridworld Agent — Value Iteration Project

## 📘 Project Overview
This project implements a simple **Gridworld environment** where an agent learns to reach a goal using **Value Iteration**, one of the core algorithms in **Reinforcement Learning (RL)**.  

The Gridworld problem provides a clean and intuitive setting to understand:
- State transitions  
- Reward structures  
- Bellman optimality equations  
- Policy and value updates  

---

## 🧠 Concepts Involved

### 🔹 Markov Decision Process (MDP)
Gridworld is modeled as an MDP defined by:
- **States (S):** Each cell in the grid
- **Actions (A):** {Up, Down, Left, Right}
- **Transition Function (T):** Probability of moving to a new state given the current state and action
- **Reward Function (R):** Immediate reward received after taking an action
- **Discount Factor (γ):** Determines how much future rewards are worth

### 🔹 Value Iteration Algorithm
Value Iteration uses the Bellman Optimality Equation to iteratively improve the value function:

\[
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k(s')]
\]

The process continues until **convergence**, i.e., when the values stabilize.

---

## ⚙️ Implementation Details

### Environment Setup
- Grid size: 3x3
- Goal state: Center cell `(1,1)`
- Reward structure:
  - Goal: `0`
  - Every other move: `-0.4`
- Discount factor (γ): 0.9
- Convergence threshold: 1e-4

### Algorithm Steps
1. Initialize all state values to zero.
2. Iteratively update each state's value using the Bellman equation.
3. Stop when the maximum change in the value function between iterations is below a small threshold.
4. Derive the optimal policy by choosing the action that maximizes expected value in each state.

---

## 📊 Example Output

### ✅ Value Iteration Converged
