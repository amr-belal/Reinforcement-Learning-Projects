# Q-Learning Taxi-v3 Project

This project implements a **Q-learning agent** to solve the classic **Taxi-v3** environment from the Gymnasium library. The goal of the environment is for the taxi to pick up a passenger and deliver them to the correct destination while avoiding illegal moves and maximizing rewards.

---

## 🚀 Overview

The project uses **Q-learning**, an off-policy, model-free reinforcement learning algorithm. We train a Q-table that stores the expected future rewards for each **state-action** pair. Over many episodes, the agent learns an optimal policy for navigating the Taxi environment.

---

## 📦 Requirements

Make sure you have the following installed:

```bash
gymnasium
numpy
```

Install Gymnasium (with classic control environments):

```bash
pip install gymnasium[classic-control]
```

---

## 🧠 Q-Learning Algorithm

Q-learning is based on the update rule:

```
Q(s,a) ← (1 − α) * Q(s,a) + α * ( r + γ * max_a' Q(s',a') )
```

Where:

* **α (alpha)** → Learning rate
* **γ (gamma)** → Discount factor
* **r** → Reward
* **s, a** → Current state and action
* **s'** → Next state
* **max Q(s', a')** → Best future value

We use an **ε-greedy policy** to balance exploration and exploitation.

---

## 📄 Code Summary

Key features of the project:

* Initialize Q-table with zeros
* Use ε-greedy strategy to explore
* Update Q-values according to Q-learning equation
* Decay ε gradually each episode
* Render trained agent for visualization

---

## 📊 Hyperparameters

The main training hyperparameters used are:

```python
alpha = 0.9       # Learning rate
gamma = 0.95      # Discount factor
epsilon = 1.0     # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 10000
max_steps = 100
```

These control how fast the agent learns, how much it explores, and when it begins exploiting its learned policy.

---

## 🧪 Testing the Agent

After training, the agent is tested using `render_mode="human"` for visual feedback. The policy becomes fully greedy (uses only `argmax`).

---

## 📁 Project Structure

```
📦 taxi-q-learning
 ┣ 📜 q_learning_taxi.py
 ┣ 📜 README.md
 ┗ 📂 results (optional)
```

---

## 📈 Possible Improvements

Here are some enhancements you can try:

* Plot reward per episode
* Implement SARSA for comparison
* Use Double Q-learning to reduce overestimation
* Convert the agent to a DQN using neural networks
* Add logging and evaluation metrics

---

## 🧑‍💻 Author

Amr Belal — Reinforcement Learning Student & Developer

Feel free to expand and build upon this project!
