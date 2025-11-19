
# 🎮 Attention-Augmented DQN for Atari Breakout

This project implements a **Deep Q-Network (DQN)** agent capable of learning to play Atari Breakout from raw pixels. The architecture is enhanced with a **Convolutional Block Attention Module (CBAM)** to improve the agent's ability to focus on relevant game features (like the ball and paddle) while ignoring background noise.

## 🧠 Key Features

* **Vanilla DQN Core:** Implements the classic DQN algorithm with Experience Replay and Target Networks.
* **Attention Mechanism (CBAM):** Integrates Channel and Spatial attention modules into the CNN feature extractor to boost performance and stability.
* **Preprocessing Wrapper:** Custom Gymnasium wrapper for frame resizing ($84 \times 84$), grayscale conversion, and frame stacking (4 frames).
* **Laptop Friendly:** Optimized hyperparameters to allow training on consumer-grade hardware (CPU/Lower-end GPU).

## 📂 Project Structure

* **`train.py`**: The main entry point to start training the agent.
* **`test.py`**: Loads a trained model (`.pth`) and renders the game to watch the agent play.
* **`model.py`**: Defines the Neural Network architecture (CNN + CBAM + FC Layers).
* **`dqn_agent.py`**: Contains the Agent class (Action selection, Epsilon-Greedy, Learning logic).
* **`replay_buffer.py`**: Implements the Experience Replay memory.
* **`breakout.py`**: Custom environment wrapper for preprocessing Atari frames.

## 🛠️ Installation

1.  **Clone the repository** (if applicable) or download the files.
2.  **Install dependencies**:

```bash
pip install torch torchvision numpy gymnasium[atari,accept-rom-license] ale-py opencv-python
````

3.  **Auto-Install ROMs**:
    The script attempts to download ROMs automatically. If you face issues, run:
    ```bash
    pip install "autorom[accept-rom-license]"
    AutoROM --accept-license
    ```

## 🚀 Usage

### 1\. Training the Agent

To start training from scratch:

```bash
python train.py
```

  * *Note:* By default, rendering is set to `'rgb_array'` (no window) for faster training. You can change it to `'human'` in `train.py` to watch the training process.
  * The model will be saved as `breakout_model.pth` after training completes.

### 2\. Watching the Agent (Testing)

To watch your trained agent play:

```bash
python test.py
```

  * Ensure `breakout_model.pth` exists in the directory.
  * This mode disables epsilon-greedy exploration (plays purely based on learned policy).

## 🧩 Architecture Details

### The Network (CNN + CBAM)

Instead of a standard CNN, this project inserts a **CBAM** block after the convolutional layers.

1.  **Input:** Stack of 4 Grayscale Frames $(4, 84, 84)$.
2.  **Conv Layers:** 3 Convolutional layers to extract features.
3.  **Channel Attention:** Focuses on *what* features are important (e.g., moving objects vs static walls).
4.  **Spatial Attention:** Focuses on *where* the important features are located.
5.  **Fully Connected:** Output layer with 4 actions (No-op, Fire, Right, Left).

## 📊 Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `batch_size` | 32 | Number of samples per training step |
| `gamma` | 0.99 | Discount factor for future rewards |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_decay` | 0.995 | Decay rate per step |
| `epsilon_min` | 0.1 | Minimum exploration rate |
| `target_update` | 1000 | Frequency of updating Target Net |
| `buffer_size` | 100,000 | Replay Memory capacity |

## 📝 References

1.  **DQN:** Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*.
2.  **CBAM:** Woo, S., et al. (2018). "CBAM: Convolutional Block Attention Module." *ECCV*.

-----

*Created with ❤️ for Reinforcement Learning Enthusiasts.*
