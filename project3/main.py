import torch
import gymnasium as gym
from PIL import Image
import numpy as np
import os
from breakout import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   


environment = DQNBreakout(device=device,render_mode="human")

# state = environment.reset()
state, info = environment.reset()

for _ in range(100):
    action = environment.action_space.sample()

    state, reward, done, truncated, info= environment.step(action)
