import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class Agent(nn.Module):
    def __init__(self, envAI):
        super().__init__()
        self.envAI = envAI

        self.sequential1 = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)  # Softmax pour obtenir des probabilités
        )
    
    def forward(self, state):
        return self.sequential1(state)
