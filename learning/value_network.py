import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
from environment.state import State

class ValueNetwork(nn.Module):
    def __init__(self, num_nodes: int = 20, num_stations: int = 5, 
                 hidden_size: int = 64, learning_rate: float = 0.001):
        super().__init__()
        
        self.input_size = num_nodes + num_stations + 1  # encoding + battery level
        
        # Neural network architecture
        self.network = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def _preprocess_state(self, state: State) -> torch.Tensor:
        """Convert state to tensor format"""
        # Combine encoding and normalized battery level
        features = np.concatenate([
            state.encoding,
            [state.battery_level / state.drp.battery_capacity]
        ])
        return torch.FloatTensor(features)
    
    def forward(self, state: State) -> torch.Tensor:
        """Predict value for a single state"""
        x = self._preprocess_state(state)
        return self.network(x)
    
    def predict(self, state: State) -> float:
        """Predict value for a state (returns scalar)"""
        self.eval()
        with torch.no_grad():
            value = self.forward(state)
        return value.item()
    
    def train_step(self, states: List[State], values: List[float]) -> float:
        """Train network on batch of state-value pairs"""
        self.train()
        self.optimizer.zero_grad()
        
        # Convert states to batch
        X = torch.stack([self._preprocess_state(s) for s in states])
        y = torch.FloatTensor(values).reshape(-1, 1)
        
        # Forward pass
        predictions = self.network(X)
        loss = self.loss_fn(predictions, y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()