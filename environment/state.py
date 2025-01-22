from dataclasses import dataclass
import numpy as np
from typing import List, Set
from ..drp import DRP

@dataclass
class State:
    """
    Represents the state of the DRP using binary encoding.
    First |R| components represent charging stations (all 1s),
    Remaining |N| components represent nodes (0=visited, 1=unvisited)
    """
    drp: DRP  # Reference to DRP instance
    instance: int  # Which problem instance (D1-D20)
    encoding: np.ndarray  # Binary state encoding
    current_node: int  # Current node position
    battery_level: float  # Current battery level
    
    def __init__(self, drp: DRP, instance: int, current_node: int = 0):
        self.drp = drp
        self.instance = instance
        self.current_node = current_node
        self.battery_level = drp.battery_capacity
        # Initialize encoding using DRP parameters
        self.encoding = np.ones(drp.num_nodes + drp.num_stations, dtype=int)
        
    def is_terminal(self) -> bool:
        """Check if all nodes have been visited"""
        return np.sum(self.encoding[self.drp.num_stations:]) == 0
    
    def _to_matrix_index(self, state_idx: int) -> int:
        """Convert state encoding index to distance matrix index"""
        if state_idx < self.drp.num_stations:
            # Convert station index 0-4 to 20-24
            return state_idx + self.drp.num_nodes
        else:
            # Convert node index 5-24 to 0-19
            return state_idx - self.drp.num_stations

    def _to_state_index(self, matrix_idx: int) -> int:
        """Convert distance matrix index to state encoding index"""
        if matrix_idx >= self.drp.num_nodes:
            # Convert station index 20-24 to 0-4
            return matrix_idx - self.drp.num_nodes
        else:
            # Convert node index 0-19 to 5-24
            return matrix_idx + self.drp.num_stations

    def visit_node(self, node_idx: int) -> None:
        """Update state after visiting a node"""
        matrix_idx = self._to_matrix_index(node_idx)
        travel_time = self.drp.get_travel_time(
            self.instance, 
            self._to_matrix_index(self.current_node),
            matrix_idx
        )
        
        if node_idx >= self.drp.num_stations:  # Only mark nodes as visited
            self.encoding[node_idx] = 0
            
        self.current_node = node_idx  # Store state index
        self.battery_level -= travel_time
        
    def recharge(self, station_idx: int) -> None:
        """Update state after recharging at a station"""
        if station_idx >= self.drp.num_stations:
            raise ValueError("Invalid charging station index")
            
        travel_time = self.drp.get_travel_time(
            self.instance,
            self._to_matrix_index(self.current_node), 
            self._to_matrix_index(station_idx)
        )
        
        self.current_node = station_idx
        self.battery_level = self.drp.battery_capacity - travel_time