import numpy as np
from utils.parser import parse_instance, parsing_static_files
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from itertools import permutations

class DRP:
    def __init__(self, excel_path, battery_capacity=1800):
        """
        Initialize DRP instance
        Args:
            excel_path (str): Path to Excel file containing distance matrices
            battery_capacity (int): Drone battery capacity in seconds (default 1 hour)
        """
        self.excel_path = excel_path
        self.battery_capacity = battery_capacity
        self.num_customers = 20  # Fixed number of customer nodes
        self.num_stations = 5  # Fixed number of charging stations
        self.distance_matrices = {}
        
        # Load all instances (D1-D20)
        for i in range(1, 21):
            sheet_name = f'D{i}'
            self.distance_matrices[sheet_name] = parse_instance(excel_path, sheet_name)

    def get_travel_time(self, instance, from_node, to_node):
        """Get travel time between two nodes for a specific instance"""
        matrix = self.distance_matrices[f'D{instance}']
        return matrix[from_node, to_node]

    def calculate_route_cost(self, instance, route):
        """Calculate total travel time for a route"""
        total_time = 0
        for i in range(len(route)-1):
            total_time += self.get_travel_time(instance, route[i], route[i+1])
        return total_time

    def is_route_feasible(self, instance, route):
        """Check if route is feasible considering battery constraints"""
        current_battery = self.battery_capacity
        for i in range(len(route)-1):
            travel_time = self.get_travel_time(instance, route[i], route[i+1])
            if travel_time > current_battery:
                return False
            current_battery -= travel_time
            if route[i+1] >= self.num_customers:  # Charging stations are last 5 nodes
                current_battery = self.battery_capacity
        return True

    
    def __repr__(self):
        pass
    