import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PureMILPSolver:
    def __init__(self, excel_path: str, instance_num: int = 1):
        self.instance_num = instance_num
        self.battery_capacity = 1800
        self.num_customers = 20
        self.num_stations = 5
        
        # Load distance matrix
        sheet_name = f'D{instance_num}'
        self.distances = pd.read_excel(excel_path, sheet_name=sheet_name, header=None).values
        print(f"Loaded distance matrix for instance {instance_num} with shape {self.distances.shape}")

    def solve(self):
        model = gp.Model("Pure_DRP")
        model.setParam('TimeLimit', 300)

        num_nodes = self.num_customers + self.num_stations

        # Define valid travel arcs (skip unnecessary ones)
        valid_arcs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) 
                      if i != j and self.distances[i, j] < self.battery_capacity]

        # Decision Variables
        x = model.addVars(valid_arcs, vtype=GRB.BINARY, name="x")  # Route selection
        b = model.addVars(self.num_customers, lb=0, ub=self.battery_capacity, vtype=GRB.CONTINUOUS, name="battery")  # Battery level at customers

        # Objective: Minimize total travel time
        model.setObjective(
            gp.quicksum(self.distances[i, j] * x[i, j] for i, j in valid_arcs),
            GRB.MINIMIZE
        )

        # Constraints

        # 1. Each customer must be visited exactly once
        for j in range(self.num_customers):  
            model.addConstr(gp.quicksum(x[i, j] for i in range(num_nodes) if (i, j) in valid_arcs) == 1, f"visit_{j}_in")
            model.addConstr(gp.quicksum(x[j, k] for k in range(num_nodes) if (j, k) in valid_arcs) == 1, f"visit_{j}_out")

        # 2. Flow conservation: Only enforce for customers to reduce model size
        for j in range(self.num_customers):  
            model.addConstr(
                gp.quicksum(x[i, j] for i in range(num_nodes) if (i, j) in valid_arcs) ==
                gp.quicksum(x[j, k] for k in range(num_nodes) if (j, k) in valid_arcs),
                f"flow_conservation_{j}"
            )

        for s in range(self.num_customers, num_nodes):  # Charging stations
            model.addConstr(
                gp.quicksum(x[i, s] for i in range(num_nodes) if (i, s) in valid_arcs) ==
                gp.quicksum(x[s, j] for j in range(num_nodes) if (s, j) in valid_arcs),
                f"flow_station_{s}"
    )


        # 3. Battery constraints: Battery decreases after each trip (Only track at customers)
        for i in range(self.num_customers):
            for j in range(self.num_customers):
                if (i, j) in valid_arcs:
                    model.addConstr(
                        b[j] <= b[i] - self.distances[i, j] + self.battery_capacity * (1 - x[i, j]),
                        name=f"battery_decrease_{i}_{j}"
                    )

        # 4. Recharging: If visiting a station, battery resets to full
        for i in range(self.num_customers, num_nodes):  # For each charging station
            for j in range(self.num_customers):
                if (i, j) in valid_arcs:
                    model.addConstr(
                        b[j] >= self.battery_capacity * x[i, j],  # If going to station i, recharge fully
                        name=f"battery_recharge_{i}_{j}"
                    )

        # 5. No station-to-station moves
        for i in range(self.num_customers, num_nodes):
            for j in range(self.num_customers, num_nodes):
                if i != j and (i, j) in valid_arcs:
                    model.addConstr(x[i, j] == 0, f"station_no_move_{i}_{j}")

        # Subtour elimination will be handled dynamically using lazy constraints
        model.update()
        model.write("model_debug.lp")
        print("Model constraints and variables saved to model_debug.lp for debugging.")

        model._x = x  # Store variable for callback
        model.optimize(self.subtour_elimination)  # Solve with subtour elimination

        # Check solver status
        if model.status == GRB.INFEASIBLE:
            print("Model is infeasible! Computing IIS...")
            model.computeIIS()
            model.write("infeasible_constraints.ilp")
            return None, None

        # Extract solution
        if model.status == GRB.OPTIMAL:
            active_arcs = [(i, j) for i, j in valid_arcs if x[i, j].X > 0.5]
            return model.objVal, active_arcs
        else:
            print(f"Solver did not find an optimal solution! Status: {model.status}")
            return None, None

    @staticmethod
    def subtour_elimination(model, where):
        """Callback function to eliminate subtours dynamically"""
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._x)
            selected = [(i, j) for (i, j) in model._x.keys() if vals[i, j] > 0.5]

            # Find subtours
            subtours = PureMILPSolver.find_subtours(selected)

            # Add lazy constraints to remove subtours
            for tour in subtours:
                if len(tour) < len(model._x):  # Ensure we have a full tour
                    model.cbLazy(gp.quicksum(model._x[i, j] for i in tour for j in tour if (i, j) in model._x) <= len(tour) - 1)

    @staticmethod
    def find_subtours(selected_arcs):
        """Detect subtours in the current solution"""
        nodes = set(i for i, _ in selected_arcs) | set(j for _, j in selected_arcs)
        subtours = []
        unvisited = nodes.copy()

        while unvisited:
            current = unvisited.pop()
            tour = {current}
            while True:
                for i, j in selected_arcs:
                    if i == current and j in unvisited:
                        current = j
                        tour.add(j)
                        unvisited.remove(j)
                        break
                else:
                    break
            if len(tour) > 1:
                subtours.append(tour)

        return subtours

    def visualize_solution(self, arcs, cost):
        """Visualize solution with stations and customers"""
        def get_circular_positions(n, radius, center=(0.5, 0.5)):
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            positions = np.zeros((n, 2))
            positions[:, 0] = center[0] + radius * np.cos(angles)
            positions[:, 1] = center[1] + radius * np.sin(angles)
            return positions

        # Calculate node positions
        station_pos = get_circular_positions(self.num_stations, 0.3)
        customer_pos = get_circular_positions(self.num_customers, 0.7)
        positions = np.vstack([station_pos, customer_pos])

        # Setup plot
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        ax.set_facecolor('aliceblue')

        # Plot stations (red squares)
        plt.scatter(positions[:self.num_stations,0], positions[:self.num_stations,1],
                   c='red', s=200, marker='s', label='Stations')

        # Plot customers (blue circles)
        plt.scatter(positions[self.num_stations:,0], positions[self.num_stations:,1],
                   c='blue', s=100, label='Customers')

        # Add labels
        for i in range(self.num_stations):
            plt.annotate(f'S{i}', positions[i], xytext=(5,5), textcoords='offset points')
        for i in range(self.num_customers):
            plt.annotate(f'C{i}', positions[i+self.num_stations], xytext=(5,5), textcoords='offset points')

        # Plot routes with arrows
        for idx, (i,j) in enumerate(arcs):
            plt.arrow(positions[i,0], positions[i,1],
                     positions[j,0]-positions[i,0], positions[j,1]-positions[i,1],
                     head_width=0.02, color=plt.cm.rainbow(idx/len(arcs)),
                     length_includes_head=True)

        plt.title(f'DRP Solution\nTotal Cost: {cost:.1f} seconds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    excel_path = r"C:\Users\issam\Desktop\Files\PhD\DRPRL\data\StaticProblemInstances\GroupA\distancematrix_with_20_nodes(seconds).xlsx"
    
    solver = PureMILPSolver(excel_path, instance_num=1)
    
    cost, arcs = solver.solve()
    
    if cost is not None:
        print(f"Total cost: {cost:.2f} seconds")
        print("Route:", arcs)
        solver.visualize_solution(arcs, cost)  # Add visualization
    else:
        print("No feasible solution found.")
