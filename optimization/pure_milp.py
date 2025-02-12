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
        b = model.addVars(num_nodes, lb=0, ub=self.battery_capacity, vtype=GRB.CONTINUOUS, name="battery")  # Battery level at nodes
        u = model.addVars(num_nodes, vtype=GRB.CONTINUOUS, lb=0, ub=num_nodes - 1, name="order")  # Ordering variables

        # Set battery full at initial position
        model.addConstr(b[0] == self.battery_capacity, name="initial_battery")

        # Objective: Minimize total travel time
        model.setObjective(
            gp.quicksum(self.distances[i, j] * x[i, j] for i, j in valid_arcs),
            GRB.MINIMIZE
        )

        # 1. Each customer must be visited exactly once
        for j in range(self.num_customers):  
            model.addConstr(
                gp.quicksum(x[i, j] for i in range(num_nodes) if (i, j) in valid_arcs) == 1,
                f"visit_{j}_in"
            )
            model.addConstr(
                gp.quicksum(x[j, k] for k in range(num_nodes) if (j, k) in valid_arcs) == 1,
                f"visit_{j}_out"
            )

        # 2. Flow conservation: Ensure balanced in/out flows
        for s in range(self.num_customers, num_nodes):  # Charging stations
            model.addConstr(
                gp.quicksum(x[i, s] for i in range(num_nodes) if (i, s) in valid_arcs) >= 
                gp.quicksum(x[s, j] for j in range(num_nodes) if (s, j) in valid_arcs) - 1,
                f"flow_station_{s}"
            )

        # 3. Battery constraints: Prevent depletion and ensure correct recharging
        for i, j in valid_arcs:
            if i < self.num_customers:  # Track battery only for customers
                model.addConstr(
                    b[j] >= b[i] - self.distances[i, j] * x[i, j],
                    f"battery_update_{i}_{j}"
                )
                model.addConstr(
                    b[j] >= 0,  # Ensure battery never drops below zero
                    f"battery_nonnegative_{j}"
                )

        # 4. Ensure battery resets at stations
        for s in range(self.num_customers, num_nodes):  # Charging stations
            for j in range(num_nodes):
                if (s, j) in valid_arcs:
                    model.addConstr(
                        b[j] >= self.battery_capacity * x[s, j],
                        name=f"station_recharge_{s}_{j}"
                    )

        # 5. Prevent backtracking (i → j → i)
        for i, j in valid_arcs:
            if (j, i) in valid_arcs:
                model.addConstr(
                    x[i, j] + x[j, i] <= 1,
                    f"no_backtrack_{i}_{j}"
                )

        # 6. Enforce a single connected path using ordering variables
        for i, j in valid_arcs:
            if i != 0 and j != 0:  # Ignore first node to allow flexible start
                model.addConstr(
                    u[i] + 1 <= u[j] + num_nodes * (1 - x[i, j]),
                    f"order_{i}_{j}"
                )

        # 7. Subtour elimination using a lazy constraint approach
        model._x = x
        model.optimize(self.subtour_elimination)

        if model.status == GRB.INFEASIBLE:
            print("Model is infeasible! Computing IIS...")
            model.computeIIS()
            model.write("infeasible_constraints.ilp")
            return None, None

        if model.status == GRB.OPTIMAL:
            active_arcs = [(i, j) for i, j in valid_arcs if x[i, j].X > 0.5]
            return model.objVal, active_arcs
        else:
            print(f"Solver did not find an optimal solution! Status: {model.status}")
            return None, None

    @staticmethod
    def subtour_elimination(model, where):
        """Lazy constraints to eliminate subtours and ensure a single connected path"""
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._x)
            selected = [(i, j) for (i, j) in model._x.keys() if vals[i, j] > 0.5]

            subtours = PureMILPSolver.find_subtours(selected)

            for tour in subtours:
                if len(tour) < len(model._x):
                    model.cbLazy(
                        gp.quicksum(model._x[i, j] for i in tour for j in tour if (i, j) in model._x) <= len(tour) - 1
                    )

    @staticmethod
    def find_subtours(selected_arcs):
        """Detects subtours in the solution"""
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


if __name__ == "__main__":
    excel_path = r"C:\Users\issam\Desktop\Files\PhD\DRPRL\data\StaticProblemInstances\GroupA\distancematrix_with_20_nodes(seconds).xlsx"
    
    solver = PureMILPSolver(excel_path, instance_num=1)
    
    cost, arcs = solver.solve()
    
    if cost is not None:
        print(f"Total cost: {cost:.2f} seconds")
        print("Route:", arcs)
    else:
        print("No feasible solution found.")
