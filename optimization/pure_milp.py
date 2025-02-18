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

        # Allow arcs only from a customer to a station if not both nodes are stations.
        valid_arcs = [(i, j) for i in range(num_nodes) for j in range(num_nodes)
                    if i != j and self.distances[i, j] < self.battery_capacity
                    and  (i < self.num_customers or j < self.num_customers)]

        # Decision variables: route selection and battery levels.
        x = model.addVars(valid_arcs, vtype=GRB.BINARY, name="x")
        b = model.addVars(num_nodes, lb=0, ub=self.battery_capacity, vtype=GRB.CONTINUOUS, name="battery")

        # Identify stations (indices num_customers to num_nodes-1)
        stations = list(range(self.num_customers, num_nodes))
        # Choose a start and an end station (ensure they are distinct)
        start_station = int(np.random.choice(stations))
        stations_end = stations.copy()
        stations_end.remove(start_station)
        end_station = int(np.random.choice(stations_end))
        #print(f"Start station: {start_station}, End station: {end_station}")
        
        # For the starting station, fix battery to full.
        model.addConstr(b[start_station] == self.battery_capacity, name="start_battery")
        # For intermediate stations (and optionally the end station) we require full battery if visited.
        for s in range(self.num_customers, num_nodes):
            if s != start_station:  # You might also relax this for the end station if desired.
                model.addConstr(b[s] == self.battery_capacity, name=f"station_battery_{s}")

        # ----- Flow / Visit constraints -----
        # For the start station: out_arcs - in_arcs = 1
        """
        model.addConstr(
            gp.quicksum(x[start_station, j] for j in range(num_nodes) if (start_station, j) in valid_arcs) -
            gp.quicksum(x[i, start_station] for i in range(num_nodes) if (i, start_station) in valid_arcs)
            == 1, name="net_flow_start")


        # For the end station: in_arcs - out_arcs = 1
        model.addConstr(
            gp.quicksum(x[i, end_station] for i in range(num_nodes) if (i, end_station) in valid_arcs) -
            gp.quicksum(x[end_station, j] for j in range(num_nodes) if (end_station, j) in valid_arcs)
            == 1, name="net_flow_end")
        """


        # For customers (nodes 0 to num_customers-1):
        # Each customer must be visited exactly once: one incoming and one outgoing arc.
        for j in range(self.num_customers):
            model.addConstr(gp.quicksum(x[i, j] for i in range(num_nodes) if (i, j) in valid_arcs) == 1,
                            name=f"cust_{j}_in")
            model.addConstr(gp.quicksum(x[j, k] for k in range(num_nodes) if (j, k) in valid_arcs) == 1,
                            name=f"cust_{j}_out")

        # For any intermediate station (other than start and end), enforce flow conservation if used.
        for s in range(self.num_customers, num_nodes):
            if s != start_station and s != end_station:
                model.addConstr(gp.quicksum(x[i, s] for i in range(num_nodes) if (i, s) in valid_arcs) -
                                gp.quicksum(x[s, j] for j in range(num_nodes) if (s, j) in valid_arcs) == 0,
                                name=f"flow_station_{s}")

        # ----- Battery constraints -----
        for i, j in valid_arcs:
            # Ensure sufficient battery to travel arc (i,j)
            model.addConstr(b[i] - self.distances[i, j]*x[i, j] >= 0,
                            name=f"battery_feasible_{i}_{j}")
            if j < self.num_customers:
                # Battery update if arc (i,j) is used (big-M formulation)
                model.addConstr(b[j] <= b[i] - self.distances[i, j]*x[i, j] +
                                self.battery_capacity*(1 - x[i, j]),
                                name=f"battery_update_{i}_{j}")
                



        # Objective: minimize total travel time.
        model.setObjective(gp.quicksum(self.distances[i, j]*x[i, j] for i, j in valid_arcs),
                        GRB.MINIMIZE)

        # ----- Lazy subtour elimination constraints (same as before) -----
        model._x = x
        model.Params.lazyConstraints = 1
        model.optimize(self.subtour_elimination)

        if model.status == GRB.INFEASIBLE:
            print("Model is infeasible! Computing IIS...")
            model.computeIIS()
            model.write("infeasible_constraints.ilp")
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"Infeasible constraint: {c.ConstrName}")
            return None, None

        if model.status == GRB.OPTIMAL:
            active_arcs = [(i, j) for (i, j) in valid_arcs if x[i, j].X > 0.5]
            return model.objVal, active_arcs
        else:
            print(f"Solver did not find an optimal solution! Status: {model.status}")
            return None, None

    def find_subtours(self, selected_arcs):
        """Return a list of subtours, each as an ordered list of nodes."""
        n = self.num_customers + self.num_stations
        visited = [False] * n
        subtours = []
        for i in range(n):
            if not visited[i]:
                current_tour = []
                j = i
                while not visited[j]:
                    visited[j] = True
                    current_tour.append(j)
                    successors = [k for (p, k) in selected_arcs if p == j]
                    if successors:
                        j = successors[0]
                    else:
                        break
                subtours.append(current_tour)
        return subtours

    def subtour_elimination(self, model, where):
        """Lazy constraint callback using standard subtour elimination:
        For any subtour S (with |S| < total nodes), enforce:
            ∑₍i,j∈S₎ x[i,j] ≤ |S| – 1
        """
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._x)
            selected = [(i, j) for (i, j) in model._x.keys() if vals[i, j] > 0.5]
            total_nodes = self.num_customers + self.num_stations
            for tour in self.find_subtours(selected):
                if len(tour) < total_nodes:
                    model.cbLazy(
                        gp.quicksum(model._x[i, j] for i in tour for j in tour if (i, j) in model._x)
                        <= len(tour) - 1
                    )

    def visualize_solution(self, arcs, cost):
        """Visualize solution with customers and stations retaining their original indexes."""
        def get_circular_positions(n, radius, center=(0.5, 0.5)):
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            positions = np.zeros((n, 2))
            positions[:, 0] = center[0] + radius * np.cos(angles)
            positions[:, 1] = center[1] + radius * np.sin(angles)
            return positions

        # Compute positions: customers first then stations.
        customer_pos = get_circular_positions(self.num_customers, 0.7)
        station_pos = get_circular_positions(self.num_stations, 0.3)
        positions = np.vstack([customer_pos, station_pos])
        
        # Setup plot
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        ax.set_facecolor('aliceblue')

        # Plot customers (blue circles). Customers have indices 0 to self.num_customers-1.
        plt.scatter(positions[:self.num_customers, 0], positions[:self.num_customers, 1],
                    c='blue', s=100, label='Customers')
        # Plot stations (red squares). Stations start at index self.num_customers.
        plt.scatter(positions[self.num_customers:, 0], positions[self.num_customers:, 1],
                    c='red', s=200, marker='s', label='Stations')

        # Add labels: annotate customers and stations using their original indices.
        for i in range(self.num_customers):
            plt.annotate(f'{i}', positions[i], xytext=(5, 5), textcoords='offset points')
        for i in range(self.num_stations):
            # Station index in positions is i + self.num_customers.
            plt.annotate(f'{i + self.num_customers}', positions[i + self.num_customers], xytext=(5, 5), textcoords='offset points')

        # Plot routes with arrows
        for idx, (i, j) in enumerate(arcs):
            plt.arrow(positions[i, 0], positions[i, 1],
                    positions[j, 0] - positions[i, 0], positions[j, 1] - positions[i, 1],
                    head_width=0.02, color=plt.cm.rainbow(idx / len(arcs)),
                    length_includes_head=True, alpha=0.6)

        plt.title(f'DRP Solution\nTotal Cost: {cost:.1f} seconds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


    def get_ordered_path(self, arcs):
        """Convert unordered arcs into a proper sequence"""
        # Start from depot (node 0)
        path = [0]
        current = 0
        used_arcs = set()
        
        while len(used_arcs) < len(arcs):
            for i, j in arcs:
                if i == current and (i,j) not in used_arcs:
                    path.append(j)
                    used_arcs.add((i,j))
                    current = j
                    break
                    
        # Validate path
        if len(path) != len(arcs) + 1:
            print("Warning: Path is disconnected!")
            
        return path

# Update main to include visualization
if __name__ == "__main__":
    excel_path = r"C:\Users\issam\Desktop\Files\PhD\DRPRL\data\StaticProblemInstances\GroupA\distancematrix_with_20_nodes(seconds).xlsx"
    
    solver = PureMILPSolver(excel_path, instance_num=1)
    cost, arcs = solver.solve()
    
    if cost is not None:
        print(f"Total cost: {cost:.2f} seconds")
        print("Route:", arcs)
        solver.visualize_solution(arcs, cost)  # Add visualization call
        #path = solver.get_ordered_path(arcs)
        #print("Ordered path:", ' → '.join(str(x) for x in path))
    else:
        print("No feasible solution found.")
