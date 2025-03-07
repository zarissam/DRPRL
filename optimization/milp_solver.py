from typing import List, Set, Tuple
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from environment.state import State
from environment.action import Action, ActionType
from learning.value_network import ValueNetwork

class MILPSolver:
    def __init__(self, time_limit: int = 300):
        self.time_limit = time_limit

    def solve_subproblem(self, state: State, value_network: ValueNetwork) -> Action:
        """Solve MILP subproblem to find next best action combining MILP & RL"""
        # Create Gurobi model
        model = gp.Model("DRP_Subproblem")
        model.setParam('TimeLimit', self.time_limit)
        model.setParam('LazyConstraints', 1)

        num_customers = state.drp.num_customers  # Customer count
        num_stations = state.drp.num_stations  # Charging station count
        num_nodes = num_customers + num_stations  # Total nodes
        battery_capacity = state.drp.battery_capacity

        # Unvisited customers
        unvisited = {i for i in range(num_stations, num_nodes) if state.encoding[i] == 1}
        stations = set(range(num_stations))  # Charging stations

        # Valid next nodes (excluding current node)
        valid_nodes = (unvisited | stations) - {state.current_node}

        # Decision variables: x[i, j] for choosing an arc (binary)
        x = model.addVars(valid_nodes, valid_nodes, vtype=GRB.BINARY, name="x")

        # Battery variables
        b = model.addVars(num_nodes, lb=0, ub=battery_capacity, vtype=GRB.CONTINUOUS, name="battery")

        # Flow conservation constraints
        for h in unvisited:
            model.addConstr(
                gp.quicksum(x[i, h] for i in {state.current_node} | unvisited | stations if (i, h) in x) ==
                gp.quicksum(x[h, j] for j in unvisited | stations if (h, j) in x),
                name=f"flow_conservation_{h}"
            )

        # Each customer must be visited exactly once
        for j in unvisited:
            model.addConstr(
                gp.quicksum(x[i, j] for i in {state.current_node} | unvisited | stations if (i, j) in x) == 1,
                name=f"visit_customer_{j}"
            )

        # Battery constraints
        for (i, j) in x.keys():
            travel_time = state.drp.get_travel_time(
                state.instance,
                int(state._to_matrix_index(i)),
                int(state._to_matrix_index(j))
            )
            model.addConstr(b[i] >= travel_time * x[i, j], name=f"battery_feasible_{i}_{j}")
            model.addConstr(
                b[j] <= b[i] - travel_time * x[i, j] + battery_capacity * (1 - x[i, j]),
                name=f"battery_update_{i}_{j}"
            )

        # No direct routes between charging stations
        for i in stations:
            for j in stations:
                if (i, j) in x:
                    model.addConstr(x[i, j] == 0)

        # Objective function: immediate cost + future value
        obj = gp.quicksum(
            state.drp.get_travel_time(
                state.instance,
                int(state._to_matrix_index(i)),
                int(state._to_matrix_index(j))
            ) * x[i, j]
            for (i, j) in x.keys()
        )

        # Adding Value Network estimation for future cost
        for (i, j) in x.keys():
            next_state = state.__class__(state.drp, state.instance, j)
            action = Action(ActionType.RECHARGE if j < num_stations else ActionType.MOVE, j)
            action.execute(next_state)
            obj += value_network.predict(next_state) * x[i, j]

        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        # Extract solution (after optimization)
        best_milp_action = None
        if model.status == GRB.OPTIMAL:
            for (i, j) in x.keys():
                if i == state.current_node and x[i, j].X > 0.5:
                    best_milp_action = Action(ActionType.RECHARGE if j < num_stations else ActionType.MOVE, j)
                    break

        if best_milp_action is None:
            raise ValueError("No solution found")

        return best_milp_action
