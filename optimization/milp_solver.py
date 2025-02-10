from typing import List, Set, Tuple
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from environment.state import State
from environment.action import Action, ActionType
from learning.policy import Policy

class MILPSolver:
    def __init__(self, time_limit: int = 300):
        self.time_limit = time_limit

    def solve_subproblem(self, state: State, value_network) -> Action:
        """Solve MILP subproblem to find next immediate action"""
        # Create model
        model = gp.Model("DRP_Subproblem")
        model.setParam('TimeLimit', self.time_limit)

        # Get valid next nodes (exclude current node)
        unvisited = {i for i in range(state.drp.num_stations, 
                                    state.drp.num_nodes + state.drp.num_stations) 
                    if state.encoding[i] == 1}
        stations = set(range(state.drp.num_stations))
        valid_nodes = (unvisited | stations) - {state.current_node}  # Exclude current node
        
        # Variables - only for valid moves
        x = {}
        for j in valid_nodes:  # Only create variables for other nodes
            x[j] = model.addVar(vtype=GRB.BINARY, name=f'x_{j}')

        # Must move to a different node
        model.addConstr(gp.quicksum(x[j] for j in x) == 1)

        # Battery feasibility constraint
        for j in x:
            travel_time = state.drp.get_travel_time(
                state.instance,
                state._to_matrix_index(state.current_node),
                state._to_matrix_index(j)
            )
            if travel_time > state.battery_level:
                model.addConstr(x[j] == 0)

        # No station-to-station moves
        if state.current_node < state.drp.num_stations:  # If at station
            for j in stations:
                if j in x:
                    model.addConstr(x[j] == 0)

        # Visit each customer exactly once (maintained by state.encoding)
        for j in unvisited:
            if j in x:
                model.addConstr(x[j] <= 1)  # Can't visit more than once

        # Objective: immediate cost + future value
        obj = gp.quicksum(
            state.drp.get_travel_time(
                state.instance,
                state._to_matrix_index(state.current_node),
                state._to_matrix_index(j)
            ) * x[j] for j in x
        )

        # Add future value estimation
        for j in x:
            next_state = state.__class__(state.drp, state.instance, j)
            action = Action(
                ActionType.RECHARGE if j < state.drp.num_stations else ActionType.MOVE,
                j
            )
            action.execute(next_state)
            obj += value_network.predict(next_state) * x[j]

        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        # Return selected action
        if model.status == GRB.OPTIMAL:
            for j in x:
                if x[j].X > 0.5:
                    return Action(
                        ActionType.RECHARGE if j < state.drp.num_stations else ActionType.MOVE,
                        j
                    )
        return None

    def _find_subtours(self, arcs: List[Tuple[int, int]]) -> List[Set[int]]:
        """Find subtours in current solution"""
        nodes = set(i for i,_ in arcs) | set(j for _,j in arcs)
        subtours = []
        unvisited = nodes.copy()
        
        while unvisited:
            curr = unvisited.pop()
            tour = {curr}
            while True:
                for i,j in arcs:
                    if i == curr and j in unvisited:
                        curr = j
                        tour.add(j)
                        unvisited.remove(j)
                        break
                else:
                    break
            if len(tour) > 1:
                subtours.append(tour)
                
        return subtours