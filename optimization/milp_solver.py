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
        """Solve MILP subproblem to find next best action"""
        # Create model
        model = gp.Model("DRP_Subproblem")
        model.setParam('TimeLimit', self.time_limit)
        model.setParam('LazyConstraints', 1)

        # Setup nodes
        unvisited = {i for i in range(state.drp.num_stations, 
                                    state.drp.num_nodes + state.drp.num_stations) 
                    if state.encoding[i] == 1}
        stations = set(range(state.drp.num_stations))

        # Variables
        x = {}
        for i in {state.current_node} | unvisited | stations:
            for j in unvisited | stations:
                if i != j:
                    x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')

        # Objective with future cost estimation
        obj = 0
        
        # First term: ∑∑ T_ij x_ij
        immediate_cost = gp.quicksum(
            state.drp.get_travel_time(
                state.instance,
                state._to_matrix_index(i),
                state._to_matrix_index(j)
            ) * x[i,j] 
            for i,j in x
        )
        
        # Second term: V̂(s')
        future_costs = 0
        for j in unvisited | stations:
            if (state.current_node, j) in x:
                next_state = state.__class__(state.drp, state.instance, j)
                action = Action(
                    ActionType.RECHARGE if j < state.drp.num_stations else ActionType.MOVE, 
                    j
                )
                action.execute(next_state)
                future_costs += value_network.predict(next_state) * x[state.current_node, j]

        # Complete objective
        obj = immediate_cost + future_costs
        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints
        # 1. Flow conservation
        for h in unvisited:
            model.addConstr(
                gp.quicksum(x[i, h] for i in {state.current_node} | unvisited | stations if (i, h) in x) ==
                gp.quicksum(x[h, j] for j in unvisited | stations if (h, j) in x)
            )

        # 2. Visit each unvisited node exactly once
        for j in unvisited:
            model.addConstr(
                gp.quicksum(x[i, j] for i in {state.current_node} | unvisited | stations if (i, j) in x) == 1
            )

        # 3. Battery constraints
        for i, j in x:
            travel_time = state.drp.get_travel_time(
                state.instance,
                state._to_matrix_index(i),
                state._to_matrix_index(j)
            )
            if travel_time > state.drp.battery_capacity:
                model.addConstr(x[i, j] == 0)

        # 4. No direct routes between charging stations
        for i in stations:
            for j in stations:
                if (i, j) in x:
                    model.addConstr(x[i, j] == 0)

        # Lazy constraints callback for subtour elimination
        def subtour_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                # Get solution values
                vals = model.cbGetSolution(model._x)
                selected = [(i, j) for (i, j) in model._x.keys() if vals[i, j] > 0.5]

                # Find subtours
                subtours = self._find_subtours(selected)

                # Add lazy constraints
                for tour in subtours:
                    if len(tour) < len(unvisited) + len(stations):
                        model.cbLazy(
                            gp.quicksum(x[i, j] for i in tour for j in tour if (i, j) in x)
                            <= len(tour) - 1
                        )

        # Set callback data
        model._x = x

        # Single optimization
        model.optimize(subtour_callback)

        # Extract solution
        if model.status == GRB.OPTIMAL:
            best_action = None
            for i, j in x:
                if i == state.current_node and x[i, j].X > 0.5 :
                    best_action = Action(
                        ActionType.RECHARGE, j
                    ) if j < state.drp.num_stations else Action(
                        ActionType.MOVE, j
                    )
                    break
            if best_action is None:
                print("No valid action found")
                return None                  

            return best_action
        else:
            print("No optimal solution found")
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