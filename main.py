import numpy as np
import pandas as pd
from drp import DRP
from environment.state import State
from environment.action import Action, ActionType
from environment.cost import Cost
from learning.value_network import ValueNetwork
from optimization.milp_solver import MILPSolver
from visualization import visualize_solution
from datetime import datetime

def solve_instance(instance_num: int, excel_path: str):
    """Solve a single instance of the problem."""
    
    print(f"\n=== Solving Instance D{instance_num} ===\n")

    # Initialize components
    drp = DRP(excel_path, instance_num)
    state = State(drp, instance_num)
    cost_function = Cost()
    value_network = ValueNetwork(num_nodes=drp.num_customers, num_stations=drp.num_stations)
    milp_solver = MILPSolver(time_limit=300)
    
    total_cost = 0
    route = [state.current_node]
    step = 0

    print("Step | From | To  | Action     | Battery  | Step Cost")
    print("------------------------------------------------------")

    while not state.is_terminal():
        try:
            step += 1
            from_node = state.current_node

            action = milp_solver.solve_subproblem(state, value_network)
            if action is None:
                print("No valid action found. Exiting.")
                break

            cost = cost_function(state, action)
            action.execute(state)
            to_node = state.current_node

            # Print step information
            print(f"{step:>3}  | {from_node:>4} | {to_node:>3} | {action.action_type.value:<10} | {state.battery_level:>7.1f} | {cost:>9.1f}")

            total_cost += cost
            route.append(to_node)

        except ValueError as e:
            print(f"Error: {e}")
            break

    # Calculate final cost using travel time matrix
    matrix_cost = sum(
        state.drp.get_travel_time(
            instance_num,
            int(state._to_matrix_index(route[i])),
            int(state._to_matrix_index(route[i+1]))
        )
        for i in range(len(route)-1)
    )

    print("\n=== Solution Summary ===")
    print(f"Total Steps: {len(route)-1}")
    print(f"Route: {' → '.join(map(str, route))}")
    print(f"Total Time: {matrix_cost:.1f} seconds ({matrix_cost/60:.1f} minutes)\n")

    return matrix_cost, route

def main():
    excel_path = r"C:\Users\issam\Desktop\Files\PhD\DRPRL\data\StaticProblemInstances\GroupA\distancematrix_with_20_nodes(seconds).xlsx"
    
    for instance_num in range(1, 21):  # Iterate over 20 instances
        try:
            cost, route = solve_instance(instance_num, excel_path)
            visualize_solution(route, State(DRP(excel_path), instance_num), cost)
        except Exception as e:
            print(f"Error solving instance {instance_num}: {str(e)}")

if __name__ == "__main__":
    main()
