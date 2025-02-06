import numpy as np
from drp import DRP
from environment.state import State
from environment.action import Action, ActionType
from environment.cost import Cost
from learning.policy import Policy
from learning.value_network import ValueNetwork
from optimization.milp_solver import MILPSolver

def solve_instance(instance_num: int, excel_path: str):
    # Initialize components
    drp = DRP(excel_path)
    state = State(drp, instance_num)
    cost_function = Cost()
    value_network = ValueNetwork(num_nodes=drp.num_nodes, num_stations=drp.num_stations)
    milp_solver = MILPSolver(time_limit=300) 
    
    # Results tracking
    total_cost = 0
    route = [state.current_node]
    
    while not state.is_terminal():
        try:
            action = milp_solver.solve_subproblem(state, value_network)
            if action is None:
                raise ValueError("MILP solver returned no valid action")
                
            # Debug info
            print(f"State: Node {state.current_node}, Battery: {state.battery_level:.2f}")
            print(f"Action: {action} (cost: {cost_function(state, action):.2f})")
            
            # Execute action
            action.execute(state)
            total_cost += cost_function(state, action)
            route.append(state.current_node)
            
        except ValueError as e:
            print(f"Error: {e}")
            break
            
    return total_cost, route

def main():
    # Configuration
    excel_path = r"C:\Users\issam\Desktop\Files\PhD\DRPRL\data\StaticProblemInstances\GroupA\distancematrix_with_20_nodes(seconds).xlsx"
    
    # Solve instance 1
    print(f"\nSolving Instance D1:")
    print("-" * 50)
    
    total_cost, route = solve_instance(1, excel_path)
    
    print(f"\nInstance D1 Results:")
    print(f"Total Cost: {total_cost:.2f}")
    print(f"Route: {route}")
    print("=" * 50)

if __name__ == "__main__":
    main()
