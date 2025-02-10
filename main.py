import numpy as np
from drp import DRP
from environment.state import State
from environment.action import Action, ActionType
from environment.cost import Cost
from learning.policy import Policy
from learning.value_network import ValueNetwork
from optimization.milp_solver import MILPSolver
from visualization import visualize_solution
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import track
from datetime import datetime

console = Console()

def solve_instance(instance_num: int, excel_path: str):
    # Setup logging
    log_filename = f"drp_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    
    # Initialize components
    drp = DRP(excel_path)
    state = State(drp, instance_num)
    cost_function = Cost()
    value_network = ValueNetwork(num_nodes=drp.num_nodes, num_stations=drp.num_stations)
    milp_solver = MILPSolver(time_limit=300)
    
    # Create results table
    table = Table(title=f"Instance D{instance_num} Progress")
    table.add_column("Step")
    table.add_column("Current Node")
    table.add_column("Action")
    table.add_column("Battery Level")
    table.add_column("Cost")
    
    total_cost = 0
    route = [state.current_node]
    step = 0
    
    while not state.is_terminal():
        try:
            step += 1
            
            # Log state
            console.print(f"\n[bold cyan]Step {step}[/bold cyan]")
            console.print(f"Current position: [green]Node {state.current_node}[/green]")
            console.print(f"Battery level: [yellow]{state.battery_level:.2f}[/yellow]")
            
            action = milp_solver.solve_subproblem(state, value_network)
            if action is None:
                raise ValueError("No valid action found")
            
            cost = cost_function(state, action)
            action.execute(state)
            total_cost += cost
            route.append(state.current_node)
            
            # Update table
            table.add_row(
                str(step),
                str(state.current_node),
                str(action),
                f"{state.battery_level:.2f}",
                f"{cost:.2f}"
            )
            
            # Log to file
            logging.info(f"Step {step}: Node {state.current_node} -> {action} (Cost: {cost:.2f})")
            
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            logging.error(str(e))
            break
    
    # Display final table
    console.print(table)
    console.print(f"\n[bold green]Total Cost: {total_cost:.2f}[/bold green]")
    
    return total_cost, route

def main():
    # 1. Configuration
    excel_path = r"C:\Users\issam\Desktop\Files\PhD\DRPRL\data\StaticProblemInstances\GroupA\distancematrix_with_20_nodes(seconds).xlsx"
    instance_num = 1  # Start with D1
    
    print(f"\nSolving Instance D{instance_num}:")
    print("-" * 50)
    
    # 2. Initialize and solve
    try:
        total_cost, route = solve_instance(instance_num, excel_path)
        
        # 3. Print detailed results
        print("\nResults:")
        print("-" * 50)
        print("Route:")
        for i in range(len(route)-1):
            from_node = route[i]
            to_node = route[i+1]
            print(f"Step {i+1}: {from_node} -> {to_node}")
        
        print("\nSummary:")
        print(f"Total nodes visited: {len(route)-1}")
        print(f"Total cost (seconds): {total_cost:.2f}")
        #print(f"Total cost (minutes): {total_cost/60:.2f}")
        print("-" * 50)
        
        # 4. Visualize
        #visualize_solution(route, State(DRP(excel_path), instance_num), total_cost)
        
    except Exception as e:
        print(f"Error solving instance: {str(e)}")

if __name__ == "__main__":
    main()
