import numpy as np
from drp import DRP
from environment.state import State
from environment.action import Action, ActionType
from environment.cost import Cost
from learning.value_network import ValueNetwork
from optimization.milp_solver import MILPSolver
from visualization import visualize_solution
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
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
    
    # Progress tracking
    table = Table(title=f"Instance D{instance_num} Solution Progress")
    table.add_column("Step", justify="center")
    table.add_column("From", justify="center")
    table.add_column("To", justify="center")
    table.add_column("Action", justify="left")
    table.add_column("Battery", justify="right")
    table.add_column("Step Cost", justify="right")
    
    total_cost = 0
    route = [state.current_node]
    step = 0
    
    # Solve instance
    while not state.is_terminal():
        try:
            step += 1
            from_node = state.current_node
            
            action = milp_solver.solve_subproblem(state, value_network)
            if action is None:
                raise ValueError("No valid action found")
            
            cost = cost_function(state, action)
            action.execute(state)
            to_node = state.current_node
            
            # Update tracking
            table.add_row(
                str(step),
                str(from_node),
                str(to_node),
                str(action.action_type.value),
                f"{state.battery_level:.1f}",
                f"{cost:.1f}"
            )
            
            total_cost += cost
            route.append(to_node)
            logging.info(f"Step {step}: {from_node}->{to_node} ({action.action_type.value}, cost={cost:.1f})")
            
        except ValueError as e:
            console.print(Panel(f"[red]Error: {e}[/red]"))
            logging.error(str(e))
            break
    
    # Calculate matrix-based cost
    matrix_cost = sum(state.drp.get_travel_time(instance_num, 
                                               state._to_matrix_index(route[i]),
                                               state._to_matrix_index(route[i+1]))
                     for i in range(len(route)-1))
    
    # Display results
    console.print("\n")
    console.print(table)
    console.print(Panel.fit(
        f"[bold]Solution Summary[/bold]\n"
        f"Total Steps: {len(route)-1}\n"
        f"Route: {' → '.join(map(str, route))}\n"
        f"Total Time: {matrix_cost:.1f} seconds ({matrix_cost/60:.1f} minutes)"
    ))
    
    return matrix_cost, route

def main():
    excel_path = r"C:\Users\issam\Desktop\Files\PhD\DRPRL\data\StaticProblemInstances\GroupA\distancematrix_with_20_nodes(seconds).xlsx"
    instance_num = 1
    
    try:
        cost, route = solve_instance(instance_num, excel_path)
        # Add visualization
        visualize_solution(route, State(DRP(excel_path), instance_num), cost)
    except Exception as e:
        console.print(Panel(f"[red]Error solving instance: {str(e)}[/red]"))

if __name__ == "__main__":
    main()
