import matplotlib.pyplot as plt
import numpy as np

def get_circular_positions(n, radius, center=(0.5, 0.5), start_angle=0):
    angles = np.linspace(start_angle, start_angle + 2*np.pi, n, endpoint=False)
    positions = np.zeros((n, 2))
    positions[:, 0] = center[0] + radius * np.cos(angles)
    positions[:, 1] = center[1] + radius * np.sin(angles)
    return positions

def visualize_solution(route, state, cost):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_facecolor('aliceblue')
    
    # Calculate positions
    station_positions = get_circular_positions(
        state.drp.num_stations, 
        radius=0.3,  # Inner circle for stations
        start_angle=np.pi/2
    )
    
    customer_positions = get_circular_positions(
        state.drp.num_nodes,
        radius=0.7,  # Outer circle for customers
        start_angle=np.pi/2
    )
    
    # Combine positions
    positions = np.vstack([station_positions, customer_positions])
    
    # Plot stations (red squares)
    for i in range(state.drp.num_stations):
        plt.scatter(positions[i,0], positions[i,1], 
                   c='red', s=200, marker='s', 
                   label='Charging Station' if i==0 else "")
        plt.annotate(f'S{i}', (positions[i,0], positions[i,1]))
    
    # Plot customers (blue circles)
    for i in range(state.drp.num_stations, len(positions)):
        plt.scatter(positions[i,0], positions[i,1], 
                   c='blue', s=100, 
                   label='Customer' if i==state.drp.num_stations else "")
        plt.annotate(f'C{i-state.drp.num_stations}', 
                    (positions[i,0], positions[i,1]))
    
    # Plot routes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(route)-1))
    for idx in range(len(route)-1):
        i, j = route[idx], route[idx+1]
        plt.arrow(positions[i,0], positions[i,1],
                 positions[j,0]-positions[i,0], 
                 positions[j,1]-positions[i,1],
                 head_width=0.02, color=colors[idx], 
                 length_includes_head=True,
                 label=f'Step {idx+1}')
    
    plt.title(f'Drone Route Visualization\nTotal Cost: {cost:.1f} seconds')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()