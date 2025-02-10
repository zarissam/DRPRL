import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def visualize_solution(route, state, total_cost):
    G = nx.DiGraph()
    
    # Add nodes
    pos = nx.spring_layout(G)
    for i in range(state.drp.num_stations + state.drp.num_nodes):
        G.add_node(i)
    
    # Add edges from route
    edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
    G.add_edges_from(edges)
    
    # Draw
    plt.figure(figsize=(12, 8))
    
    # Draw stations (red)
    stations = list(range(state.drp.num_stations))
    nx.draw_networkx_nodes(G, pos, nodelist=stations, 
                          node_color='red', node_size=500, label='Stations')
    
    # Draw customers (blue)
    customers = list(range(state.drp.num_stations, state.drp.num_nodes + state.drp.num_stations))
    nx.draw_networkx_nodes(G, pos, nodelist=customers, 
                          node_color='blue', node_size=300, label='Customers')
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, edgelist=edges, 
                          edge_color='green', arrows=True, arrowsize=20)
    
    # Labels
    labels = {i: f'S{i}' if i < state.drp.num_stations 
             else f'C{i-state.drp.num_stations}' for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels)
    
    plt.title(f'Drone Route (Total Cost: {total_cost:.2f})')
    plt.legend()
    plt.axis('off')
    plt.show()