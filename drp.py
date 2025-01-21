import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from itertools import permutations

class DRP:
    def __init__(self, distance_matrix):
        """
        Initializes the Drone Routing Problem instance.
        Args:
            distance_matrix (np.ndarray): Distance matrix representing nodes.
        """
        self.distance_matrix = distance_matrix
        self.num_nodes = self.distance_matrix.shape[0]
        self.nodes_positions = None
        self.groups = None
        self.recharging_stations = None

        self.nodes_positions = self.extract_nodes_position()
        self.groups, self.recharging_stations = self.cluster_nodes()

    def extract_nodes_position(self, num_components=2):
        """
        Generates 2D coordinates for the nodes based on the distance matrix
        using multidimensional scaling (MDS).
        Args:
            n_components (int): Number of dimensions for the positions (x,y) --> 2.
        """
        mds = MDS(n_components=num_components, dissimilarity='precomputed',random_state=42)
        self.nodes_positions = mds.fit_transform(self.distance_matrix)
        return self.nodes_positions
    
    def cluster_nodes(self, num_clusters=5) :
        """
        Clusters nodes into groups and selects recharging stations as cluster centers.
        Args:
            num_clusters (int): Number of clusters (default is 5).
        """
        if self.nodes_positions is None:
            raise ValueError("Nodes positions not extracted. Call extract_nodes_position() first.")
        kmeans = KMeans(n_clusters=num_clusters,init="k-means++", random_state=42)
        self.groups = kmeans.fit_predict(self.nodes_positions)
        self.recharging_stations = kmeans.cluster_centers_
        return self.groups, self.recharging_stations
    
    def __repr__(self):
        """
        String representation of the Drone Routing Problem instance.
        """
        return (
            f"Drone Routing Problem with {self.num_nodes} nodes.\n"
            f"Nodes positions: {self.nodes_positions}\n"
            f"Recharging stations: {self.recharging_stations}\n"
        )
        

