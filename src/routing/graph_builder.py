"""Graph construction for routing."""

import networkx as nx
import numpy as np
from typing import Dict, Optional, Tuple, Any


def build_graph(adj_matrix: np.ndarray,
                node_positions: Optional[Dict[int, Tuple[float, float]]] = None,
                current_speeds: Optional[Dict[int, float]] = None,
                predicted_speeds: Optional[Dict[int, float]] = None,
                default_speed: float = 50.0) -> nx.DiGraph:
    """
    Build a directed graph from adjacency matrix for routing.

    Args:
        adj_matrix: 2D array where adj_matrix[i,j] is distance from i to j (0 if no edge).
        node_positions: Optional dict of node_id to (lat, lon) tuples.
        current_speeds: Optional dict of node_id to current speed.
        predicted_speeds: Optional dict of node_id to predicted speed.
        default_speed: Default speed in km/h if not provided.

    Returns:
        NetworkX DiGraph with nodes and edges.

    Example:
        >>> import numpy as np
        >>> adj = np.array([[0, 1], [0, 0]])
        >>> G = build_graph(adj)
        >>> G.number_of_edges()
        1
    """
    G = nx.DiGraph()

    n = adj_matrix.shape[0]
    for i in range(n):
        # Add node with position if available
        node_data = {}
        if node_positions and i in node_positions:
            node_data['pos'] = node_positions[i]
        G.add_node(i, **node_data)

    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] > 0:
                distance = adj_matrix[i, j]
                # Use predicted speed if available, else current, else default
                speed = (predicted_speeds.get(j, None) if predicted_speeds else None) or \
                        (current_speeds.get(j, None) if current_speeds else None) or \
                        default_speed
                travel_time_min = (distance / speed) * 60  # assuming km and km/h

                edge_data = {
                    'distance': distance,
                    'current_speed': current_speeds.get(j, default_speed) if current_speeds else default_speed,
                    'predicted_speed': predicted_speeds.get(j, default_speed) if predicted_speeds else default_speed,
                    'travel_time_min': travel_time_min
                }
                G.add_edge(i, j, **edge_data)

    return G


def update_edge_weights(G: nx.DiGraph, predicted_speeds: Dict[int, float]) -> None:
    """
    Update edge weights in the graph using new predicted speeds.

    Args:
        G: The graph to update.
        predicted_speeds: Dict of node_id to new predicted speed.
    """
    for u, v, data in G.edges(data=True):
        if v in predicted_speeds:
            data['predicted_speed'] = predicted_speeds[v]
            data['travel_time_min'] = (data['distance'] / data['predicted_speed']) * 60