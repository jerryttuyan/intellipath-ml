"""A* routing implementation for traffic-aware pathfinding."""

import heapq
import math
from typing import List, Tuple, Optional, Dict, Any
import networkx as nx


def euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def a_star_routing(G: nx.DiGraph, start: int, goal: int,
                   weight: str = 'travel_time_min') -> Tuple[Optional[List[int]], Optional[float]]:
    """
    Find the shortest path using A* algorithm.

    Args:
        G: NetworkX DiGraph with edge weights.
        start: Starting node ID.
        goal: Goal node ID.
        weight: Edge attribute to use as weight (default 'travel_time_min').

    Returns:
        Tuple of (path as list of nodes, total cost) or (None, None) if no path.
    """
    if start not in G or goal not in G:
        return None, None

    # Priority queue: (f_score, node)
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from: Dict[int, int] = {}
    g_score: Dict[int, float] = {node: float('inf') for node in G.nodes}
    g_score[start] = 0
    f_score: Dict[int, float] = {node: float('inf') for node in G.nodes}
    f_score[start] = heuristic(G, start, goal)

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[goal]

        for neighbor in G.neighbors(current):
            edge_data = G.get_edge_data(current, neighbor)
            if weight not in edge_data:
                continue  # Skip edges without the weight
            tentative_g = g_score[current] + edge_data[weight]

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(G, neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, None  # No path found


def heuristic(G: nx.DiGraph, node: int, goal: int) -> float:
    """
    Calculate heuristic for A* (Euclidean distance if positions available).

    Args:
        G: The graph.
        node: Current node.
        goal: Goal node.

    Returns:
        Heuristic value (0 if no positions).
    """
    if 'pos' not in G.nodes[node] or 'pos' not in G.nodes[goal]:
        return 0  # No heuristic
    pos_node = G.nodes[node]['pos']
    pos_goal = G.nodes[goal]['pos']
    return euclidean_distance(pos_node, pos_goal)
