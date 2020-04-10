"""Shortest paths and path lengths using the Beam Stacked Search algorithm.
"""
from heapq import heappush, heappop
from itertools import count

import networkx as nx
from networkx.utils import not_implemented_for
from networkx.algorithms.shortest_paths.weighted import _weight_function

__all__ = ['beam_path']


def beam_path(G, source, target, heuristic=None, weight='weight'):
    """Returns a list of nodes in a shortest path between source and target
    using the Beam Stacked Search (without 'anytime').

    There may be more than one shortest path.  This returns only one.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> print(nx.astar_path(G, 0, 4))
    [0, 1, 2, 3, 4]
    >>> G = nx.grid_graph(dim=[3, 3])  # nodes are two-tuples (x,y)
    >>> nx.set_edge_attributes(G, {e: e[1][0]*2 for e in G.edges()}, 'cost')
    >>> def dist(a, b):
    ...    (x1, y1) = a
    ...    (x2, y2) = b
    ...    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    >>> print(nx.astar_path(G, (0, 0), (2, 2), heuristic=dist, weight='cost'))
    [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]


    See Also
    --------
    shortest_path, dijkstra_path

    """
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)


    #It is defined a specific width
    width = 2
   
    #This function gives us the "best" two neighbours of a node "v" by using the function "relative_distance"
    def successors(v):
        return iter(sorted(G.neighbors(v), key = relative_distance, reverse = True) [:width])

    for e in generic_dfs_edges(G, source, successors(source)):
       yield e 

    
    def relative_distance(source, node):
        return dist(source, node)

    #This is the DFS function that returns a path if it exists
    def generic_dfs_edges(G, source, neighbors = None, depth_limit = None):

        visited ={source}

        if depth_limit is None:
            depth_limit = len(G)

        queue = deque([(source, depth_limit, neighbors(source))])

        while queue:
            
            parent, depth_now, children = queue[-1]

            try:
                child = next(children)

                if child == target:
                    path = [child]
                    node = parent
                    
                    while node is not None:
                        path.append(node)
                        node = visited[node]
                    path.reverse()
                    return path

                if child not in visited:
                    yield parent, child
                    visited.add(child)
                    
                    if depth_now > 1:
                        queue.append((child, depth_now - 1, neighbors(child)))
           
            except StackIteration:
                queue.pop()
        
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

 
