import numpy as np
from graph.base import Graph, Vertex, VertexName


def _init_erdos_renyi_graph(graph: Graph, n: int, p: float) -> Graph:
    """Init Erdos-Renyi graph.
    n - number of vertices
    p - probability of edge between any two vertices
    """
    for i in range(n):
        graph.add_vertex(Vertex(name=VertexName(f"{i}")))
    for i in range(n):
        edge_exists = np.random.random(size=n - i - 1) < p
        for j in range(i + 1, n):
            if edge_exists[j - i - 1]:
                graph.add_edge(
                    Vertex(name=VertexName(f"{i}")), Vertex(name=VertexName(f"{j}"))
                )
    return graph


def _init_gilbert_graph(graph: Graph, n: int, m: int) -> Graph:
    """Init Gilbert graph.
    n - number of vertices
    m - number of edges
    """
    for i in range(n):
        graph.add_vertex(Vertex(name=VertexName(f"{i}")))

    pairs: set[tuple[int, int]] = set()
    while len(pairs) < m:
        new_pairs = np.random.randint(0, n, size=(m - len(pairs), 2))
        new_pairs = new_pairs[new_pairs[:, 0] < new_pairs[:, 1]]
        pairs.update(map(tuple, new_pairs))
    for i, j in pairs:
        graph.add_edge(
            Vertex(name=VertexName(f"{i}")),
            Vertex(name=VertexName(f"{j}")),
        )
    return graph


class RandomGraph(Graph):
    def __init__(self, n: int, p: float):
        super().__init__()
        for i in range(n):
            self.add_vertex(Vertex(name=VertexName(f"{i}")))
        for i in range(n):
            edge_exists = np.random.random(size=n - i - 1) < p
            for j in range(i + 1, n):
                if edge_exists[j - i - 1]:
                    self.add_edge(
                        Vertex(name=VertexName(f"{i}")), Vertex(name=VertexName(f"{j}"))
                    )


class WattStrogatzGraph(Graph):
    pass


class BarabasiAlbertGraph(Graph):
    pass
