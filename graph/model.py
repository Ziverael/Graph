import numpy as np
from graph.base import (
    Graph,
    Vertex,
    VertexName,
    GraphError,
    FlexibleGraph,
    UndirectedEdge,
)
from graph.utils import get_unique_combinations, is_link_in_ring_lattice


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
        pairs.update(((i, j) for i, j in new_pairs))
    for i, j in pairs:
        graph.add_edge(
            Vertex(name=VertexName(f"{i}")),
            Vertex(name=VertexName(f"{j}")),
        )
    return graph


class RandomGraph(Graph):
    def __init__(self, n: int, m: int | None = None, p: float | None = None):
        super().__init__()
        if m is not None and p is None:
            _init_gilbert_graph(self, n=n, m=m)
        elif p is not None and m is None:
            _init_erdos_renyi_graph(self, n=n, p=p)
        else:
            msg = "Init arguments do not match any model"
            raise GraphError(msg)


def _init_ring_lattice_graph(n: int, k: int) -> FlexibleGraph:
    """Init ring lattice graph.
    n - number of vertices
    k - mean degree (even int)
    """
    graph = FlexibleGraph()
    if k % 2 == 1:
        msg = "k must be even"
        raise GraphError(msg)

    graph.add_vertices_from_list(
        list(Vertex(name=VertexName(f"{i}")) for i in range(1, n + 1))
    )
    for i, j in get_unique_combinations(np.arange(1, n + 1), np.arange(1, n + 1)):
        if is_link_in_ring_lattice(i, j, k, n):
            graph.add_edge(
                Vertex(name=VertexName(f"{i}")), Vertex(name=VertexName(f"{j}"))
            )
    return graph


def _init_watts_strogatz_graph(n: int, k: int, beta: float) -> FlexibleGraph:
    """Init Watts-Strogatz graph.
    n - number of vertices
    k - mean degree (even int)
    beta such that 0 <= beta <= 1
    """
    graph = _init_ring_lattice_graph(n, k)
    new_edges = []
    for edge in graph.edges:
        if np.random.random() <= beta:
            graph.remove_edge(edge)
            new_edges.append(
                UndirectedEdge(
                    vertex1=np.random.choice(graph.vertices),
                    vertex2=np.random.choice(graph.vertices),
                )
            )
    return graph


class WattStrogatzGraph(Graph):
    def __init__(self, n: int, k: int, beta: float):
        super().__init__()
        _init_watts_strogatz_graph(self, n=n, k=k, beta=beta)


class BarabasiAlbertGraph(Graph):
    pass
