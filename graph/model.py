import numpy as np
import random
from graph.base import (
    Graph,
    Vertex,
    VertexName,
    GraphError,
    FlexibleGraph,
)
from graph.utils import get_unique_combinations, is_link_in_ring_lattice


def _init_connected_graph(graph: Graph, n: int) -> Graph:
    for i in range(n):
        graph.add_vertex(Vertex(name=VertexName(f"{i}")))
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(
                Vertex(name=VertexName(f"{i}")), Vertex(name=VertexName(f"{j}"))
            )
    return graph


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


def _init_ring_lattice_graph(graph: FlexibleGraph, n: int, k: int) -> FlexibleGraph:
    """Init ring lattice graph.
    n - number of vertices
    k - mean degree (even int)
    """
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


class LatticeRingGraph(FlexibleGraph):
    def __init__(self, n: int, k: int):
        super().__init__()
        _init_ring_lattice_graph(self, n, k)


def _init_watts_strogatz_graph(
    graph: FlexibleGraph, n: int, k: int, beta: float
) -> FlexibleGraph:
    """Init Watts-Strogatz graph.
    n - number of vertices
    k - mean degree (even int)
    beta such that 0 <= beta <= 1
    """
    _init_ring_lattice_graph(graph, n, k)
    old_edges = graph.edges
    for edge in old_edges:
        if np.random.random() <= beta:
            graph.remove_edge(edge)
            new_vertices_names = random.sample(graph.vertices, k=2)
            while (
                graph.get_edge((new_vertices_names[0], new_vertices_names[1]))
                is not None
            ):
                new_vertices_names = random.sample(graph.vertices, k=2)
            graph.add_edge(
                vertex_from=new_vertices_names[0],
                vertex_to=new_vertices_names[1],
            )
    return graph


class WattStrogatzGraph(FlexibleGraph):
    def __init__(self, n: int, k: int, beta: float):
        super().__init__()
        _init_watts_strogatz_graph(self, n=n, k=k, beta=beta)


def _init_barabasi_albert_graph(graph: Graph, n0: int, n: int, m: int) -> Graph:
    """Init Barabasi-Albert graph.
    n0 - number of initial vertices
    n - number of vertices
    m - number of neighbors to connect to
    """
    if n <= m:
        msg = f"Initial number of nodes ({n0}) must be greater than sampling size ({m})"
        raise GraphError(msg)
    elif n0 <= m:
        msg = f"Initial number of nodes ({n0}) must be greater than sampling size ({m})"
        raise GraphError(msg)
    elif n <= n0:
        msg = f"Initial number of nodes ({n0}) must be less than final number of nodes ({n})"
        raise GraphError(msg)
    graph = _init_connected_graph(graph, n0)

    total_vertices = n0

    while total_vertices < n:
        selected_vertices_idx = np.random.choice(
            np.arange(len(graph.vertices)), size=m, replace=False
        )
        degrees = np.array([graph.get_degree(v) for v in graph.vertices])
        probs = degrees / np.sum(degrees)
        edge_exists = np.random.random(size=m) < probs[selected_vertices_idx]

        graph.add_vertex(
            Vertex(name=VertexName(str(total_vertices := total_vertices + 1)))
        )
        for idx, flag in zip(selected_vertices_idx, edge_exists):
            if flag:
                graph.add_edge(
                    vertex_from=graph.vertices[idx],
                    vertex_to=graph.vertices[total_vertices - 1],
                )
    return graph


class BarabasiAlbertGraph(Graph):
    def __init__(self, n0: int, n: int, m: int):
        super().__init__()
        _init_barabasi_albert_graph(self, n0=n0, n=n, m=m)
