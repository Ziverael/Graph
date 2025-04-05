import logging
from pydantic import BaseModel, Field
from typing import NewType, cast, Literal
from pathlib import Path
import heapq  # I hate this implementation
import numpy as np

VertexName = NewType("VertexName", str)
FileFormat = Literal["dot", "csv"]

UNDEFINED_VERTEX = cast(VertexName, "UNDEFINED")
INFINITY = float("inf")
UNKNOWN_MEAN_DEGREE = -1
logger = logging.getLogger(__name__)


class GraphError(Exception): ...


class Vertex(BaseModel):
    name: VertexName


class UndirectedEdge(BaseModel):
    vertex1: Vertex
    vertex2: Vertex
    weight: float = Field(default=1.0, ge=0, le=1.0)

    def __contains__(self, vertex: Vertex) -> bool:
        return self.vertex1 == vertex or self.vertex2 == vertex


def _does_vertex_exist(vertex: Vertex, vertices: dict[VertexName, Vertex]) -> bool:
    return vertex.name in vertices


class Graph:
    def __init__(self):
        self._vertices: dict[VertexName, Vertex] = {}
        self._edges: list[UndirectedEdge] = []
        self._mean_degree: float = UNKNOWN_MEAN_DEGREE

    def add_vertex(self, vertex: Vertex):
        if _does_vertex_exist(vertex, self._vertices):
            msg = f"Vertex {vertex.name} already exists"
            raise GraphError(msg)
        self._vertices[vertex.name] = vertex

    def add_vertices_from_list(self, vertices: list[Vertex]):
        for vertex in vertices:
            try:
                self.add_vertex(vertex)
            except GraphError:
                logger.warning(
                    "Cannot add vertex %s to graph. This vertex will be ignored.",
                    vertex.name,
                )

    def add_edge(self, vertex_from: Vertex, vertex_to: Vertex, weight: float = 1.0):
        for vertex in [vertex_from, vertex_to]:
            if not _does_vertex_exist(vertex, self._vertices):
                logger.warning(
                    "Vertex %s does not exist. Adding it to the graph.", vertex.name
                )
                self.add_vertex(vertex)
        self._edges.append(
            UndirectedEdge(vertex1=vertex_from, vertex2=vertex_to, weight=weight)
        )

    def add_edges_from_list(
        self, edges: list[tuple[Vertex, Vertex, float]] | list[tuple[Vertex, Vertex]]
    ):
        for edge in edges:
            try:
                self.add_edge(*edge)
            except Exception:
                logger.warning(
                    "Cannot add edge %s to graph. This edge will be ignored.", edge
                )

    def get_neighbors(self, vertex_name: VertexName) -> list[Vertex]:
        return [
            edge.vertex1 if edge.vertex2.name == vertex_name else edge.vertex2
            for edge in self._edges
            if Vertex(name=vertex_name) in edge
        ]

    def get_strangers(self, vertex_name: VertexName) -> list[Vertex]:
        excluded = [*self.get_neighbors(vertex_name), Vertex(name=vertex_name)]
        return [vertex for vertex in self._vertices.values() if vertex not in excluded]

    def save_graph(self, file_name: str, format_: FileFormat = "dot"):
        if format_ == "dot":
            with Path(file_name).open("w") as f:
                f.write(self.as_dot())
        elif format_ == "csv":
            with Path(file_name).open("w") as f:
                f.write(self.as_csv())

    def as_dot(self) -> str:
        dot_representation = "graph G {"
        vertices_names_in_edges = set()
        for edge in self._edges:
            dot_representation += f'"{edge.vertex1.name}" -- "{edge.vertex2.name}" [weight={edge.weight}];'
            vertices_names_in_edges.add(edge.vertex1.name)
            vertices_names_in_edges.add(edge.vertex2.name)
        for vertex in self._vertices.values():
            if vertex.name not in vertices_names_in_edges:
                dot_representation += f'"{vertex.name}";'
        dot_representation += "}"
        return dot_representation

    def get_shortest_paths(self, vertex_from: Vertex) -> dict[VertexName, float]:
        graph_vertices_names = set(self._vertices.keys())
        vertices_distances_queue: list[tuple[float, VertexName]] = []
        distances: dict[VertexName, float] = {}
        prev: dict[VertexName, VertexName] = {}

        distances[vertex_from.name] = 0
        heapq.heappush(vertices_distances_queue, (0, vertex_from.name))
        graph_vertices_names.remove(vertex_from.name)

        for v_name in graph_vertices_names:
            distances[v_name] = INFINITY
            prev[v_name] = UNDEFINED_VERTEX
            heapq.heappush(vertices_distances_queue, (INFINITY, v_name))

        while vertices_distances_queue:
            _, current_vertex_name = heapq.heappop(vertices_distances_queue)
            for neighbor_vertex in self.get_neighbors(current_vertex_name):
                if (
                    current_path_weight := distances[current_vertex_name]
                    + cast(
                        UndirectedEdge,
                        self.get_edge(
                            (
                                Vertex(name=current_vertex_name),
                                Vertex(name=neighbor_vertex.name),
                            )
                        ),
                    ).weight
                ) < distances[neighbor_vertex.name]:
                    distances[neighbor_vertex.name] = current_path_weight
                    prev[neighbor_vertex.name] = current_vertex_name
                    # This is not optimal, because vertices paths length are not updated,
                    # because heapq do not deliver decrease_priority method.
                    heapq.heappush(
                        vertices_distances_queue,
                        (current_path_weight, neighbor_vertex.name),
                    )
        return distances

    def get_edge(self, vertices: tuple[Vertex, Vertex]) -> UndirectedEdge | None:
        for edge in self._edges:
            if vertices[0] in edge and vertices[1] in edge:
                return edge
        return None

    def as_csv(self) -> str:
        """Gephi tool csv. Basically it does not support weighted edges and base on edges only."""
        return "\n".join(
            [f"{edge.vertex1.name},{edge.vertex2.name}" for edge in self._edges]
        )

    @property
    def is_undirected(self) -> bool:
        return True

    @property
    def is_weighted(self) -> bool:
        return all([edge.weight == 1.0 for edge in self._edges])

    @property
    def vertices(self) -> list[Vertex]:
        return list(self._vertices.values())

    @property
    def edges(self) -> list[UndirectedEdge]:
        return self._edges

    def __contains__(self, vertex: Vertex) -> bool:
        return _does_vertex_exist(vertex, self._vertices)

    @property
    def mean_degree(self) -> float:
        if self._mean_degree == UNKNOWN_MEAN_DEGREE:
            if len(self._vertices) == 0:
                self._mean_degree = float("nan")
            else:
                self._mean_degree = sum(
                    len(self.get_neighbors(v.name)) for v in self._vertices.values()
                ) / len(self._vertices)
        return self._mean_degree


class FlexibleGraph(Graph):
    def remove_edge(self, edge: UndirectedEdge) -> None:
        if edge not in self._edges:
            msg = f"Edge {edge.vertex1.name} -- {edge.vertex2.name} not found"
            raise GraphError(msg)
        self._edges.remove(edge)

    def get_random_neighbor(self, vertex: Vertex) -> Vertex:
        return np.random.choice(self.get_neighbors(vertex.name))

    def get_random_stranger(self, vertex: Vertex) -> Vertex:
        return np.random.choice(self.get_strangers(vertex.name))

    def randomly_replace_edge(
        self, edge: UndirectedEdge, static_vertex: Vertex
    ) -> None:
        if edge not in self._edges or static_vertex not in edge:
            msg = f"Cannot replace edge ({edge.vertex1.name}, {edge.vertex2.name}) with static vertex {static_vertex.name}"
            raise GraphError(msg)
        new_vertex = self.get_random_neighbor(static_vertex)
        self._edges.remove(edge)
        self.add_edge(static_vertex, new_vertex)
