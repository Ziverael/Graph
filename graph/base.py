import logging
from pydantic import BaseModel, Field
from typing import NewType, cast
from pathlib import Path
import heapq  # I hate this implementation

VertexName = NewType("VertexName", str)

UNDEFINED_VERTEX = cast(VertexName, "UNDEFINED")
INFINITY = float("inf")

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

    def save_graph(self, file_name: str):
        with Path(file_name).open("w") as f:
            f.write(self.as_dot())

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
