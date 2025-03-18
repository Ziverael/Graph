from graph import base
from graph.base import Vertex, VertexName
import pytest
import logging
from typing import cast


def test_init_graph():
    # when
    g = base.Graph()

    # then
    assert g._vertices == {}
    assert g._edges == []
    assert g.is_undirected
    assert g.is_weighted


def test_add_vertex(empty_graph):
    # given
    g = empty_graph
    vertices_in_graph = ["A", "B", "C"]
    vertex_to_add = cast(VertexName, "D")
    g._vertices = {
        name: base.Vertex(name=cast(VertexName, name)) for name in vertices_in_graph
    }

    # when
    g.add_vertex(base.Vertex(name=vertex_to_add))

    # then
    assert vertex_to_add in g._vertices
    assert base.Vertex(name=vertex_to_add) in g._vertices.values()


def test_add_vertex__vertex_already_exists():
    # given
    g = base.Graph()
    vertex_to_add = cast(VertexName, "A")
    g._vertices = {vertex_to_add: base.Vertex(name=vertex_to_add)}
    expected_error_message = f"Vertex {vertex_to_add} already exists"

    # when
    with pytest.raises(base.GraphError, match=expected_error_message):
        g.add_vertex(base.Vertex(name=vertex_to_add))


@pytest.mark.parametrize(
    ("vertices_to_add", "expected_vertices_in_graph", "expected_log_message"),
    [
        pytest.param(["A", "B", "C"], ["A", "B", "C"], "", id="unique-vertices"),
        pytest.param(
            ["A", "A", "A", "C"],
            ["A", "C"],
            "Cannot add vertex A to graph. This vertex will be ignored.",
            id="duplicated-vertex",
        ),
        pytest.param([], [], "", id="empty-list"),
    ],
)
def test_add_vertices_from_list(
    empty_graph,
    vertices_to_add,
    expected_vertices_in_graph,
    expected_log_message,
    caplog: pytest.LogCaptureFixture,
):
    # given
    g = empty_graph

    # when
    with caplog.at_level(logging.WARNING):
        g.add_vertices_from_list([base.Vertex(name=name) for name in vertices_to_add])

    # then
    assert g._vertices == {
        name: base.Vertex(name=name) for name in expected_vertices_in_graph
    }
    assert expected_log_message in caplog.text


@pytest.mark.parametrize(
    ("vertices_in_graph", "edges_params", "expected_edges_in_graph"),
    [
        pytest.param(
            ["A", "B", "C"],
            [("A", "B", 1.0)],
            [
                base.UndirectedEdge(
                    vertex1=base.Vertex(name=cast(VertexName, "A")),
                    vertex2=base.Vertex(name=cast(VertexName, "B")),
                    weight=1.0,
                )
            ],
            id="simple-edge",
        ),
        pytest.param(
            ["A", "B", "C"],
            [("A", "B")],
            [
                base.UndirectedEdge(
                    vertex1=base.Vertex(name=cast(VertexName, "A")),
                    vertex2=base.Vertex(name=cast(VertexName, "B")),
                    weight=1.0,
                )
            ],
            id="not-weighted-edge",
        ),
        pytest.param(
            ["A", "B", "C"],
            [("A", "C", 0.3)],
            [
                base.UndirectedEdge(
                    vertex1=base.Vertex(name=cast(VertexName, "A")),
                    vertex2=base.Vertex(name=cast(VertexName, "C")),
                    weight=0.3,
                )
            ],
            id="weighted-edge",
        ),
        pytest.param(
            ["A", "B", "C"],
            [("A", "D", 0.3)],
            [
                base.UndirectedEdge(
                    vertex1=base.Vertex(name=cast(VertexName, "A")),
                    vertex2=base.Vertex(name=cast(VertexName, "D")),
                    weight=0.3,
                )
            ],
            id="non-existent-vertex",
        ),
        pytest.param(
            [],
            [("A", "D", 0.3)],
            [
                base.UndirectedEdge(
                    vertex1=base.Vertex(name=cast(VertexName, "A")),
                    vertex2=base.Vertex(name=cast(VertexName, "D")),
                    weight=0.3,
                )
            ],
            id="non-existent-vertices",
        ),
    ],
)
def test_add_edge(
    empty_graph, vertices_in_graph, edges_params, expected_edges_in_graph
):
    # given
    g = empty_graph
    vertices = [base.Vertex(name=name) for name in vertices_in_graph]
    g._vertices = {vertex.name: vertex for vertex in vertices}
    edges = []
    for edge in edges_params:
        if len(edge) == 2:
            edges.append((Vertex(name=edge[0]), Vertex(name=edge[1]), 1.0))
        else:
            edges.append((Vertex(name=edge[0]), Vertex(name=edge[1]), edge[2]))

    # when
    for edge in edges:
        g.add_edge(*edge)

    # then
    assert g._edges == expected_edges_in_graph


@pytest.mark.parametrize(
    (
        "vertices_in_graph",
        "edges_params",
        "expected_edges_in_graph",
        "expected_vertices_in_graph",
    ),
    [
        pytest.param(
            ["A", "B", "C"],
            [("A", "B", 1.0), ("A", "C", 0.2)],
            [
                base.UndirectedEdge(
                    vertex1=base.Vertex(name=cast(VertexName, "A")),
                    vertex2=base.Vertex(name=cast(VertexName, "B")),
                    weight=1.0,
                ),
                base.UndirectedEdge(
                    vertex1=base.Vertex(name=cast(VertexName, "A")),
                    vertex2=base.Vertex(name=cast(VertexName, "C")),
                    weight=0.2,
                ),
            ],
            ["A", "B", "C"],
            id="simple-weighted-edges",
        ),
        pytest.param(
            ["A", "B", "C"],
            [("A", "B"), ("C", "B")],
            [
                base.UndirectedEdge(
                    vertex1=base.Vertex(name=cast(VertexName, "A")),
                    vertex2=base.Vertex(name=cast(VertexName, "B")),
                    weight=1.0,
                ),
                base.UndirectedEdge(
                    vertex1=base.Vertex(name=cast(VertexName, "C")),
                    vertex2=base.Vertex(name=cast(VertexName, "B")),
                    weight=1.0,
                ),
            ],
            ["A", "B", "C"],
            id="not-weighted-edges",
        ),
        pytest.param(
            ["A", "B", "C"],
            [("A", "C", 0.3), ("E", "F", 0.4)],
            [
                base.UndirectedEdge(
                    vertex1=base.Vertex(name=cast(VertexName, "A")),
                    vertex2=base.Vertex(name=cast(VertexName, "C")),
                    weight=0.3,
                ),
                base.UndirectedEdge(
                    vertex1=base.Vertex(name=cast(VertexName, "E")),
                    vertex2=base.Vertex(name=cast(VertexName, "F")),
                    weight=0.4,
                ),
            ],
            ["A", "B", "C", "E", "F"],
            id="weighted-edges",
        ),
        pytest.param(
            ["A", "B", "C"],
            [("A", "D", 0.3)],
            [
                base.UndirectedEdge(
                    vertex1=base.Vertex(name=cast(VertexName, "A")),
                    vertex2=base.Vertex(name=cast(VertexName, "D")),
                    weight=0.3,
                ),
            ],
            ["A", "B", "C", "D"],
            id="non-existent-vertex",
        ),
        pytest.param(
            [],
            [("A", "D", 0.3)],
            [
                base.UndirectedEdge(
                    vertex1=base.Vertex(name=cast(VertexName, "A")),
                    vertex2=base.Vertex(name=cast(VertexName, "D")),
                    weight=0.3,
                ),
            ],
            ["A", "D"],
            id="non-existent-vertices",
        ),
    ],
)
def test_add_edges_from_list(
    empty_graph,
    vertices_in_graph,
    edges_params,
    expected_edges_in_graph,
    expected_vertices_in_graph,
):
    # given
    g = empty_graph
    vertices = [base.Vertex(name=name) for name in vertices_in_graph]
    g._vertices = {vertex.name: vertex for vertex in vertices}
    expected_vertices_in_graph = [
        base.Vertex(name=name) for name in expected_vertices_in_graph
    ]
    edges = []
    for edge in edges_params:
        if len(edge) == 2:
            edges.append((Vertex(name=edge[0]), Vertex(name=edge[1]), 1.0))
        else:
            edges.append((Vertex(name=edge[0]), Vertex(name=edge[1]), edge[2]))

    # when
    g.add_edges_from_list(edges)

    # then
    assert g._edges == expected_edges_in_graph
    vertices_in_graph = list(g._vertices.values())
    vertices_in_graph.sort(key=lambda x: x.name)
    expected_vertices_in_graph.sort(key=lambda x: x.name)
    assert vertices_in_graph == expected_vertices_in_graph


@pytest.mark.parametrize(
    ("edges", "vertex_name", "expected_neighbors"),
    [
        (
            [("A", "B"), ("A", "C"), ("B", "C")],
            "A",
            [base.Vertex(name="B"), base.Vertex(name="C")],
        ),
        (
            [("A", "B")],
            "A",
            [base.Vertex(name="B")],
        ),
        (
            [("C", "B")],
            "A",
            [],
        ),
    ],
)
def test_get_neighbors(empty_graph, edges, vertex_name, expected_neighbors):
    # given
    g = empty_graph
    g.add_vertex(base.Vertex(name=cast(VertexName, "A")))
    g.add_vertex(base.Vertex(name=cast(VertexName, "B")))
    g.add_vertex(base.Vertex(name=cast(VertexName, "C")))
    edges = [(Vertex(name=edge[0]), Vertex(name=edge[1]), 1.0) for edge in edges]
    g.add_edges_from_list(edges)

    # when
    neighbors = g.get_neighbors(vertex_name)

    # then
    assert neighbors == expected_neighbors


@pytest.mark.parametrize(
    ("vertex_name", "expected_result"),
    [
        ("A", True),
        ("B", True),
        ("C", False),
    ],
)
def test_contains(empty_graph, vertex_name, expected_result):
    # given
    g = empty_graph
    g.add_vertex(base.Vertex(name=cast(VertexName, "A")))
    g.add_edge(
        base.Vertex(name=cast(VertexName, "A")),
        base.Vertex(name=cast(VertexName, "B")),
    )

    # when
    result = base.Vertex(name=vertex_name) in g

    # then
    assert result == expected_result


def test_as_dot(empty_graph):
    # given
    g = empty_graph
    g.add_vertex(base.Vertex(name=cast(VertexName, "A")))
    g.add_vertex(base.Vertex(name=cast(VertexName, "B")))
    g.add_vertex(base.Vertex(name=cast(VertexName, "C")))
    g.add_edge(
        base.Vertex(name=cast(VertexName, "A")),
        base.Vertex(name=cast(VertexName, "B")),
    )

    # when
    dot = g.as_dot()

    # then
    assert dot == 'graph G {"A" -- "B" [weight=1.0];"C";}'


@pytest.mark.parametrize(
    ("vertices", "expected_edge"),
    [
        (
            (
                base.Vertex(name=cast(VertexName, "A")),
                base.Vertex(name=cast(VertexName, "B")),
            ),
            base.UndirectedEdge(
                vertex1=base.Vertex(name=cast(VertexName, "A")),
                vertex2=base.Vertex(name=cast(VertexName, "B")),
            ),
        ),
        (
            (
                base.Vertex(name=cast(VertexName, "B")),
                base.Vertex(name=cast(VertexName, "A")),
            ),
            base.UndirectedEdge(
                vertex1=base.Vertex(name=cast(VertexName, "A")),
                vertex2=base.Vertex(name=cast(VertexName, "B")),
            ),
        ),
        (
            (
                base.Vertex(name=cast(VertexName, "A")),
                base.Vertex(name=cast(VertexName, "C")),
            ),
            None,
        ),
        (
            (
                base.Vertex(name=cast(VertexName, "C")),
                base.Vertex(name=cast(VertexName, "D")),
            ),
            None,
        ),
    ],
)
def test_get_edge(empty_graph, vertices, expected_edge):
    # given
    g = empty_graph
    g.add_vertex(base.Vertex(name=cast(VertexName, "A")))
    g.add_vertex(base.Vertex(name=cast(VertexName, "B")))
    g.add_edge(
        base.Vertex(name=cast(VertexName, "A")),
        base.Vertex(name=cast(VertexName, "B")),
    )

    # when
    edge = g.get_edge(vertices)

    # then
    assert edge == expected_edge


def test_get_shortest_paths(empty_graph):
    # given
    g = empty_graph
    g.add_vertex(base.Vertex(name=cast(VertexName, "A")))
    g.add_vertex(base.Vertex(name=cast(VertexName, "B")))
    g.add_vertex(base.Vertex(name=cast(VertexName, "C")))
    g.add_vertex(base.Vertex(name=cast(VertexName, "D")))
    g.add_vertex(base.Vertex(name=cast(VertexName, "E")))
    g.add_edge(
        base.Vertex(name=cast(VertexName, "A")),
        base.Vertex(name=cast(VertexName, "B")),
    )
    g.add_edge(
        base.Vertex(name=cast(VertexName, "A")),
        base.Vertex(name=cast(VertexName, "C")),
    )
    g.add_edge(
        base.Vertex(name=cast(VertexName, "B")),
        base.Vertex(name=cast(VertexName, "D")),
    )
    g.add_edge(
        base.Vertex(name=cast(VertexName, "C")),
        base.Vertex(name=cast(VertexName, "D")),
    )

    # when
    shortest_paths = g.get_shortest_paths(base.Vertex(name=cast(VertexName, "A")))

    # then
    assert shortest_paths == {
        "A": 0,
        "B": 1,
        "C": 1,
        "D": 2,
        "E": base.INFINITY,
    }
