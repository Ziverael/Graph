from graph import model
import numpy as np
import pytest
from graph.base import Graph
from typing import cast
from unittest.mock import call, ANY
from pytest_mock import MockerFixture
from collections import defaultdict


@pytest.mark.parametrize("n", [0, 1, 3, 10, 20, 30])
def test_connected_graph(empty_graph, n):
    # when
    g = model._init_connected_graph(empty_graph, n=n)

    # then
    assert isinstance(g, Graph)
    assert len(g.vertices) == n
    for v in g.vertices:
        expected_neighbors = (
            g.vertices[g.vertices.index(v) + 1 :] + g.vertices[: g.vertices.index(v)]
        )
        for expected_neighbor in expected_neighbors:
            assert g.get_edge((v, expected_neighbor)) is not None


def test_erdos_renyi_graph():
    # given
    n = 10
    p = 0.5
    mc_steps = 10_000
    number_of_edges: list[int] = []

    # when
    for _ in range(mc_steps):
        g = Graph()
        g = model._init_erdos_renyi_graph(g, n=n, p=p)
        number_of_edges.append(len(g.edges))

        # then
        assert len(g.vertices) == n

    assert np.mean(number_of_edges) == pytest.approx(n * (n - 1) * 0.5 * p, abs=0.1)


def test_gilbert_graph(empty_graph):
    # given
    n = 10
    m = 10
    mc_steps = 10_000

    # when / then
    for _ in range(mc_steps):
        g = Graph()
        g = model._init_gilbert_graph(g, n=n, m=m)

        assert len(g.vertices) == n
        assert len(g.edges) == m


@pytest.mark.parametrize(
    ("parameters", "expected_model"),
    [
        pytest.param(["m"], "Gilbert", id="Gilbert"),
        pytest.param(["p"], "Erdos-Renyi", id="Erdos-Renyi"),
        pytest.param(["m", "p"], None, id="Invalid"),
        pytest.param([], None, id="Invalid2"),
    ],
)
def test_random_graph_init(mocker: MockerFixture, parameters, expected_model):
    # given
    n = 10
    m = 10 if "m" in parameters else None
    p = 0.5 if "p" in parameters else None
    kwargs: dict[str, int | float] = {}
    if m is not None:
        kwargs["m"] = m
    if p is not None:
        kwargs["p"] = p
    expected_graph = cast(model.RandomGraph, "dummy_graph")

    mocked_call_init_gilbert_graph = mocker.patch.object(
        model,
        model._init_gilbert_graph.__name__,
        return_value=expected_graph,
    )
    mocked_call_init_erdos_renyi_graph = mocker.patch.object(
        model,
        model._init_erdos_renyi_graph.__name__,
        return_value=expected_graph,
    )

    # when / then
    if expected_model is None:
        expected_error_message = "Init arguments do not match any model"
        with pytest.raises(model.GraphError, match=expected_error_message):
            g = model.RandomGraph(
                n=n, m=m if m is not None else None, p=p if p is not None else None
            )
    else:
        g = model.RandomGraph(
            n=n, m=m if m is not None else None, p=p if p is not None else None
        )
        assert isinstance(g, model.RandomGraph)
        if expected_model == "Gilbert":
            assert mocked_call_init_gilbert_graph.call_count == 1
            assert mocked_call_init_gilbert_graph.call_args == call(ANY, n=n, m=m)
        elif expected_model == "Erdos-Renyi":
            assert mocked_call_init_erdos_renyi_graph.call_count == 1
            assert mocked_call_init_erdos_renyi_graph.call_args == call(ANY, n=n, p=p)


@pytest.mark.parametrize("n, k", [(10, 4), (9, 2), (28, 10), (100, 10)])
def test_ring_lattice_graph(empty_flexible_graph, n, k):
    # when
    g = model._init_ring_lattice_graph(empty_flexible_graph, n, k)

    # then
    assert len(g.vertices) == n
    for v in g.vertices:
        assert len(g.get_neighbors(v.name)) == k


def test_watts_strogatz_graph():
    # given
    n = 10
    k = 4
    beta = 0.5
    mc_steps = 1000

    # when
    for _ in range(mc_steps):
        g = model._init_watts_strogatz_graph(model.FlexibleGraph(), n, k, beta)

        # then
        assert len(g.vertices) == n
        assert len(g.edges) == n * k / 2
        assert all(e.vertex1.name != e.vertex2.name for e in g.edges)
        assert all(e.weight == 1.0 for e in g.edges)
        for i in range(len(g.edges)):
            for j in range(i + 1, len(g.edges)):
                assert (
                    g.edges[i].vertex1.name != g.edges[j].vertex1.name
                    or g.edges[i].vertex2.name != g.edges[j].vertex2.name
                ), (
                    f"{g.edges[i].vertex1.name} -- {g.edges[i].vertex2.name} == {g.edges[j].vertex1.name} -- {g.edges[j].vertex2.name}"
                )


def test_watts_strogatz_graph_init(empty_flexible_graph, mocker: MockerFixture):
    # given
    n = 10
    k = 4
    beta = 0.5
    expected_graph = cast(model.WattStrogatzGraph, "dummy_graph")
    mocked_call_watts_strogatz_graph = mocker.patch.object(
        model,
        model._init_watts_strogatz_graph.__name__,
        return_value=expected_graph,
    )

    # when
    g = model.WattStrogatzGraph(n=n, k=k, beta=beta)

    # then
    assert isinstance(g, model.WattStrogatzGraph)
    assert mocked_call_watts_strogatz_graph.call_count == 1
    assert mocked_call_watts_strogatz_graph.call_args == call(ANY, n=n, k=k, beta=beta)


def test_lattice_ring_graph():
    # given
    n = 10
    k = 4

    # when
    g = model.LatticeRingGraph(n, k)

    # then
    assert isinstance(g, model.LatticeRingGraph)


@pytest.mark.parametrize(("n", "m"), [(10, 2), (15, 9), (16, 8)])
def test_barabasi_albert_graph(empty_graph, n, m):
    # given
    mc_steps = 1000
    degrees: dict[int, list[int]] = defaultdict(list)

    # when / then
    for _ in range(mc_steps):
        g = model._init_barabasi_albert_graph(model.Graph(), n=n, m=m)
        assert len(g.vertices) == n
        for v in g.vertices:
            degrees[int(v.name)].append(g.get_degree(v))


@pytest.mark.parametrize(
    ("n", "m", "expected_error"),
    [
        pytest.param(
            2,
            2,
            r"Initial number of nodes \(2\) must be greater than sampling size \(2\)",
        ),
    ],
)
def test_barabasi_albert_graph__error(empty_graph, n0, n, m, expected_error):
    # when / then
    with pytest.raises(model.GraphError, match=expected_error):
        model.BarabasiAlbertGraph(n=n, m=m)


def test_barabasi_albert_graph_init(empty_graph, mocker: MockerFixture):
    # given
    n = 10
    m = 2
    expected_graph = cast(model.BarabasiAlbertGraph, "dummy_graph")
    mocked_call_barabasi_albert_graph = mocker.patch.object(
        model,
        model._init_barabasi_albert_graph.__name__,
        return_value=expected_graph,
    )

    # when
    g = model.BarabasiAlbertGraph(n=n, m=m)

    # then
    assert isinstance(g, model.BarabasiAlbertGraph)
    assert mocked_call_barabasi_albert_graph.call_count == 1
    assert mocked_call_barabasi_albert_graph.call_args == call(ANY, n=n, m=m)
