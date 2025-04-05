from graph import model
import numpy as np
import pytest
from graph.base import Graph
from typing import cast
from unittest.mock import call, ANY
from pytest_mock import MockerFixture


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
def test_ring_lattice_graph(n, k):
    # when
    g = model._init_ring_lattice_graph(n, k)

    # then
    assert len(g.vertices) == n
    for v in g.vertices:
        degree = sum(1 for e in g.edges if v in e)
        assert degree == k


def test_watts_strogatz_graph():
    # given
    n = 10
    k = 4
    beta = 0.5
    mc_steps = 1000

    # when
    mean_degree = []
    for _ in range(mc_steps):
        g = model._init_watts_strogatz_graph(n, k, beta)
        mean_degree.append(g.mean_degree)

    # then
    assert len(g.vertices) == n
    assert np.mean(mean_degree) == pytest.approx(k, abs=0.1)
