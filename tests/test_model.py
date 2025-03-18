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


def test_random_graph_erdos_renyi(mocker: MockerFixture):
    # given
    n = 10
    p = 0.5
    expected_graph = cast(Graph, "dummy_graph")

    mocked_call_init_erdos_renyi_graph = mocker.patch.object(
        model,
        model._init_erdos_renyi_graph.__name__,
        return_value=expected_graph,
    )
    mocked_call_graph_init = mocker.patch.object(
        model.Graph, "__init__", return_value=None
    )

    # when
    g = model.RandomGraph(n=n, p=p)

    # then
    assert g == expected_graph
    assert mocked_call_init_erdos_renyi_graph.call_count == 1
    assert mocked_call_init_erdos_renyi_graph.call_args == call(ANY, n=n, p=p)
    assert mocked_call_graph_init.call_count == 1
    assert mocked_call_graph_init.call_args == call()


def test_random_graph_gilbert(mocker: MockerFixture):
    # given
    n = 10
    m = 10
    expected_graph = cast(Graph, "dummy_graph")

    mocked_call_init_gilbert_graph = mocker.patch.object(
        model,
        model._init_gilbert_graph.__name__,
        return_value=expected_graph,
    )
    mocked_call_graph_init = mocker.patch.object(
        model.Graph, "__init__", return_value=None
    )

    # when
    g = model.RandomGraph(n=n, m=m)

    # then
    assert g == expected_graph
    assert mocked_call_init_gilbert_graph.call_count == 1
    assert mocked_call_init_gilbert_graph.call_args == call(ANY, n=n, m=m)
    assert mocked_call_graph_init.call_count == 1
    assert mocked_call_graph_init.call_args == call()
