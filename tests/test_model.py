from graph import model
import numpy as np
import pytest
from graph.base import Graph


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
