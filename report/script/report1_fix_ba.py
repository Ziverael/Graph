# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: graph-dXADCnCX-py3.12
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Barabasi-Albert: fix

# %% [markdown]
# The solution is still not efficient, but now the initialization is correct

# %%
# Replaced code: Do not run thos cell; it is dummy from source code

def get_weighted_random_neighbors(weights, subset_size):
    mapped_weights = [(random.expovariate(w), i) for i, w in enumerate(weights)]
    return {i for _, i in sorted(mapped_weights)[:subset_size]}


def _init_barabasi_albert_graph(graph: Graph, n: int, m: int) -> Graph:
    """Init Barabasi-Albert graph.
    n - number of vertices
    m - number of neighbors to connect to
    """
    if n <= m:
        msg = f"Initial number of nodes ({n}) must be greater than sampling size ({m})"
        raise GraphError(msg)
    graph = _init_connected_graph(graph, m)
    degrees = [graph.get_degree(v) for v in graph.vertices]

    for i in range(m, n):
        n_neighbours = [
            i + 1 for i in get_weighted_random_neighbors(degrees, subset_size=m)
        ]

        graph.add_edges_from_list(
            [
                (Vertex(name=VertexName(str(i))), Vertex(name=VertexName(str(j))))
                for j in n_neighbours
            ]
        )
        degrees.append(m)
        for j in n_neighbours:
            degrees[j] += 1
            edge_params = (
                Vertex(name=VertexName(str(i))),
                Vertex(name=VertexName(str(j))),
            )
            if graph.get_edge(edge_params) is None:
                graph.add_edge(*edge_params)
    return graph


# %%
from graph.model import (
    BarabasiAlbertGraph,
    RandomGraph,
    WattStrogatzGraph,
)
from graph.base import Graph
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from rich.table import Table
from scipy.stats import poisson


# %%
def get_stats(g: Graph):
    table = Table(title="Graph params")

    table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta", justify="right")

    table.add_row("Nodes", f"{n}")
    table.add_row("Edges", f"{len(g.edges)}")
    table.add_row("Mean degree", f"{g.mean_degree:.3f}")
    table.add_row("Variance of degree", f"{var_of_degree:.3f}")

    return table


def plot_degree_distribution(degrees, **kwargs):
    fig, ax = plt.subplots()
    ax.hist(degrees, density=True, **kwargs)
    return fig, ax


# %%
n = 500
m = 7
g = BarabasiAlbertGraph(n, m)

# %%
degrees = np.array([g.get_degree(v) for v in g.vertices])
var_of_degree = np.var(degrees)

# %%
fig, ax = plot_degree_distribution(degrees, bins=8)
ax.set_title("Degree distribution")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k=x)$")
ax.set_xscale("log")
ax.set_yscale("log")
plt.show()
get_stats(g)

# %%
n = 500
m = 16
g = BarabasiAlbertGraph(n, m)

# %%
degrees = np.array([g.get_degree(v) for v in g.vertices])
var_of_degree = np.var(degrees)

# %%
fig, ax = plot_degree_distribution(degrees, bins=8)
ax.set_title("Degree distribution")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k=x)$")
ax.set_xscale("log")
ax.set_yscale("log")
plt.show()
get_stats(g)
