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
# # Report 1: Assignment 3

# %%
from graph.model import (
    RandomGraph,
    WattStrogatzGraph,
)
from graph.base import Graph
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.widgets import Slider
from rich.table import Table
from scipy.stats import poisson

matplotlib.use("module://ipympl.backend_nbagg")


# %% [markdown]
# # Properties analysis


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


# %% [markdown]
# ## Random Graph (Erdos-Renyi)

# %%
n = 2000
p = 0.004
g = RandomGraph(n, p=p)
# %%
degrees = np.array([g.get_degree(v) for v in g.vertices])
var_of_degree = np.var(degrees)

# %%
fig, ax = plot_degree_distribution(degrees, bins=8)
mu = 8
x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
ax.scatter(x, poisson.pmf(x, mu), color="red")
ax.set_title("Degree distribution")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k=x)$")
plt.show()
get_stats(g)

# %%
p = 0.003
g = RandomGraph(n, p=p)
degrees = np.array([g.get_degree(v) for v in g.vertices])
var_of_degree = np.var(degrees)

# %%
fig, ax = plot_degree_distribution(degrees, bins=8)
mu = 7
x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
ax.scatter(x, poisson.pmf(x, mu), color="red")
ax.set_title("Degree distribution")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k=x)$")
plt.show()
get_stats(g)

# %%
p = 0.001
g = RandomGraph(n, p=p)
degrees = np.array([g.get_degree(v) for v in g.vertices])
var_of_degree = np.var(degrees)

# %%
fig, ax = plot_degree_distribution(degrees, bins=8)
mu = 2
x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
ax.scatter(x, poisson.pmf(x, mu), color="red")
ax.set_title("Degree distribution")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k=x)$")
plt.show()
get_stats(g)

# %% [markdown]
# ## Random Graph (Gilbert)

# %%
n = 2000
m = 4000
g = RandomGraph(n, m=m)
degrees = np.array([g.get_degree(v) for v in g.vertices])
var_of_degree = np.var(degrees)

# %%
fig, ax = plot_degree_distribution(degrees, bins=8)
mu = 4.2
x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
ax.scatter(x, poisson.pmf(x, mu), color="red")
ax.set_title("Degree distribution")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k=x)$")
plt.show()
get_stats(g)

# %%
n = 2000
m = 2000
g = RandomGraph(n, m=m)
degrees = np.array([g.get_degree(v) for v in g.vertices])
var_of_degree = np.var(degrees)

# %%
fig, ax = plot_degree_distribution(degrees, bins=8)
mu = 2
x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
ax.scatter(x, poisson.pmf(x, mu), color="red")
ax.set_title("Degree distribution")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k=x)$")
plt.show()
get_stats(g)

# %%
n = 2000
m = 6000
g = RandomGraph(n, m=m)
degrees = np.array([g.get_degree(v) for v in g.vertices])
var_of_degree = np.var(degrees)

# %%
fig, ax = plot_degree_distribution(degrees, bins=8)
mu = 6.5
x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
ax.scatter(x, poisson.pmf(x, mu), color="red")
ax.set_title("Degree distribution")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k=x)$")
plt.show()
get_stats(g)

# %% [markdown]
# ## Watt-Strogatz

# %%
k = 4
beta = 0.8
g = WattStrogatzGraph(n, k=k, beta=beta)
degrees = np.array([g.get_degree(v) for v in g.vertices])
var_of_degree = np.var(degrees)

# %%
fig, ax = plot_degree_distribution(degrees, bins=8)
mu = 2
x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
ys = poisson.pmf(x, mu)
ax.scatter(x + 2, ys, color="red")
ax.set_title("Degree distribution")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k=x)$")
plt.show()
get_stats(g)

# %%
k = 4
beta = 0.3
g = WattStrogatzGraph(n, k=k, beta=beta)
degrees = np.array([g.get_degree(v) for v in g.vertices])
var_of_degree = np.var(degrees)

# %%
fig, ax = plot_degree_distribution(degrees, bins=8)
mu = 2.2
x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
ys = poisson.pmf(x, mu)
ax.scatter(x + 2, ys, color="red")
ax.set_title("Degree distribution")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k=x)$")
plt.show()
get_stats(g)

# %%
k = 6
beta = 0.7
g = WattStrogatzGraph(n, k=k, beta=beta)
degrees = np.array([g.get_degree(v) for v in g.vertices])
var_of_degree = np.var(degrees)

# %%
fig, ax = plot_degree_distribution(degrees, bins=7)
mu = 4
x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
ys = poisson.pmf(x, mu)
ax.scatter(x + 2, ys, color="red")
ax.set_title("Degree distribution")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k=x)$")
plt.show()
get_stats(g)

# %%
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)


axcolor = "lightgoldenrodyellow"
ax_n_slider = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_p_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
n_size = Slider(ax_n_slider, "N", 1, 5_000.0, valinit=15)
probability = Slider(ax_p_slider, "p", 0.0, 1.0, valinit=0.5)


def update(val):
    n = n_size.val
    p = probability.val
    line.set_ydata(np.linspace(1, 100, 100) * p)
    fig.canvas.draw_idle()


(line,) = ax.plot(np.linspace(1, 100, 100), np.linspace(1, 100, 100) * 2)

n_size.on_changed(update)
probability.on_changed(update)
plt.show()
