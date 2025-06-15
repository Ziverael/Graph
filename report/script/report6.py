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

# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from enum import Enum, auto
import random

# %% [markdown]
# # Task 1

# %%
np.random.randint(0, 10, size=100) < 3


# %%
class State(Enum):
    positive = auto()
    negative = auto()

class QVoter:
    def __init__(self, p: float, graph: nx.Graph, q: int):
        self.p = p
        self.graph = graph
        self.n = len(self.graph.nodes())
        self.q = q
        self.i = 0
        self.n
    
    def _init_graph(self):
        np.random.choice
        nx.set_node_attributes(self.graph, State)

    def init_state(self, p: float):
        self.i = 0
        self.picks = np.random.randint(0, self.n, size=self.n)
        self.conformism = np.random.random(size=self.n) < self.p

        self.indep_swtches = np.random.choice([-1, 1], size=self.conformism.sum())
        self.neighbors_picks = np.random.random(size=(self.n - self.conformism.sum(), self.q))

    def update_state(self):
        node = self.picks[self.i]
        if self.conformism[self.i]:
    #         nocd["state"] = 
    #     else:
    #         n
    #     self.graph.nodes[self.picks[self.i]]['state'] = self.probs[self.i]

    #     self.i += 1



# %%
n, k, p = 16, 4, 0.5
graph = nx.watts_strogatz_graph(n, k, p)
q = QVoter(0.5, graph, 4)
q.init_state()
q.conformism

# %%
n, k, p = 100, 4, 0.5
graph = nx.watts_strogatz_graph(n, k, p)
nx.draw(graph)
