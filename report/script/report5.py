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
# # Report 5

# %% [markdown]
# ## Imports

# %%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from collections import namedtuple
import networkx as nx
from enum import Enum, auto
from random import sample
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import seaborn as sns


# %% [markdown]
# ## Task 1

# %%
SIRStep = namedtuple('SIRStep', ['S', 'I', 'R'])

class SIR:
    def __init__(self, N, I0, beta: float, r: float):
        self.N = N
        self.I0 = I0
        self.R0 = beta / r
        self.S0 = N - I0 - self.R0
        self.beta = beta
        self.r = r
    
    @property
    def R0_value(self):
        return self.R0

    
    def deriv(self, y: SIRStep, t: float) -> tuple[float, float, float]:
        S, I, R = y
        dSdt = -self.beta * S * I / self.N
        dIdt = self.beta * S * I / self.N - self.r * I
        dRdt = self.r * I
        return dSdt, dIdt, dRdt
    
    def solve(self, t0: float, T: float, steps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y0 = SIRStep(S=self.S0, I=self.I0, R=self.R0)
        t = np.linspace(t0, T, steps)
        ret = odeint(self.deriv, y0, t)
        S, I, R = ret.T
        return S, I, R

    def plot_SIR(self, t0, T, steps, sir_solution):
        t = np.linspace(t0, T, steps)
        S, I, R = sir_solution
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        ax.plot(t, S/self.N, 'b', alpha=0.5, lw=2, label='Susceptible')
        ax.plot(t, I/self.N, 'r', alpha=0.5, lw=2, label='Infected')
        ax.plot(t, R/self.N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
        ax.set_xlabel(r'$t$')
        ax.set_ylabel('Population %')
        ax.set_ylim(0,1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(which='major', c='w', lw=1, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.show()

# %%
N = 1000
t0, T, steps = 0, 100, 100

# %%
I0 = 10
beta = 0.3
r = .1
model=SIR(N, I0, beta, r)
model.plot_SIR(t0, T, steps, model.solve(t0, T, steps))
print("R0=",model.R0_value)

# %%
I0 = 1
beta = 0.3
r = .1
model=SIR(N, I0, beta, r)
model.plot_SIR(t0, T, steps, model.solve(t0, T, steps))
print("R0=",model.R0_value)

# %%
I0 = 2
beta = 0.3
r = .1
model=SIR(N, I0, beta, r)
model.plot_SIR(t0, T, steps, model.solve(t0, T, steps))
print("R0=",model.R0_value)

# %%
I0 = 30
beta = 0.3
r = .1
model=SIR(N, I0, beta, r)
model.plot_SIR(t0, T, steps, model.solve(t0, T, steps))
print("R0=",model.R0_value)

# %%
I0 = 100
beta = 0.1
r = .1
model=SIR(N, I0, beta, r)
model.plot_SIR(t0, T, steps, model.solve(t0, T, steps))
print("R0=",model.R0_value)

# %%
I0 = 200
beta = 0.12
r = .1
model=SIR(N, I0, beta, r)
model.plot_SIR(t0, T, steps, model.solve(t0, T, steps))
print("R0=",model.R0_value)

# %%
I0 = 1
beta = 0.2
r = .1
model=SIR(N, I0, beta, r)
model.plot_SIR(t0, T, steps, model.solve(t0, T, steps))
print("R0=",model.R0_value)

# %%
I0 = 1
beta = 0.2
r = .005
model=SIR(N, I0, beta, r)
model.plot_SIR(t0, T, steps, model.solve(t0, T, steps))
print("R0=",model.R0_value)

# %%
I0 = 100
beta = 0.1
r = .3
model=SIR(N, I0, beta, r)
model.plot_SIR(t0, T, steps, model.solve(t0, T, steps))
print("R0=",model.R0_value)

# %%
I0 = 10
beta = 0.2
r = .3
model=SIR(N, I0, beta, r)
model.plot_SIR(t0, T, steps, model.solve(t0, T, steps))
print("R0=",model.R0_value)

# %% [markdown]
# ### Task1.2

# %%
SIRStep = namedtuple('SIRStep', ['S', 'I', 'R'])

class SIRReduced:
    def __init__(self, N, I0, beta: float, r: float):
        self.N = N
        self.I0 = I0
        self.R0 = beta / r
        self.S0 = N - I0 - self.R0
        self.beta = beta
        self.r = r
    
    @property
    def R0_value(self):
        return self.R0

    
    def deriv(self, y: SIRStep, t: float) -> tuple[float, float]:
        S, I = y
        dSdt = -self.beta * S * I / self.N
        dIdt = self.beta * S * I / self.N - self.r * I
        return dSdt, dIdt
    
    def solve(self, t0: float, T: float, steps: float) -> tuple[np.ndarray, np.ndarray]:
        y0 = SIRStep(S=self.S0, I=self.I0)
        t = np.linspace(t0, T, steps)
        ret = odeint(self.deriv, y0, t)
        S, I = ret.T
        return S, I

    def plot_phase_portrait(
        self,
        S_range: tuple[float, float | None]=(0, None),
        I_range: tuple[float, float | None] = (0, None),
        density = 20,
        scale = 100,
    ):
        if S_range[1] is None:
            S_range = (S_range[0], self.N)
        if I_range[1] is None:
            I_range = (I_range[0], self.N / 10)

        S_space = np.linspace(*S_range, density)
        I_space = np.linspace(*I_range, density)
        S, I = np.meshgrid(S_space, I_space)

        dS, dI = self.deriv((S, I), 0)

        plt.figure(figsize=(8, 6))
        plt.quiver(S, I, dS, dI, color='blue', scale=scale, alpha=0.7)
        plt.xlabel('S')
        plt.ylabel('I')
        plt.title('Phase Portrait of Reduced SIR')
        plt.grid(True)
        plt.xlim(S_range)
        plt.ylim(I_range)
        plt.show()

# %%
sir = SIRReduced(N=1000, I0=1, beta=0.3, r=0.1)
sir.plot_phase_portrait(density=40, scale=1000)

# %%
sir = SIRReduced(N=1000, I0=100, beta=0.3, r=0.1)
sir.plot_phase_portrait(density=40, scale=1000)

# %%
sir = SIRReduced(N=1000, I0=100, beta=0.3, r=0.7)
sir.plot_phase_portrait(density=20, scale = 1000)

# %%
sir = SIRReduced(N=1000, I0=100, beta=0.8, r=0.4)
sir.plot_phase_portrait(density=20, scale = 1000)

# %%
I0 = 1
beta = 0.2
r = .005
model=SIRReduced(N, I0, beta, r)
model.plot_phase_portrait(density=20, scale = 500)
print("R0=",model.R0_value)

# %%
I0 = 200
beta = 0.12
r = .1
model=SIRReduced(N, I0, beta, r)
model.plot_phase_portrait(density=20, scale = 500)
print("R0=",model.R0_value)


# %% [markdown]
# ## Task 2

# %%
class State(Enum):
    suspectible = auto()
    infected = auto()
    removed = auto()

def _change_state(unit: dict[str, State]) -> dict[str, State]:
    pass


# %%
N=100

class SIROnGraph():
    models = {
        "2D lattice": nx.grid_2d_graph,
        "Erdos-Renyi": nx.erdos_renyi_graph,
        "Watts-Strogatz": nx.watts_strogatz_graph,
        "Barabasi-Albert": nx.barabasi_albert_graph,
    }

    state_color_mapping = {
        State.suspectible: "blue",
        State.infected: "red",
        State.removed: "green",
    }
    
    def __init__(
        self,
        model: str,
        mc_steps: int,
        I0: int,
        p_infection: float,
        **kwargs,
    ) -> None:
        self.graph = SIROnGraph.models[model](**kwargs)
        nx.set_node_attributes(self.graph, State.suspectible, "state")
        nx.set_node_attributes(
            self.graph,
            {
                node: {"state": State.infected}
                for node in sample(list(self.graph.nodes), k=I0)
            }
        )

        self.mc_steps = mc_steps
        self.pos = nx.spring_layout(self.graph, seed=10)
        self.p = p_infection

        self.init_state = nx.get_node_attributes(self.graph, "state")
    
    def reset_state(self):
        nx.set_node_attributes(self.graph, self.init_state, "state")
    
    def _update_state(self):
        if len(infected_population:=[k for k, v in self.graph.nodes.items() if v["state"] == State.infected]) > 0:
            dangered_nodes = set()
            for infected_node in infected_population:
                dangered_nodes.update({node for node in self.graph.neighbors(infected_node) if self.graph.nodes[node]["state"] == State.suspectible})
            is_infected_states = np.random.random(len(dangered_nodes)) < self.p
            new_infected_nodes = [node for node, is_infected in zip(dangered_nodes, is_infected_states) if is_infected]
            nx.set_node_attributes(
                self.graph,
                {
                    node: {"state": State.infected}
                    for node in new_infected_nodes
                }
            )
            nx.set_node_attributes(
                self.graph,
                {
                    node: {"state": State.removed}
                    for node in infected_population
                }
            )
    
    def run_simulation(self):
        statistics = {
            "infected_count": [],
            "suspectible_count": [],
            "removed_count": [],
        }
        while len([v for v in nx.get_node_attributes(self.graph, "state").values() if v == State.infected]) > 0:
            statistics["infected_count"].append(len([v for v in nx.get_node_attributes(self.graph, "state").values() if v == State.infected]))
            statistics["suspectible_count"].append(len([v for v in nx.get_node_attributes(self.graph, "state").values() if v == State.suspectible]))
            statistics["removed_count"].append(len([v for v in nx.get_node_attributes(self.graph, "state").values() if v == State.removed]))
            self._update_state()
        return statistics


    def _get_nodes_state_mapping(self) -> list[int]:
        return [self.state_color_mapping[s] for s in nx.get_node_attributes(self.graph,'state').values()]        
    
    def _plot_current_state(self, ax):
        ax.clear()
        nx.draw_networkx(
            self.graph, 
            pos=self.pos, 
            with_labels=True, 
            node_color=self._get_nodes_state_mapping(),
            edge_color='gray',
            width=1.0,
            alpha=0.8,
            ax=ax
        )
        ax.set_title(f"Random Walk on graph")
        ax.set_axis_off()


def plot_SIR_on_graph(model, steps):
    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame):
        model._update_state()
        model._plot_current_state(ax)

    return FuncAnimation(fig, update, frames=steps, repeat=False)


VIDEO_WRITER = animation.FFMpegWriter(fps=30, bitrate=1800, extra_args=['-vcodec', 'libx264', '-preset', 'ultrafast'])
VIDEO_WRITER_SLOW = animation.FFMpegWriter(fps=2, bitrate=1800, extra_args=['-vcodec', 'libx264', '-preset', 'slow'])
def save_animation(ani: FuncAnimation, filename: str, writer: animation.FFMpegWriter = VIDEO_WRITER_SLOW):
    ani.save(filename, writer=writer)



# %%
save_animation(plot_SIR_on_graph(SIROnGraph("2D lattice", m=10, n=10, I0=1, p_infection = 1.0, mc_steps=10), steps=20), "SIR_2d_p1.mp4")
save_animation(plot_SIR_on_graph(SIROnGraph("2D lattice", m=10, n=10, I0=1, p_infection = .5, mc_steps=10), steps=20), "SIR_2d_p05.mp4")
save_animation(plot_SIR_on_graph(SIROnGraph("2D lattice", m=10, n=10, I0=1, p_infection = .3, mc_steps=10), steps=20), "SIR_2d_p03.mp4")


# %%
# save_animation(plot_SIR_on_graph(SIROnGraph("Erdos-Renyi", p = 0.07, n=100, I0=1, p_infection = 1.0, mc_steps=10), steps=20), "SIR_Erdos_renyi_p1.mp4")
# save_animation(plot_SIR_on_graph(SIROnGraph("Erdos-Renyi", p = 0.07, n=100, I0=1, p_infection = 1.0, mc_steps=10), steps=20), "SIR_Erdos_renyi_p1.mp4")
# save_animation(plot_SIR_on_graph(SIROnGraph("Erdos-Renyi", p = 0.07, n=100, I0=1, p_infection = 1.0, mc_steps=10), steps=20), "SIR_Erdos_renyi_p1.mp4")

# %% [markdown]
# ## Get simulation stats

# %%
def simulate_and_plot_stats(model: SIROnGraph, title: str, ax: plt.Axes):
    stats = model.run_simulation()
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=stats, ax=ax)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$N(t)$")
    ax.set_title(title)



# %%

fig, ax = plt.subplots(4, 3, figsize=(20, 20), sharex=False, sharey=True)
for idx, (model, title) in enumerate([
    (SIROnGraph("2D lattice", m=20, n=20, I0=1, p_infection = 1.0, mc_steps=20), r"2D lattice ($p=1.0$)"),
    (SIROnGraph("2D lattice", m=20, n=20, I0=1, p_infection = .5, mc_steps=20), r"2D lattice ($p=0.5$)"),
    (SIROnGraph("2D lattice", m=20, n=20, I0=1, p_infection = .3, mc_steps=20), r"2D lattice ($p=0.3$)"),

    (SIROnGraph("Erdos-Renyi", p = 0.07, n=400, I0=1, p_infection = 1.0, mc_steps=20), r"Erdos-Renyi ($p=1.0$)"),
    (SIROnGraph("Erdos-Renyi", p = 0.07, n=400, I0=1, p_infection = .5, mc_steps=20), r"Erdos-Renyi ($p=0.5$)"),
    (SIROnGraph("Erdos-Renyi", p = 0.07, n=400, I0=1, p_infection = .3, mc_steps=20), r"Erdos-Renyi ($p=0.3$)"),

    (SIROnGraph("Watts-Strogatz", k=4, p=0.07, n=400, I0=1, p_infection = 1.0, mc_steps=20), r"Watts-Strogatz ($p=1.0$)"),
    (SIROnGraph("Watts-Strogatz", k=4, p=0.07, n=400, I0=1, p_infection = .5, mc_steps=20), r"Watts-Strogatz ($p=0.5$)"),
    (SIROnGraph("Watts-Strogatz", k=4, p=0.07, n=400, I0=1, p_infection = .3, mc_steps=20), r"Watts-Strogatz ($p=0.3$)"),

    (SIROnGraph("Barabasi-Albert", m=300, n=400, I0=1, p_infection = 1.0, mc_steps=20), r"Barabasi-Albert ($p=1.0$)"),
    (SIROnGraph("Barabasi-Albert", m=300, n=400, I0=1, p_infection = .5, mc_steps=20), r"Barabasi-Albert ($p=0.5$)"),
    (SIROnGraph("Barabasi-Albert", m=300, n=400, I0=1, p_infection = .3, mc_steps=20), r"Barabasi-Albert ($p=0.3$)"),
]):
    current_ax = ax.flat[idx]
    simulate_and_plot_stats(model, title, current_ax)
fig.tight_layout()


# %%
