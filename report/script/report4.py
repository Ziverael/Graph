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
# # List 4

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import networkx as nx
from matplotlib.animation import FuncAnimation
# %matplotlib inline

# %%
VIDEO_WRITER = animation.FFMpegWriter(fps=60) 

# %%
def save_animation(ani: FuncAnimation, filename: str):
    ani.save(filename, writer=VIDEO_WRITER)

# %% [markdown]
# # Task 1

# %%
steps = 1_000_000
fig, ax = plt.subplots()
x = np.cumsum(np.random.choice((-1, 1), size=steps, p=(0.5, 0.5)))
y = np.cumsum(np.random.choice((-1, 1), size=steps, p=(0.5, 0.5)))

line, = ax.plot([], [], lw=1)
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))

def update(frame):
    line.set_data(x[:frame], y[:frame])
    return line,

ani = animation.FuncAnimation(fig, update, frames=np.linspace(1, steps, num=5_000, dtype=int),
                              interval=30, blit=True)
save_animation(ani, 'random_walk.mp4')

# %% [markdown]
# # Taks 2

# %%

steps = 1_000_000
step_length = 1

fig, ax = plt.subplots()

angles = np.random.uniform(0, 2 * np.pi, size=steps)
x = np.cumsum(step_length * np.cos(angles))
y = np.cumsum(step_length * np.sin(angles))

line, = ax.plot([], [], lw=1)
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))

def update(frame):
    line.set_data(x[:frame], y[:frame])
    return line,

ani = animation.FuncAnimation(fig, update, frames=np.linspace(1, steps, num=5_000, dtype=int),
                              interval=30, blit=True)

save_animation(ani, 'random_pearson_walk.mp4')


# %%
def get_pearson_random_walk(steps, step_length):
    angles = np.random.uniform(0, 2 * np.pi, size=steps)
    x = np.cumsum(step_length * np.cos(angles))
    y = np.cumsum(step_length * np.sin(angles))
    return np.stack((x, y), axis=1)



# %%
mc_steps = 100_000
simulations = np.stack([get_pearson_random_walk(1000, 1) for _ in range(mc_steps)])

# %%
right_side_means = (simulations[:, :, 0] > 0.0).mean(axis=0)
first_quadrant_means = ((simulations[:, :, 0] > 0.0) & (simulations[:, :, 1] > 0.0)).mean(axis=0)

# %%
# %matplotlib inline
fig, ax = plt.subplots(1, 2, figsize=(12, 7))
ax[0].hist(right_side_means, bins=24)
ax[0].set_title('Right side')
ax[1].hist(first_quadrant_means, bins=24)
ax[1].set_title('First quadrant')
plt.show()

# %% [markdown]
# # Task 3

# %%
N_view = 20
steps_view = 100
N_tests = 100


# %%
def random_walk(g: nx.Graph, steps: int) -> np.ndarray:
    path = np.empty(steps + 1, dtype=int)
    nodes = list(g.nodes())
    current_node = np.random.choice(nodes)
    path[0] = current_node

    neighbor_cache = {}
    for i in range(steps):
        if current_node not in neighbor_cache:
            neighbor_cache[current_node] = list(g.neighbors(current_node))
        neighbors = neighbor_cache[current_node]
        current_node = neighbors[np.random.randint(len(neighbors))]
        path[i + 1] = current_node
    return path


# %%
def get_random_walk_animation(
    g: nx.Graph, 
    node_colors: np.ndarray,
    seed: int = 1,
) -> FuncAnimation:
    pos = nx.spring_layout(g, seed=seed)
    fig, ax = plt.subplots(figsize=(10, 8))
    def update(frame):
        ax.clear()
        nx.draw_networkx(
            g, 
            pos=pos, 
            with_labels=True, 
            node_color=node_colors[:, frame],
            edge_color='gray',
            width=1.0,
            alpha=0.8,
            ax=ax
        )
        ax.set_title(f"Random Walk on graph")
        ax.set_axis_off()
        
    return FuncAnimation(
        fig, 
        update, 
        frames=node_colors.shape[1], 
        interval=500,
        repeat=True,
        repeat_delay=1000
    )


# %%
def show_random_walk_animation(g: nx.Graph, steps: int, seed: int = 1):
    path = random_walk(g, steps_view)
    node_colors = np.full((N_view, steps_view + 1), 'lightblue', dtype='<U10')
    node_colors[path, np.arange(steps_view + 1)] = 'red'
    return get_random_walk_animation(g, node_colors, seed)


# %%
save_animation(show_random_walk_animation(nx.erdos_renyi_graph(N_view, 0.3), steps_view), "graph_walk_erdos_p03.mp4")

# %%
save_animation(show_random_walk_animation(nx.erdos_renyi_graph(N_view, 0.7), steps_view), "graph_walk_erdos_p04.mp4")

# %%
save_animation(show_random_walk_animation(nx.watts_strogatz_graph(N_view, 4, 0.1), steps_view), "graph_walk_watts_4_01.mp4")

# %%
save_animation(show_random_walk_animation(nx.watts_strogatz_graph(N_view, 4, 0.0), steps_view), "graph_walk_watts_4_00.mp4")

# %%
save_animation(show_random_walk_animation(nx.watts_strogatz_graph(N_view, 10, 0.0), steps_view), "graph_walk_watts_10_00.mp4")

# %%
save_animation(show_random_walk_animation(nx.barabasi_albert_graph(N_view, 7), steps_view), "graph_walk_ba_7.mp4")

# %%
save_animation(show_random_walk_animation(nx.barabasi_albert_graph(N_view, 3), steps_view), "graph_walk_ba_3.mp4")

# %%
MAX_LIMIT = 10_000
def get_all_nodes_visit_time(g: nx.Graph, max_limit: int = MAX_LIMIT) -> np.ndarray:
    nodes = set(g.nodes())
    visited_nodes: set[int] = set()
    current_node = np.random.choice(list(nodes))
    visited_nodes.add(current_node)

    neighbor_cache = {}
    step: int = 0
    while visited_nodes != nodes and step < MAX_LIMIT:
        if current_node not in neighbor_cache:
            neighbor_cache[current_node] = list(g.neighbors(current_node))
        neighbors = neighbor_cache[current_node]
        current_node = neighbors[np.random.randint(len(neighbors))]
        
        visited_nodes.add(current_node)
        step += 1
    return step


def get_all_nodes_visit_time_and_path(g: nx.Graph, max_limit: int = MAX_LIMIT) -> tuple[np.ndarray, np.ndarray]:
    path = []
    nodes = set(g.nodes())
    visited_nodes: set[int] = set()
    current_node = np.random.choice(list(nodes))
    visited_nodes.add(current_node)
    path.append(current_node)

    neighbor_cache = {}
    step: int = 0
    while visited_nodes != nodes and step < MAX_LIMIT:
        if current_node not in neighbor_cache:
            neighbor_cache[current_node] = list(g.neighbors(current_node))
        neighbors = neighbor_cache[current_node]
        current_node = neighbors[np.random.randint(len(neighbors))]
        
        visited_nodes.add(current_node)
        path.append(current_node)
        step += 1
    return step, np.array(path)


# %%
def show_random_walk_visiting_animation(g: nx.Graph, seed: int = 1):
    steps, path = get_all_nodes_visit_time_and_path(g)
    node_colors = np.full((N_view, steps + 1), 'lightblue', dtype='<U10')
    for idx, node in enumerate(path):
        node_colors[node, idx:] = 'darkred'
        node_colors[node, idx] = 'red'
    return get_random_walk_animation(g, node_colors, seed)


# %%
save_animation(show_random_walk_visiting_animation(nx.erdos_renyi_graph(N_view, 0.8)), "visits_erdos_p08.mp4")

# %%
get_all_nodes_visit_time(nx.erdos_renyi_graph(N_view, 0.8))

