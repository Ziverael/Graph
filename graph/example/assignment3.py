from graph.model import (
    RandomGraph,
    WattStrogatzGraph,
    LatticeRingGraph,
    BarabasiAlbertGraph,
)


def main():
    graph_erdos_renyi = RandomGraph(n=28, p=0.4)
    graph_erdos_renyi.save_graph("erdos_renyi.csv", format_="csv")
    graph_gilbert = RandomGraph(n=28, m=72)
    graph_gilbert.save_graph("gilbert.csv", format_="csv")
    graph_ring_lattice = LatticeRingGraph(n=28, k=4)
    graph_ring_lattice.save_graph("ring_lattice.csv", format_="csv")
    graph_watts_strogatz = WattStrogatzGraph(n=28, k=4, beta=0.15)
    graph_watts_strogatz.save_graph("watts_strogatz.csv", format_="csv")
    graph_barabasi_albert = BarabasiAlbertGraph(n0=10, n=30, m=6)
    graph_barabasi_albert.save_graph("barabasi_albert.csv", format_="csv")


if __name__ == "__main__":
    main()
