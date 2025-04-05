from graph.model import RandomGraph


def main():
    graph_erdos_renyi = RandomGraph(n=28, p=0.4)
    graph_erdos_renyi.save_graph("erdos_renyi.csv", format_="csv")
    graph_gilbert = RandomGraph(n=28, m=72)
    graph_gilbert.save_graph("gilbert.csv", format_="csv")


if __name__ == "__main__":
    main()
