"""
Example from Assignment 1
"""

from graph.base import Graph, Vertex, VertexName

EDGES_PARAMS = [
    (Vertex(name=VertexName("Alice")), Vertex(name=VertexName("Bob"))),
    (Vertex(name=VertexName("Bob")), Vertex(name=VertexName("Gail"))),
    (Vertex(name=VertexName("Irene")), Vertex(name=VertexName("Gail"))),
    (Vertex(name=VertexName("Carl")), Vertex(name=VertexName("Alice"))),
    (Vertex(name=VertexName("Gail")), Vertex(name=VertexName("Harry"))),
    (Vertex(name=VertexName("Irene")), Vertex(name=VertexName("Jen"))),
    (Vertex(name=VertexName("Alice")), Vertex(name=VertexName("David"))),
    (Vertex(name=VertexName("Harry")), Vertex(name=VertexName("Jen"))),
    (Vertex(name=VertexName("Ernst")), Vertex(name=VertexName("Frank"))),
    (Vertex(name=VertexName("Alice")), Vertex(name=VertexName("Ernst"))),
    (Vertex(name=VertexName("Jen")), Vertex(name=VertexName("Gail"))),
    (Vertex(name=VertexName("David")), Vertex(name=VertexName("Carl"))),
    (Vertex(name=VertexName("Alice")), Vertex(name=VertexName("Frank"))),
    (Vertex(name=VertexName("Harry")), Vertex(name=VertexName("Irene"))),
    (Vertex(name=VertexName("Carl")), Vertex(name=VertexName("Frank"))),
]


def main():
    g = Graph()
    g.add_edges_from_list(EDGES_PARAMS)
    print("Alice's friends:")
    print(g.get_neighbors(VertexName("Alice")))
    print("Is graph weighted?", g.is_weighted)
    print("Jen in graph?", Vertex(name=VertexName("Jen")) in g)
    print("Huan in graph?", Vertex(name=VertexName("Huan")) in g)
    g.save_graph("graph.dot")
    print("Graph saved to graph.dot")
    print("Shortest paths from Alice:")
    print(g.get_shortest_paths(Vertex(name=VertexName("Alice"))))


if __name__ == "__main__":
    main()
