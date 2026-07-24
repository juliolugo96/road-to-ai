import networkx as nx
import matplotlib.pyplot as plt
 
graph = nx.Graph()
regions = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]
edges = [
    ("WA", "NT"), ("WA", "SA"),
    ("NT", "SA"), ("NT", "Q"),
    ("SA", "Q"), ("SA", "NSW"), ("SA", "V"),
    ("Q", "NSW"),
    ("NSW", "V"),
]
graph.add_nodes_from(regions)
graph.add_edges_from(edges)
 
colors = ["red", "green", "blue"]
assignment = {}
pos = nx.spring_layout(graph, seed=42)  # fixed layout between frames
 
def is_valid(node, color):
    return all(assignment.get(neighbor) != color
               for neighbor in graph.neighbors(node))
 
def draw_graph():
    plt.clf()
    node_colors = [assignment.get(node, "gray") for node in graph.nodes]
    nx.draw(graph, pos, with_labels=True,
            node_color=node_colors, node_size=2000)
    plt.pause(0.5)
 
def backtrack():
    if len(assignment) == len(graph.nodes):
        return True
    node = next(n for n in graph.nodes if n not in assignment)
    for color in colors:
        if is_valid(node, color):
            assignment[node] = color
            draw_graph()
            if backtrack():
                return True
            del assignment[node]   # undo and try the next color
            draw_graph()
    return False
 
plt.ion()
draw_graph()
backtrack()
plt.ioff()
plt.show()
