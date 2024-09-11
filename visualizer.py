from graphviz import Digraph

#Perform a depth first search starting at the node we call trace at to gather all the nodes
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB'], "Invalid rankdir value. Use 'LR' or 'TB'."
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    for n in nodes:
        # Label nodes with just their data and gradient
        node_label = "{ data %.4f | grad %.4f }" % (n.data, n.grad)
        dot.node(name=str(id(n)), label=node_label, shape='record')
    
    for n1, n2 in edges:
        # Create edges between nodes
        dot.edge(str(id(n1)), str(id(n2)))
    
    return dot