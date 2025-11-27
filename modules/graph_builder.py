import networkx as nx
from pyvis.network import Network
import pandas as pd
from modules.cache import cache
from io import StringIO

# -------- GRAPH CREATION -------- #

def build_knowledge_graph(triples_df, max_nodes=200):
    """Build a NetworkX knowledge graph from triples."""
    try:
        G = nx.DiGraph()

        count = 0
        for _, row in triples_df.iterrows():
            if count >= max_nodes:
                break
            G.add_edge(row["Subject"], row["Object"], relation=row["Relation"])
            count += 1

        cache.add_log("INFO", f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    except Exception as e:
        cache.add_log("ERROR", f"Graph build failed: {str(e)}")
        raise e


def get_graph_stats(G):
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G)
    }


# -------- GRAPH VISUALIZATION -------- #

def visualize_graph(G, height="600px"):
    """Return HTML representation of the graph using PyVis."""
    net = Network(height=height, width="100%", directed=True)
    net.barnes_hut()

    for node in G.nodes():
        net.add_node(node, label=node)

    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, label=data.get("relation", ""))

    return net.generate_html()


# -------- GRAPH SEARCH -------- #

def search_subgraph(G, query, depth=1):
    """Return a subgraph containing nodes within N hops from the query."""
    if query not in G.nodes():
        return nx.DiGraph()

    nodes = set([query])
    frontier = {query}

    for _ in range(depth):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(G.successors(node))
            next_frontier.update(G.predecessors(node))
        nodes.update(next_frontier)
        frontier = next_frontier

    return G.subgraph(nodes).copy()
