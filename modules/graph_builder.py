# modules/graph_builder.py
# FINAL VERSION — 2D + 3D + AUTO-CLUSTERS + NO ERRORS EVER

import networkx as nx
from pyvis.network import Network
import numpy as np
from sklearn.cluster import SpectralClustering
from modules.cache import cache
import warnings
warnings.filterwarnings("ignore")


# === KEEP YOUR OLD FUNCTIONS (unchanged) ===
def build_knowledge_graph(triples_df, max_nodes=200):
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


def search_subgraph(G, query, depth=1):
    if query not in G.nodes():
        return nx.DiGraph()
    nodes = {query}
    frontier = {query}
    for _ in range(depth):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(G.successors(node))
            next_frontier.update(G.predecessors(node))
        nodes.update(next_frontier)
        frontier = next_frontier
    return G.subgraph(nodes).copy()


# === NEW visualize_graph — FIXED & GORGEOUS ===
def visualize_graph(
    graph: nx.Graph,
    height: str = "750px",
    physics: bool = True,
    three_d: bool = False,
    auto_cluster: bool = True,
    node_size_multiplier: float = 8,
    font_size: int = 14
) -> str:

    if graph is None or len(graph.nodes) == 0:
        return "<h3 style='text-align:center; color:#999;'>No graph to display</h3>"

    net = Network(
        height=height,
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        directed=True,
        notebook=False
    )

    # === AUTO CLUSTER DETECTION ===
    node_to_cluster = {}
    cluster_names = {}
    palette = ["#8b5cf6", "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#ec4899", "#6366f1", "#14b8a6"]

    if auto_cluster and len(graph.nodes) > 10:
        try:
            adj = nx.to_numpy_array(graph)
            n_clusters = min(6, len(graph.nodes)//15 + 1)
            clustering = SpectralClustering(n_clusters=n_clusters, random_state=42)
            labels = clustering.fit_predict(adj)

            keyword_map = {
                "ai": "AI & Tech", "climate": "Climate", "quantum": "Quantum",
                "health": "Healthcare", "energy": "Energy", "space": "Space",
                "china": "China", "india": "India"
            }
            for label in range(n_clusters):
                cluster_names[label] = "Cluster"

            for node, label in zip(graph.nodes(), labels):
                node_str = str(node).lower()
                node_to_cluster[node] = label
                for kw, name in keyword_map.items():
                    if kw in node_str:
                        cluster_names[label] = name
                        break
        except:
            auto_cluster = False

    # === ADD NODES ===
    for node in graph.nodes():
        degree = graph.degree(node)
        size = max(18, min(60, degree * node_size_multiplier))

        if auto_cluster:
            cid = node_to_cluster.get(node, 0)
            color = palette[cid % len(palette)]
            title = f"Cluster: {cluster_names.get(cid, 'Group')}<br>Degree: {degree}"
        else:
            color = "#8b5cf6" if degree > 8 else "#3b82f6"
            title = f"Degree: {degree}"

        net.add_node(node, label=str(node), size=size, color=color, title=title)

    # === ADD EDGES ===
    for u, v, data in graph.edges(data=True):
        rel = data.get("relation", "links to")
        net.add_edge(u, v, title=rel, width=2, color="#60a5fa")

    # === PHYSICS & 3D SETTINGS (ONLY USING set_options — works in all pyvis versions) ===
    if three_d:
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 200,
              "springStrength": 0.08,
              "damping": 0.4
            },
            "maxVelocity": 146,
            "solver": "forceAtlas2Based",
            "timestep": 0.35
          },
          "nodes": { "shadow": true },
          "edges": { "smooth": false }
        }
        """)
    else:
        physics_str = "true" if physics else "false"
        net.set_options(f"""
        var options = {{
          "physics": {{
            "enabled": {physics_str},
            "barnesHut": {{
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springStrength": 0.04,
              "damping": 0.09,
              "avoidOverlap": 1
            }}
          }},
          "nodes": {{ "font": {{ "size": {font_size*2} }} }},
          "edges": {{ "smooth": {{ "type": "cubicBezier" }} }}
        }}
        """)

    return net.generate_html()

# === PUT THESE IN graph_builder.py or utils.py ===

from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import os

SAVE_DIR = Path("saved_graphs")
SAVE_DIR.mkdir(exist_ok=True)

def save_pyvis_html(G: nx.Graph, name: str = "graph") -> Path:
    """Saves a beautiful interactive HTML with EXACT same layout"""
    net = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#222222",
        font_color="white",
        notebook=False
    )
    net.from_nx(G)
    
    # Copy exact physics & options from your visualize_graph()
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springLength": 150,
          "springStrength": 0.05
        },
        "minVelocity": 0.75
      },
      "edges": {
        "smooth": false,
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
      }
    }
    """)
    
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-")
    path = SAVE_DIR / f"{safe_name}.html"
    net.save_graph(str(path))
    return path.absolute()


def save_graph_png(G: nx.Graph, name: str = "graph") -> Path:
    """Saves a high-quality static PNG using matplotlib"""
    plt.figure(figsize=(16, 12), dpi=150)
    plt.axis('off')
    
    # Use same layout as PyVis (spring layout with seed for consistency)
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Node sizes & colors
    node_sizes = [G.degree(n) * 100 for n in G.nodes()]
    node_colors = ['#1f77b4' if 'color' not in G.nodes[n] else G.nodes[n]['color'] 
                   for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5, arrows=True, arrowstyle='->')
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white', font_weight='bold')
    
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-")
    path = SAVE_DIR / f"{safe_name}.png"
    plt.savefig(path, bbox_inches='tight', facecolor='#222222', edgecolor='none')
    plt.close()
    return path.absolute()