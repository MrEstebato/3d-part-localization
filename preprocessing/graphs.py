import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import from_networkx

def create_graphs(cylinders):
    graphs = []
    for cylinder in cylinders:
        graphs.append(build_brep_graph(cylinder))
    return graphs

def build_brep_graph(heatstake):
    G = nx.Graph()

    # Vertices
    vertices = {}
    for vertex in heatstake.vertices().vals():
        vid = f"V{vertex.hashCode()}"
        vertices[vid] = vertex
        G.add_node(vid, type="vertex") # Add properties if needed like: point=vertex.toTuple() (x, y, z) coordinates

    # Edges 
    edge_to_vertices = {}
    for edge in heatstake.edges().vals():
        eid = f"E{edge.hashCode()}"
        G.add_node(eid, type="edge") 

        # Edge-Vertex adjacency
        ev_ids = [f"V{vertex.hashCode()}" for vertex in edge.Vertices()]
        edge_to_vertices[eid] = ev_ids
        for vid in ev_ids:
            if vid in G.nodes:
                G.add_edge(eid, vid)

    # Faces 
    face_to_edges = {}
    face_to_vertices = {}
    for face in heatstake.faces().vals():
        fid = f"F{face.hashCode()}"
        G.add_node(fid, type="face") # Add properties if needed like: area=face.Area()

        # Face-Edges adjacency
        fe_ids = [f"E{e.hashCode()}" for e in face.Edges()]
        face_to_edges[fid] = fe_ids
        for eid in fe_ids:
            if eid in G.nodes:
                G.add_edge(fid, eid)

        # Face-Vertices adjacency
        fv_ids = set()
        for edge in face.Edges():
            for vertex in edge.Vertices():
                fv_ids.add(f"V{vertex.hashCode()}")
        face_to_vertices[fid] = fv_ids
        for vid in fv_ids:
            if vid in G.nodes:
                G.add_edge(fid, vid)

    return G

def encode_graphs(graphs):
    type_encoding = {
        "vertex": np.array([1, 0, 0], dtype=np.float32),
        "edge":   np.array([0, 1, 0], dtype=np.float32),
        "face":   np.array([0, 0, 1], dtype=np.float32)
    }
    
    for G in graphs:
        node_features = []
        for node_id in G.nodes():
            node_type = G.nodes[node_id].get("type", "vertex")
            node_features.append(type_encoding[node_type])
        
        features_array = np.array(node_features, dtype=np.float32)
        G.graph["x"] = features_array
    
    return graphs

def transform_to_PyG(graphs):
    pyg_graphs = []
    for G in graphs:
        pyg_graph = from_networkx(G)
        pyg_graph.x = torch.from_numpy(G.graph["x"])
        pyg_graphs.append(pyg_graph)
    return pyg_graphs

def plot_graph(G: nx.Graph, with_labels=False, figsize=(9, 9)):
    """
    layout: 'spring' | 'kamada' | 'circular' | 'random'
    """
    pos = nx.spring_layout(G, seed=64)

    v_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "vertex"]
    e_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "edge"]
    f_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "face"]

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.35)

    nx.draw_networkx_nodes(G, pos, nodelist=v_nodes, node_color="#1f77b4", node_shape="o", label="vertex", node_size=60)
    nx.draw_networkx_nodes(G, pos, nodelist=e_nodes, node_color="#ff7f0e", node_shape="s", label="edge", node_size=70)
    nx.draw_networkx_nodes(G, pos, nodelist=f_nodes, node_color="#2ca02c", node_shape="^", label="face", node_size=80)

    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=7)

    plt.axis("off")
    plt.legend(scatterpoints=1, loc="upper right")
    plt.tight_layout()
    plt.show()