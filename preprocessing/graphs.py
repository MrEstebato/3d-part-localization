import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_graphs(cylinders):
    graphs = []
    for cylinder in cylinders:
        graphs.append(build_brep_graph(cylinder))
    return graphs

def build_brep_graph(heatstake):
    G = nx.Graph()

    # Vertices
    vertices = {}
    for v in heatstake.vertices().vals():
        vid = f"V{v.hashCode()}"
        vertices[vid] = v
        G.add_node(vid, type="vertex", point=v.toTuple())

    # Edges and connect to vertices
    edge_to_vertices = {}
    for e in heatstake.edges().vals():
        eid = f"E{e.hashCode()}"
        length = e.Length()
        G.add_node(eid, type="edge", length=length)

        ev_ids = [f"V{vv.hashCode()}" for vv in e.Vertices()]
        edge_to_vertices[eid] = ev_ids
        for vid in ev_ids:
            if vid in G.nodes:
                G.add_edge(eid, vid)  # edge-vertex adjacency

    # Faces and connect to edges and vertices
    face_to_edges = {}
    face_to_vertices = {}
    for f in heatstake.faces().vals():
        fid = f"F{f.hashCode()}"
        G.add_node(fid, type="face", area=f.Area())

        # Face -> edges
        fe_ids = [f"E{e.hashCode()}" for e in f.Edges()]
        face_to_edges[fid] = fe_ids
        for eid in fe_ids:
            if eid in G.nodes:
                G.add_edge(fid, eid)  # face-edge adjacency

        # Face -> vertices (union of edge vertices)
        fv_ids = set()
        for e in f.Edges():
            for vv in e.Vertices():
                fv_ids.add(f"V{vv.hashCode()}")
        face_to_vertices[fid] = fv_ids
        for vid in fv_ids:
            if vid in G.nodes:
                G.add_edge(fid, vid)  # face-vertex adjacency

    return G

def plot_graph(G: nx.Graph, with_labels=False, figsize=(9, 9)):
    """
    layout: 'spring' | 'kamada' | 'circular' | 'random'
    """
    pos = nx.spring_layout(G, seed=42)

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
