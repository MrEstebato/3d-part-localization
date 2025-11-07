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


def normalize_features(graphs):
    all_feats = np.concatenate([g.graph["x"] for g in graphs], axis=0)
    mean = all_feats.mean(axis=0, keepdims=True)
    std = all_feats.std(axis=0, keepdims=True) + 1e-6

    for g in graphs:
        g.graph["x"] = (g.graph["x"] - mean) / std


def get_face_features(face):
    # type one-hot
    gtype = face.geomType()
    types = ["PLANE", "CYLINDER", "CONE", "SPHERE", "TORUS"]
    onehot = np.zeros(len(types), dtype=np.float32)
    if gtype in types:
        onehot[types.index(gtype)] = 1.0

    # normal vector
    normal_vec = face.normalAt()
    normal = np.array([normal_vec.x, normal_vec.y, normal_vec.z], dtype=np.float32)
    if np.linalg.norm(normal) > 0:
        normal = normal / np.linalg.norm(normal)

    # bounding box dims
    bbox = face.BoundingBox()
    bbox_dims = np.array([bbox.xlen, bbox.ylen, bbox.zlen], dtype=np.float32)

    return np.concatenate([onehot, normal, bbox_dims], axis=0)


def get_edge_features(edge):
    # type one-hot
    gtype = edge.geomType()
    types = ["LINE", "CIRCLE", "ELLIPSE"]
    onehot = np.zeros(len(types), dtype=np.float32)
    if gtype in types:
        onehot[types.index(gtype)] = 1.0

    # length
    length = np.array([edge.Length()], dtype=np.float32)

    # curvature approximation
    if gtype == "CIRCLE":
        r = edge.radius()
        curv = 1.0 / r if r > 0 else 0.0
    else:
        curv = 0.0
    curvature = np.array([curv], dtype=np.float32)

    return np.concatenate([onehot, length, curvature], axis=0)


def get_vertex_features(vertex, degree):
    # coordinates (x,y,z)
    coords = np.array(vertex.toTuple(), dtype=np.float32)

    # degree matters (heat stakes ribs attach to cylinder at high valence)
    degree = np.array([degree], dtype=np.float32)

    return np.concatenate([coords, degree], axis=0)


def build_brep_graph(heatstake):
    G = nx.Graph()

    # Vertices - store the actual vertex object
    vertices = {}
    for vertex in heatstake.vertices().vals():
        vid = f"V{vertex.hashCode()}"
        vertices[vid] = vertex
        G.add_node(vid, type="vertex", obj=vertex)

    # Edges - store the actual edge object
    edge_to_vertices = {}
    for edge in heatstake.edges().vals():
        eid = f"E{edge.hashCode()}"
        G.add_node(eid, type="edge", obj=edge)

        # Edge-Vertex adjacency
        ev_ids = [f"V{vertex.hashCode()}" for vertex in edge.Vertices()]
        edge_to_vertices[eid] = ev_ids
        for vid in ev_ids:
            if vid in G.nodes:
                G.add_edge(eid, vid)

    # Faces - store the actual face object
    face_to_edges = {}
    face_to_vertices = {}
    for face in heatstake.faces().vals():
        fid = f"F{face.hashCode()}"
        G.add_node(fid, type="face", obj=face)

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
    # One-Hot Encoding for node type (3 features)
    type_encoding = {
        "vertex": np.array([1, 0, 0], dtype=np.float32),
        "edge": np.array([0, 1, 0], dtype=np.float32),
        "face": np.array([0, 0, 1], dtype=np.float32),
        "other": np.array([0, 0, 0], dtype=np.float32),
    }

    # Feature dimensions:
    # - Type one-hot: 3
    # - Face features: 5 (type) + 3 (normal) + 1 (radius) + 3 (bbox) = 12
    # - Edge features: 3 (type) + 1 (length) + 1 (curvature) = 5
    # - Vertex features: 3 (coords) + 1 (degree) = 4
    # Max feature length = 12, so total = 3 + 12 = 15

    FACE_FEAT_LEN = 11
    EDGE_FEAT_LEN = 5
    VERTEX_FEAT_LEN = 4
    MAX_GEOM_FEAT_LEN = FACE_FEAT_LEN  # 11

    for G in graphs:
        node_features = []
        for node_id, attrs in G.nodes(data=True):
            node_type = attrs.get("type", "other")
            one_hot = type_encoding.get(node_type, type_encoding["other"])

            # Extract attributes for face
            if node_type == "face":
                face_obj = attrs.get("obj")
                geom_feats = get_face_features(face_obj)

            # Extract attributes for edge
            elif node_type == "edge":
                edge_obj = attrs.get("obj")
                geom_feats = get_edge_features(edge_obj)
                geom_feats = np.pad(geom_feats, (0, MAX_GEOM_FEAT_LEN - EDGE_FEAT_LEN))

            # Extract attributes for vertex
            elif node_type == "vertex":
                vertex_obj = attrs.get("obj")
                deg = G.degree(node_id)
                geom_feats = get_vertex_features(vertex_obj, deg)
                geom_feats = np.pad(
                    geom_feats, (0, MAX_GEOM_FEAT_LEN - VERTEX_FEAT_LEN)
                )

            # Other nodes
            else:
                geom_feats = np.zeros(MAX_GEOM_FEAT_LEN, dtype=np.float32)

            feat = np.concatenate([one_hot, geom_feats], axis=0)
            node_features.append(feat)

        features_array = np.vstack(node_features).astype(np.float32)
        G.graph["x"] = features_array

    return graphs


def nx_to_PyG(graphs):
    encode_graphs(graphs)
    normalize_features(graphs)
    pyg_graphs = []
    for G in graphs:
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(G.edges())

        pyg_graph = from_networkx(H)
        pyg_graph.x = torch.from_numpy(G.graph["x"]).contiguous()
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

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=v_nodes,
        node_color="#1f77b4",
        node_shape="o",
        label="vertex",
        node_size=60,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=e_nodes,
        node_color="#ff7f0e",
        node_shape="s",
        label="edge",
        node_size=70,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=f_nodes,
        node_color="#2ca02c",
        node_shape="^",
        label="face",
        node_size=80,
    )

    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=7)

    plt.axis("off")
    plt.legend(scatterpoints=1, loc="upper right")
    plt.tight_layout()
    plt.show()
