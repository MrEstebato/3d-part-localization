from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopoDS import topods
from ..utils_OCC import vertex_to_tuple, point_inside_sphere_xyz, get_centroid
import networkx as nx

HASH_BOUND = 1_000_000


def create_graphs(piece_shape, cylinders, radius=20.0):
    """Build per-cylinder local B-Rep graphs.

    piece_shape: full part shape (TopoDS_Shape)
    cylinders: list of (face, properties) from get_cylinders
    radius: sphere radius (mm)

    Returns list of (G, payload) where payload contains shape maps.
    """
    topo = precompute_topology(piece_shape)
    graphs = []
    for face, props in cylinders:
        center = (props["center_x"], props["center_y"], props["center_z"])
        G, payload = build_local_graph(center, radius, topo, props)
        if G.number_of_nodes() == 0:
            continue
        graphs.append((G, payload))
    return graphs


def hash(shape):
    try:
        return shape.HashCode(HASH_BOUND)
    except Exception:
        return id(shape)


def precompute_topology(piece_shape):
    vertices = {}
    edges = {}
    faces = {}
    edge_to_faces = {}

    exp_face = TopExp_Explorer(piece_shape, TopAbs_FACE)
    while exp_face.More():
        fshape = topods.Face(exp_face.Current())
        fid = hash(fshape)
        # Collect face vertices
        face_vertex_ids = []
        exp_v = TopExp_Explorer(fshape, TopAbs_VERTEX)
        while exp_v.More():
            vshape = topods.Vertex(exp_v.Current())
            vid = hash(vshape)
            if vid not in vertices:
                pt = vertex_to_tuple(vshape)
                vertices[vid] = {"shape": vshape, "point": pt}
            face_vertex_ids.append(vid)
            exp_v.Next()
        # Collect face edges
        face_edge_ids = []
        exp_e = TopExp_Explorer(fshape, TopAbs_EDGE)
        while exp_e.More():
            eshape = topods.Edge(exp_e.Current())
            eid = hash(eshape)
            if eid not in edges:
                edge_vertex_ids = []
                exp_ev = TopExp_Explorer(eshape, TopAbs_VERTEX)
                while exp_ev.More():
                    vshape2 = topods.Vertex(exp_ev.Current())
                    vid2 = hash(vshape2)
                    if vid2 not in vertices:
                        pt2 = vertex_to_tuple(vshape2)
                        vertices[vid2] = {"shape": vshape2, "point": pt2}
                    edge_vertex_ids.append(vid2)
                    exp_ev.Next()
                edges[eid] = {"shape": eshape, "vertices": edge_vertex_ids}
            face_edge_ids.append(eid)
            edge_to_faces.setdefault(eid, []).append(fid)
            exp_e.Next()
        faces[fid] = {
            "shape": fshape,
            "vertices": face_vertex_ids,
            "edges": face_edge_ids,
        }
        exp_face.Next()

    # Reverse maps
    vertex_to_edges = {vid: [] for vid in vertices}
    for eid, edata in edges.items():
        for vid in edata["vertices"]:
            vertex_to_edges[vid].append(eid)
    vertex_to_faces = {vid: [] for vid in vertices}
    for fid, fdata in faces.items():
        for vid in fdata["vertices"]:
            vertex_to_faces[vid].append(fid)

    return {
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "edge_to_faces": edge_to_faces,
        "vertex_to_edges": vertex_to_edges,
        "vertex_to_faces": vertex_to_faces,
    }


def build_local_graph(center_xyz, radius, topo, cyl_props):
    vertices = topo["vertices"]
    edges = topo["edges"]
    faces = topo["faces"]

    included_vertices = {
        vid
        for vid, v in vertices.items()
        if point_inside_sphere_xyz(v["point"], center_xyz, radius)
    }
    if not included_vertices:
        return nx.Graph(), {}

    included_edges = set()
    for eid, edata in edges.items():
        vlist = edata["vertices"]
        if vlist and all(v in included_vertices for v in vlist):
            included_edges.add(eid)

    included_faces = set()
    for fid, fdata in faces.items():
        vlist = fdata["vertices"]
        if vlist and all(v in included_vertices for v in vlist):
            included_faces.add(fid)

    G = nx.Graph()

    # Add vertex nodes
    for vid in included_vertices:
        x, y, z = vertices[vid]["point"]
        G.add_node(vid, kind="vertex", x=x, y=y, z=z)

    # Add edge nodes and connect to vertices
    for eid in included_edges:
        G.add_node(eid, kind="edge")
        for v in edges[eid]["vertices"]:
            if v in included_vertices:
                G.add_edge(eid, v, relation="edge-vertex")

    # Add face nodes and connect to edges
    for fid in included_faces:
        centroid = get_centroid([vertices[v]["point"] for v in faces[fid]["vertices"]])
        G.add_node(
            fid,
            kind="face",
            centroid=centroid,
            vertex_count=len(faces[fid]["vertices"]),
        )
        for eid in faces[fid]["edges"]:
            if eid in included_edges:
                G.add_edge(fid, eid, relation="face-edge")

    # Face-face adjacency via shared edges
    for eid, flist in topo["edge_to_faces"].items():
        if eid not in included_edges:
            continue
        face_subset = [fid for fid in flist if fid in included_faces]
        if len(face_subset) >= 2:
            for i in range(len(face_subset)):
                for j in range(i + 1, len(face_subset)):
                    G.add_edge(
                        face_subset[i],
                        face_subset[j],
                        relation="face-face",
                        via_edge=eid,
                    )

    payload = {
        "faces": {fid: faces[fid]["shape"] for fid in included_faces},
        "edges": {eid: edges[eid]["shape"] for eid in included_edges},
        "vertices": {vid: vertices[vid]["shape"] for vid in included_vertices},
        "center": center_xyz,
        "cylinder_properties": cyl_props,
    }
    return G, payload
