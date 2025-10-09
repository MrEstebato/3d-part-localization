from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopoDS import topods
from ..utils import vertex_to_tuple, point_inside_sphere_xyz
import networkx as nx


def create_graphs(cylinders, radius=20.0):

    graphs = []

    for cylinder in cylinders:
        (face, properties) = cylinder
        center = (
            properties["center_x"],
            properties["center_y"],
            properties["center_z"],
        )
        print("Extracting local graph around cylinder at", center)
        G, mapping = extract_local_brep_graph(face, center, radius)

        # ?
        if not mapping:
            print("  No faces found inside the sphere, skipping.")
            continue
        graphs.append((G, mapping))

    return graphs


def extract_local_brep_graph(pieze, center_xyz, r):
    """
    Build a NetworkX graph of faces fully inside sphere(center_xyz, r).
    - detect_crossing: if True, detect edges whose curves pass into the sphere (sampling) and report them (no splitting).
    Returns: (G, face_id_to_shape)
      - G: NetworkX Graph where node = face_id (int) with node attributes
      - face_id_to_shape: dict face_id -> TopoDS_Face (so you can get the original geometry later)
    """

    # Explorer for faces in the whole shape
    exp_face = TopExp_Explorer()
    exp_face.Init(pieze, TopAbs_FACE)

    G = nx.Graph()
    face_id_to_shape = {}
    edge_to_faces = {}

    while exp_face.More():
        fshape = exp_face.Current()
        face = topods.Face(fshape)

        try:
            fid = face.HashCode(1000000)
        except Exception:
            # fallback to python hash of the python object
            fid = id(face)

        verts_xyz = face_vertices_xyz(face)  # list of (x,y,z)
        if not verts_xyz:
            exp_face.Next()
            continue

        all_inside = all(point_inside_sphere_xyz(v, center_xyz, r) for v in verts_xyz)

        if not all_inside:
            exp_face.Next()
            continue

        centroid = face_centroid_from_vertices(face)
        node_attrs = {
            "centroid": centroid,
            "vertex_count": len(verts_xyz),
        }
        G.add_node(fid, **node_attrs)
        face_id_to_shape[fid] = face

        # add mapping from each edge (hash) -> face id (for adjacency)
        for edge in face_edges(face):
            try:
                eid = edge.HashCode(1000000)
            except Exception:
                eid = id(edge)
            edge_to_faces.setdefault(eid, []).append(fid)

        exp_face.Next()

    # Build adjacency by iterating edges that belong to multiple included faces
    for eid, flist in edge_to_faces.items():
        if len(flist) >= 2:
            # connect all faces that share this edge (usually 2)
            for i in range(len(flist)):
                for j in range(i + 1, len(flist)):
                    G.add_edge(flist[i], flist[j], shared_edge=eid)

    return G, face_id_to_shape


# --- B-Rep exploration helpers ----------------------------------------------


def face_vertices_xyz(face):
    """Return list of boundary vertex (x,y,z) tuples for a TopoDS_Face."""
    coords = []
    exp_v = TopExp_Explorer(face, TopAbs_VERTEX)
    while exp_v.More():
        v_shape = exp_v.Current()
        coords.append(vertex_to_tuple(v_shape))
        exp_v.Next()
    return coords


def face_edges(face):
    """Yield TopoDS_Edge objects that bound the face."""
    exp_e = TopExp_Explorer(face, TopAbs_EDGE)
    while exp_e.More():
        yield topods.Edge(exp_e.Current())
        exp_e.Next()


def face_centroid_from_vertices(face):
    """Approximate centroid as mean of boundary vertices coordinates."""
    verts = face_vertices_xyz(face)
    if not verts:
        return (0.0, 0.0, 0.0)
    sx = sum(p[0] for p in verts)
    sy = sum(p[1] for p in verts)
    sz = sum(p[2] for p in verts)
    n = len(verts)
    return (sx / n, sy / n, sz / n)
