from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool


# Load Step File
def load_step(path):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(path)
    step_reader.TransferRoots()
    pieze = step_reader.OneShape()

    if status != 1:
        raise Exception(f"Error reading STEP file: {path}")

    return pieze


# Helper functions for geometry
def vertex_to_tuple(vtx):
    """Return (x,y,z) tuple from a TopoDS_Vertex."""
    v = topods.Vertex(vtx)  # ensure vertex type
    p = BRep_Tool.Pnt(v)  # returns gp_Pnt
    return (p.X(), p.Y(), p.Z())


def point_inside_sphere_xyz(pt_xyz, center_xyz, r):
    dx = pt_xyz[0] - center_xyz[0]
    dy = pt_xyz[1] - center_xyz[1]
    dz = pt_xyz[2] - center_xyz[2]
    return (dx * dx + dy * dy + dz * dz) <= (r * r + 1e-12)


# Visualization helper
def display_graph(G, mapping):
    pass
