import time

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Extend.TopologyUtils import TopologyExplorer


def get_cylinders(pieze):
    def analyze_cylinder_face(cylinder_id, face):
        surface_adaptor = BRepAdaptor_Surface(face, True)
        cyl = surface_adaptor.Cylinder()
        axis = cyl.Axis()
        loc = axis.Location()
        properties = {
            "cylinder_id": cylinder_id,
            "center_x": loc.X(),
            "center_y": loc.Y(),
            "center_z": loc.Z(),
        }
        return properties

    topo_explorer = TopologyExplorer(pieze)
    cylinders = []

    for i, face in enumerate(topo_explorer.faces()):
        surface_adaptor = BRepAdaptor_Surface(face, True)

        if surface_adaptor.GetType() == GeomAbs_Cylinder:
            properties = analyze_cylinder_face(i, face)

            if properties:
                """print(
                    f"Cylinder {i}: center=({properties['center_x']:.2f}, {properties['center_y']:.2f}, {properties['center_z']:.2f})"
                )"""

                cylinders.append((face, properties))

    return cylinders
