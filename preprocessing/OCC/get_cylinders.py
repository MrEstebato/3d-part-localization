import time

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Extend.TopologyUtils import TopologyExplorer


def get_cylinders(pieze):
    def analyze_cylinder_face(cylinder_id, face):
        surface_adaptor = BRepAdaptor_Surface(face, True)
        cylinder_geom = surface_adaptor.Cylinder()

        axis = cylinder_geom.Axis()
        location = axis.Location()

        # TODO define what properties we really need and use
        properties = {
            "cylinder_id": cylinder_id,
            "center_x": location.X(),
            "center_y": location.Y(),
            "center_z": location.Z(),
        }

        return properties

    start_time = time.time()

    topo_explorer = TopologyExplorer(pieze)
    cylinders = []

    for i, face in enumerate(topo_explorer.faces()):
        surface_adaptor = BRepAdaptor_Surface(face, True)

        if surface_adaptor.GetType() == GeomAbs_Cylinder:
            properties = analyze_cylinder_face(i, face)

            if properties:
                for key, value in properties.items():
                    print(f"  {key}: {value}")
                print("-" * 40)

                cylinders.append((face, properties))

    print(
        f"Obtained {len(cylinders)} cylinders in {time.time() - start_time:.3f} seconds"
    )
    return cylinders
