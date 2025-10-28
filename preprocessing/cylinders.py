import cadquery as cq
import os
from preprocessing.utils import get_centroid, PrintPercentage


def find_cylinders(path, box_size=10):
    # Full Model
    solids = cq.importers.importStep(path).solids()
    # Find possible lids
    lids = solids.edges("%CIRCLE").ancestors("Face").faces("%PLANE")

    # Find lids without strange geometries
    lids = lids.faces(
        cq.selectors.InverseSelector(cq.selectors.TypeSelector(("OTHER")))
    )

    aux = None

    for lid in lids.all():
        if len(lid.edges(cq.selectors.TypeSelector("LINE")).all()) == 0:
            if aux is None:
                aux = lid
            else:
                aux.add(lid)
    # print(len(aux.all()))

    lids = aux
    filtered_lids = None

    # Filter lids that consist of a figure with a hole
    for lid in lids.all():
        if len(lid.wires().all()) == 2:
            if filtered_lids is None:
                filtered_lids = lid
            else:
                filtered_lids.add(lid)
    # print(len(filtered_lids.all()))

    cylinders = []  # Faces that make up the body of the cylinder

    # Find the body of the cylinder that is connected to the lid
    for lid in filtered_lids.all():
        cylinders.append(lid.edges().ancestors("Face"))
    # print(len(cylinders))

    possible_heatstakes = []
    possible_heatstakes_coords = []

    heatstakes_workplane = None

    # For each cylinder, calculate its centroid and add all faces within the search box
    printer = PrintPercentage(len(filtered_lids.all()))
    i = 0
    for cylinder in cylinders:
        centroid = get_centroid(cylinder)
        possible_heatstakes.append(
            solids.faces(
                cq.selectors.BoxSelector(
                    (
                        centroid[0] - box_size,
                        centroid[1] - box_size,
                        centroid[2] - box_size,
                    ),
                    (
                        centroid[0] + box_size,
                        centroid[1] + box_size,
                        centroid[2] + box_size,
                    ),
                )
            )
        )
        # print(len(possible_heatstakes))
        # print(centroid)
        possible_heatstakes_coords.append(centroid)

        if heatstakes_workplane is None:
            heatstakes_workplane = possible_heatstakes[-1]
        else:
            heatstakes_workplane.add(possible_heatstakes[-1])
        printer.print(i)
        i += 1
    # print(len(possible_heatstakes))

    return possible_heatstakes_coords, possible_heatstakes


def export_heatstakes(
    heatstakes_workplane: list[cq.Workplane],
    path="../doors/exportaciones",
    name="cuerpo",
):
    os.makedirs(path, exist_ok=True)

    for i in range(len(heatstakes_workplane)):
        cq.exporters.export(
            heatstakes_workplane[i], path + "/" + name + str(i) + ".step"
        )


# objects = find_cylinders("../doors/doors3.stp")
# print(len(objects))

# export_heatstakes(objects, "../doors/exportaciones/p5")
