import cadquery as cq
import os
import preprocessing.utils as utils


def find_cylinders(path, box_size=10):

    # Full Model
    solids = cq.importers.importStep(path)
    solids = solids.faces()

    # Find possible lids
    shell = cq.Shell.makeShell(solids)

    solids = cq.Workplane()
    for s in shell.Shells():
        solids.add(cq.Solid.makeSolid(s))

    # Find possible lids
    lids = solids.edges("%CIRCLE").ancestors("Face").faces("%PLANE")

    # Find lids without strange geometries
    lids = lids.faces(
        cq.selectors.InverseSelector(cq.selectors.TypeSelector(("OTHER")))
    )

    # lids = solids.faces()

    aux = None

    for lid in lids.all():
        if len(lid.edges(cq.selectors.TypeSelector("LINE")).all()) == 0:
            if aux is None:
                aux = lid
            else:
                aux.add(lid)
    print(len(aux.all()))

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
    i = 0
    printer = utils.PrintPercentage(len(filtered_lids.all()), "Locating...")
    for lid in filtered_lids.all():
        cylinders.append(
            lid.edges().ancestors("Face")
        )  # Find the body of the cylinder that is connected to the lid

        origin = [0, 0, 0]
        normal = None
        for f in lid.faces():
            origin = list(f.Center().toTuple())  # Center of the top lid
            normal = list(lid.workplane().plane.zDir)

        centroid = utils.get_centroid(cylinders[-1])
        pivot = utils.translate(origin, 1, normal)

        points = []
        aux = cq.Workplane()
        for f in cylinders[-1].all():
            if len(f.wires().all()) == 1:
                aux.add(f)
            for pp in f.vertices().all():
                attachedF = pp.ancestors(
                    "Face"
                )  # Faces which one of its vertex is the current point
                if len(attachedF.all()) == len(
                    attachedF.wires().all()
                ):  # If none faces has a hole, means the point isn't in contact with a lid, so it may be a point at the height of a rib
                    for p in pp:
                        points.append(list(p.Center().toTuple()))
        traslatedPoints, newPivot = utils.translatePoints(origin, pivot, points)
        height = None
        for p in traslatedPoints:
            if height is None and p[0] != origin[0]:
                height = p[0]
            if p[0] != origin[0] and utils.distance(
                [p[0], origin[1], origin[2]], origin
            ) < utils.distance([height, origin[1], origin[2]], origin):
                height = p[0]
        if height is not None:
            newCentroid = origin.copy()
            newCentroid[0] = height
            newCentroid, _ = utils.translatePoints(
                origin, newPivot[0], [newCentroid], newPivot[1]
            )
            centroid = newCentroid[0]
        if len(aux.faces().all()) == 0:
            cylinders[-1] = [cylinders[-1], centroid]
        else:
            cylinders[-1] = [aux, centroid]
        i += 1
        printer.print(i)

    possible_heatstakes = []
    possible_heatstakes_coords = []

    heatstakes_workplane = None

    # For each cylinder, calculate its centroid and add all faces within the search box
    printer = utils.PrintPercentage(len(filtered_lids.all()), "Extracting...")
    i = 0
    for c in cylinders:
        centroid = c[1]
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
