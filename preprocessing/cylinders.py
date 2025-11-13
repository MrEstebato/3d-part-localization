import cadquery as cq
import os
import preprocessing.utils as utils


def reconstruct_solids(figure: cq.Workplane):
    """Function to generate solids according to a workplane that has shells (closed figures)

    Args:
        figure (cq.Workplane): workplane which solids will be recover

    Returns:
        cq.Workplane: A workplane that contains the solids identified
    """
    solids = figure.faces()

    shell = cq.Shell.makeShell(solids)

    solids = cq.Workplane()
    for s in shell.Shells():
        solids.add(cq.Solid.makeSolid(s))
    return solids


# END reconstruct_solids


def remove_shortest_solid(solids: cq.Workplane):
    """Function to remove the shortest solid from a workplane with a set of solids; used to remove the sole heatstake from the archives sent for training
    Args:
        solids (cq.Workplane): workplane consisting in the solids that are going to be tested

    Returns:
        cq.Workplane: A workplane that contains all the solids but the one with fewest faces
    """
    if len(solids.all()) > 1:
        i = 1e10
        j = -1
        k = 0
        for s in solids.all():
            aux = len(s.faces().all())
            if aux < i:
                i = aux
                j = k
            k += 1
        newSolids = cq.Workplane()
        k = 0
        for s in solids.all():
            if k != j:
                newSolids.add(s)
            k += 1
        return newSolids
    return solids


# END remove_shortest_solid


def get_potential_lids(figure: cq.Workplane):
    """Function to get the possible lids of heatstakes from the door

    Args:
        figure (cq.workplane): A workplane with the door where the lids will be retrieved

    Returns:
        cq.Workplane: A worplane that contains the faces that represent the lids
    """
    # lids = figure.edges("%CIRCLE").ancestors("Face").faces("%PLANE")

    # lids = lids.faces(
    #     cq.selectors.InverseSelector(cq.selectors.TypeSelector(("OTHER")))
    # )

    lids = figure.faces()

    # aux = cq.Workplane()
    # for lid in lids.all():
    #     if len(lid.edges("%LINE").all()) == 0:
    #         aux.add(lid)
    # lids = aux

    filtered_lids = None
    for lid in lids.all():
        if len(lid.wires().all()) == 2:
            if filtered_lids is None:
                filtered_lids = lid
            else:
                filtered_lids.add(lid)
    return filtered_lids


# END get_potential_lids


def get_heatstake_centroid(lid: cq.Workplane):
    """Function that gets the centroid of a heatstake at the height of the tallest rib, if there's none, will retrieve the center of mass

    Args:
        lid (cq.Workplane): Lid of the target

    Returns:
        list[float, float, float]: coordinates (x,y,z) of the centroid
    """
    body = lid.edges().ancestors("Face")
    origin = [0, 0, 0]
    normal = [0, 0, 0]
    for f in lid.faces():
        origin = list(f.Center().toTuple())
        normal = list(lid.workplane().plane.zDir)

    centroid = None
    pivot = utils.translate(origin, 1, normal)

    nearest_point = []
    distance2point = 1e10
    for pp in body.vertices().all():
        attachedF = pp.ancestors("Face")
        if len(attachedF.all()) == len(attachedF.wires().all()):
            for p in pp:
                pAux = list(p.Center().toTuple())
                dis = utils.distance(pAux, origin)
                if dis < distance2point:
                    nearest_point = [pAux]
                    distance2point = dis

    traslated_point, new_pivot = utils.translatePoints(origin, pivot, nearest_point)
    if len(traslated_point) > 0:
        p = traslated_point[0]
        if abs(p[0] - origin[0]) > 1:
            new_centroid = origin.copy()
            new_centroid[0] = p[0]
            new_centroid, _ = utils.translatePoints(
                origin, new_pivot[0], [new_centroid], new_pivot[1]
            )
            centroid = new_centroid[0]
    if centroid is None:
        centroid = utils.get_centroid(body)
    return centroid


def extract_object(figure: cq.Workplane, centroid: list[int, int, int], box_size=10):
    """Function that gets all the faces of a figure that are insed a cube

    Args:
        figure (cq.Workplane): Workplane from where get the object
        centroid (list[int, int, int]): Center of the cube
        box_size (int, optional): Distance from the center of the cube to any of its faces. Defaults to 10.

    Returns:
        cq.Workplane: Faces of the founden object
    """
    return figure.faces(
        cq.selectors.BoxSelector(
            (centroid[0] - box_size, centroid[1] - box_size, centroid[2] - box_size),
            (centroid[0] + box_size, centroid[1] + box_size, centroid[2] + box_size),
        )
    )


# END extract_object


def find_cylinders(path, box_size=10):
    solids = cq.importers.importStep(path)

    solids = reconstruct_solids(solids)

    lids = get_potential_lids(solids)

    possible_heatstakes = []
    i = 0
    printer = utils.PrintPercentage(len(lids.all()), "Locating...")
    for lid in lids.all():
        centroid = utils.get_centroid(lid.edges().ancestors("Face"))
        possible_heatstakes.append(extract_object(solids, centroid, box_size))
        i += 1
        printer.print(i)
    # for lid in lids.all():

    #     #cylinders.append(utils.get_centroid(lid))
    #     cylinders.append(get_heatstake_centroid(lid))
    #     i += 1
    #     printer.print(i)

    # possible_heatstakes = []
    # possible_heatstakes_coords = []
    # i = 0
    # printer.restart("Extracting...")
    # for c in cylinders:
    #     possible_heatstakes.append(extract_object(solids, c, box_size))
    #     possible_heatstakes_coords.append(c)
    #     i+=1
    #     printer.print(i)
    return lids, possible_heatstakes


def evaluate_model(path, box_size=10):
    solids = cq.importers.importStep(path)

    solids = reconstruct_solids(solids)

    lids = get_potential_lids(solids)

    heatstakes = []
    coordinates = []
    i = 0
    printer = utils.PrintPercentage(len(lids.all()), "Locating...")
    for lid in lids.all():
        centroid = utils.get_centroid(lid.edges().ancestors("Face"))
        possible_heatstake = extract_object(solids, centroid, box_size)
        result = True
        # result = model.predict(possible_heatstakes)
        if result:
            heatstakes.append(possible_heatstake)
            coordinates.append(get_heatstake_centroid(lid))
        i += 1
        printer.print(i)
    return coordinates, heatstakes


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
