import cadquery as cq
import os
from ..utils_CQ import get_centroid

def find_cylinders(path, box_size=10):
    # Full Model
    solids = cq.importers.importStep(path).solids()

    # Find possible lids
    lids = solids.edges("%CIRCLE").ancestors("Face").faces("%PLANE")

    # Find lids without strange geometries
    lids = lids.faces(cq.selectors.InverseSelector(cq.selectors.TypeSelector(("OTHER"))))

    aux = None

    for lid in lids.all():
        if(len(lid.edges(cq.selectors.TypeSelector("LINE")).all()) == 0):
            if(aux is None):
                aux = lid
            else:
                aux.add(lid)
    #print(len(aux.all()))

    lids = aux
    filtered_lids = None

    # Filter lids that consist of a figure with a hole
    for lid in lids.all():
        if len(lid.wires().all()) == 2:
            if(filtered_lids is None):
                filtered_lids = lid
            else:
                filtered_lids.add(lid)
    #print(len(filtered_lids.all()))


    cylinders = []              # Faces that make up the body of the cylinder

    # Find the body of the cylinder that is connected to the lid
    for lid in filtered_lids.all():
        cylinders.append(lid.edges().ancestors("Face"))
    #print(len(cylinders))

    possible_heatstakes = []

    heatstakes_workplane = None

    # For each cylinder, calculate its centroid and add all faces within the search box
    for cylinder in cylinders:
        centroid = get_centroid(cylinder)
        possible_heatstakes.append(solids.faces(cq.selectors.BoxSelector((centroid[0] - box_size, centroid[1] - box_size, centroid[2] - box_size), (centroid[0] + box_size, centroid[1] + box_size, centroid[2] + box_size))))
        #print(len(possible_heatstakes))
        #print(centroid)
        if(heatstakes_workplane is None):
            heatstakes_workplane = possible_heatstakes[-1]
        else:
            heatstakes_workplane.add(possible_heatstakes[-1])
        # Construye tu grafo
        # for f in solids.faces(cq.selectors.BoxSelector((centroid[0] - delimitador, centroid[1] - delimitador, centroid[2] -delimitador), (centroid[0] + delimitador, centroid[1] + delimitador, centroid[2] + delimitador))).all():
        #     for v in f.vertices().all():
        #         #grafo.add(v,f)
        #         pass
    #print(len(possible_heatstakes))

    return possible_heatstakes

    #cuerpo = None

    #for cylinder in cylinders:
    #    if(cuerpo == None):
    #        cuerpo = cylinder
    #    else:
    #        cuerpo.add(cylinder)

    # Ejecuta el modelo
    # modelo.predict(possible_heatstakes) -> 

def export_heatstakes(heatstakes_workplane):
    path = "../doors/exportaciones/p7"
    file = "/d1_e.step"
    os.makedirs(path, exist_ok=True)

    cq.exporters.export(heatstakes_workplane, path+file)