import cadquery as cq
import time
import os
from ..utils_CQ import get_centroid

def find_cylinders(path):
    # Full Model
    solids = cq.importers.importStep(path).solids()

    # Find possible lids
    lids = solids.edges("%CIRCLE").ancestors("Face").faces("%PLANE")

    # Find lids without strange geometries
    lids = lids.faces(cq.selectors.InverseSelector(cq.selectors.TypeSelector(
        ("OTHER")
        )))

    aux = None

    for t in lids.all():
        pos = False
        if(len(t.edges(cq.selectors.TypeSelector("LINE")).all()) == 0):
            if(aux is None):
                aux = t
            else:
                aux.add(t)
    #print(len(aux.all()))

    lids = aux
    filtered_lids = None

    # Filter lids that consist of a figure with a hole
    for t in lids.all():
        if len(t.wires().all()) == 2:
            if(filtered_lids is None):
                filtered_lids = t
            else:
                filtered_lids.add(t)
    #print(len(filtered_lids.all()))


    cylinders = []              # Faces that make up the body of the cylinder

    # Find the body of the cylinder that is connected to the lid
    for t in filtered_lids.all():
        cylinders.append(t.edges().ancestors("Face"))
    #print(len(cylinders))

    data = []

    candidate_heatstakes = None
    delimitador = 10

    # For each cylinder, calculate its centroid and add all faces within the search radius
    for c in cylinders:
        centroid = get_centroid(c)
        data.append(solids.faces(cq.selectors.BoxSelector((centroid[0] - delimitador, centroid[1] -delimitador, centroid[2] -delimitador), (centroid[0] + delimitador, centroid[1] + delimitador, centroid[2] + delimitador))))
        #print(len(data))
        #print(centroid)
        if(candidate_heatstakes is None):
            candidate_heatstakes = data[-1]
        else:
            candidate_heatstakes.add(data[-1])
        # Construye tu grafo
        # for f in solids.faces(cq.selectors.BoxSelector((centroid[0] - delimitador, centroid[1] - delimitador, centroid[2] -delimitador), (centroid[0] + delimitador, centroid[1] + delimitador, centroid[2] + delimitador))).all():
        #     for v in f.vertices().all():
        #         #grafo.add(v,f)
        #         pass
    print(len(data))

    #cuerpo = None

    #for c in cylinders:
    #    if(cuerpo == None):
    #        cuerpo = c
    #    else:
    #        cuerpo.add(c)

    # Ejecuta el modelo
    # modelo.predict(data) -> 

    path = "../doors/exportaciones/p7"
    file = "/d1_e.step"
    os.makedirs(path, exist_ok=True)

    cq.exporters.export(candidate_heatstakes, path+file)