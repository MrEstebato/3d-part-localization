import cadquery as cq
import os
import pandas as pd
import sys

class PrintPercentage:
    percent = 0
    total = None
    
    def __init__(self, total):
        self.total = total
    
    def print(self, current):
        fit = (current / self.total) * 20
        fit = round(fit)
        if(fit > self.percent):
            sys.stdout.write("#" * (fit-self.percent))
            sys.stdout.flush()
        self.percent = fit
        if(fit == 20):
            print("")
    
    

def get_centroid(cuerpo: cq.Workplane):
    sumaX = 0
    sumaY = 0
    sumaZ = 0
    areaTotal = 0
    for f in cuerpo.faces():
        vertex = f.Center().toTuple()
        vertex = list(vertex)
        vertex[0] *= f.Area() / 1000
        vertex[1] *= f.Area() / 1000
        vertex[2] *= f.Area() / 1000
        areaTotal += f.Area() / 1000
        sumaX = sumaX + vertex[0]
        sumaY = sumaY + vertex[1]
        sumaZ = sumaZ + vertex[2]
    sumaX = sumaX / (areaTotal)
    sumaY = sumaY / (areaTotal)
    sumaZ = sumaZ / (areaTotal)
    return [sumaX, sumaY, sumaZ]

# def get_centroid(cuerpo: cq.Workplane):
#     sumaX = 0
#     sumaY = 0
#     sumaZ = 0
#     for v in cuerpo.vertices():
#         vertex = v.Center().toTuple()
#         sumaX = sumaX + vertex[0]
#         sumaY = sumaY + vertex[1]
#         sumaZ = sumaZ + vertex[2]
#     sumaX = sumaX / len(cuerpo.vertices().all())
#     sumaY = sumaY / len(cuerpo.vertices().all())
#     sumaZ = sumaZ / len(cuerpo.vertices().all())
#     return [sumaX, sumaY, sumaZ]

def find_cylinders(path, box_size=10):
    # Full Model
    solids = cq.importers.importStep(path)
    # print("f", len(solids.faces().all()))
    # print("s", len(solids.solids().faces().all()))
    solids = solids.faces()
    # try:
    #     solids = solids[0].add(solids[1])
    # except:
    #     solids = solids[0]

    # Find possible lids
    shell = cq.Shell.makeShell(solids)

    solids = cq.Workplane()

    for s in shell.Shells():
        solids.add(cq.Solid.makeSolid(s))
    
    if(len(solids.all()) > 2):
        i = 100000
        j = -1
        k = 0
        for s in solids.all():
            aux = len(s.faces().all())
            if(aux < i):
                i = aux
                j = k
            k = k+1
        aux = cq.Workplane()
        k = 0
        for s in solids.all():
            if(k != j):
                aux.add(s)
            k += 1
    
    print(len(solids.all()))
        
    lids = solids.faces("%PLANE")
    print("lids", len(lids.all()))

    # Find lids without strange geometries
    #lids = lids.faces(cq.selectors.InverseSelector(cq.selectors.TypeSelector(("OTHER"))))

    aux = None

    # for lid in lids.all():
    #     if(len(lid.edges(cq.selectors.TypeSelector("LINE")).all()) == 0):
    #         if(aux is None):
    #             aux = lid
    #         else:
    #             aux.add(lid)
    #print(len(aux.all()))
    aux = lids
    lids = aux
    filtered_lids = None

    # Filter lids that consist of a figure with a hole
    for lid in lids.all():
        if len(lid.wires().all()) > 1:
            if(filtered_lids is None):
                filtered_lids = lid
            else:
                filtered_lids.add(lid)
    print("filtered",len(filtered_lids.all()))


    cylinders = []              # Faces that make up the body of the cylinder
    printer = PrintPercentage(len(filtered_lids.all()))
    title = "[Extracting..."
    print(title + " "*(19-len(title))+ "]")
    # Find the body of the cylinder that is connected to the lid
    i = 0
    for lid in filtered_lids.all():
        cylinders.append(lid.edges().ancestors("Face"))
        
    #print(len(cylinders))

    possible_heatstakes = []

    heatstakes_workplane = None

    # For each cylinder, calculate its centroid and add all faces within the search box
    for cylinder in cylinders:
        centroid = get_centroid(cylinder)
        possible_heatstakes.append([solids.faces(cq.selectors.BoxSelector((centroid[0] - box_size, centroid[1] - box_size, centroid[2] - box_size), (centroid[0] + box_size, centroid[1] + box_size, centroid[2] + box_size))), centroid])
        #print(len(possible_heatstakes))
        #print(centroid)
        if(heatstakes_workplane is None):
            heatstakes_workplane = possible_heatstakes[-1][0]
        else:
            heatstakes_workplane.add(possible_heatstakes[-1][0])
        i += 1
        printer.print(i)
    return possible_heatstakes

def export_heatstakes(heatstakes_workplane: list[cq.Workplane], path: str, name: str = "cuerpo"):
    os.makedirs(path, exist_ok=True)
    i = 0
    for cuerpo in heatstakes_workplane:
        cq.exporters.export(cuerpo, path+"/"+name+str(i)+".step")
        i+=1


def limpiar(data):
    for k in data.keys():
        if(type(data[k]) == str):
            data[k] = data[k].replace(')','')
            data[k] = float(data[k])
    return data

prefix = "Proyecto 1/EXERCISE "
sufix =  " -DOOR PANELS.stp"

i = 1

puntos = pd.read_csv("ATTACHMENTS COORDINATES - BASE.csv")

while(i != -1):
    try:
        a = open(prefix+str(i)+sufix,"r")
        a.close()
        
        centros = puntos[ puntos["Archivo"] == "A"+str(i) ]
        centros = centros[['X', 'Y', 'Z']]
        centros = centros.apply(limpiar, axis=1)
        #centros = pd.concat([centros, puntos[ puntos["Archivo"] == "A"+str(i * 2) ]])
        #print(centros)
        
        data = find_cylinders(prefix+str(i)+sufix)
        print(len(data))
        hstk = []
        otro = []
        a = set()
        completo = None
        for d in data:
            flag = False
            for c in range(centros.shape[0]):
                #print (d[1])
                #print(centros[['X', 'Y', 'Z']].iloc[c])
                if(abs(float(centros["X"].iloc[c]) - d[1][0]) < 10 and abs(float(centros["Y"].iloc[c]) - d[1][1]) < 10 and abs(float(centros["Z"].iloc[c]) - d[1][2]) < 10):
                    if(c not in a):
                        hstk.append(d[0])
                    flag = True
                    a.add(c)
                    break
            if(not flag):
                otro.append(d[0])  
            if(completo is None):
                completo = d[0]
            else:
                completo.add(d[0])
        print("Found heatstakes in archive "+ str(i)+ ": ", len(hstk))
        print("Found junk in archive "+ str(i)+ ": ", len(otro))
        
        export_heatstakes(hstk, path="dataEntrv3/Puerta_"+str(i)+"/Heastakes", name="Heastake")
        export_heatstakes(otro, path="dataEntrv3/Puerta_"+str(i)+"/Otros")
        #export_heatstakes([completo], path="dataEntrv3/Puerta_"+str(i)+"/todo") 
        i += 1
    except Exception as e:
        print(e)
        i = -1
