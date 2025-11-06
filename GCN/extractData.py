import cadquery as cq
import os
import pandas as pd
import sys


import math


def distancia(punto1,punto2):
    suma = 0
    for i in range(len(punto2)):
        suma += (punto1[i] - punto2[i])**2
    return suma**0.5

def direccion(punto1,punto2):
    ab = []
    aux = []
    for i in range(len(punto1)):
        ab.append(punto2[i]-punto1[i])
        aux.append(0)
    aux = distancia(ab,aux)
    # if(aux == 0):
    #     aux = 0.000000000000000001
    for i in range(len(ab)):
        ab[i] = ab[i] / aux
    return ab

def trasladar(punto, dist, dire):
    aux = [dire[0]*dist, dire[1]*dist, dire[2]*dist]    
    aux = [aux[0] + punto[0], aux[1]+ punto[1], aux[2] + punto[2]]
    return aux

def trasladarAng(ang, dist, dire):
    aux = [dire[0]*dist, dire[1]*dist, dire[2]*dist]    
    aux = [aux[0] - ang[0], aux[1]- ang[1], aux[2] - ang[2]]
    return aux

def cartesiano2esfera(punto1, punto2 = [0,0,0]):
    r = distancia(punto1, punto2)
    if(r == 0):
        r = 0.000000000000000001
    if(punto1[0] == 0):
        punto1[0] += 0.000000000000000001
    theta = abs(math.atan(punto1[1]/punto1[0]))
    if(punto1[0] < 0):        
        if(punto1[1] < 0):
            theta = math.pi + theta
        else:
            theta = math.pi - theta
    elif(punto1[1] < 0):
        theta = 2*math.pi - theta
    
    phi =  math.acos(punto1[2] / r)
    
    
    return r,theta, phi

def esfera2cartesiano(esf):
    return esf[0]*math.sin(esf[2])*math.cos(esf[1]), esf[0]*math.sin(esf[2])*math.sin(esf[1]), esf[0]*math.cos(esf[2])


def trasladePoints(origin, pivot, points, baseDir = [1,0,0]):
    originalDir = direccion(origin,pivot)
    dirPivot = cartesiano2esfera(originalDir)

    baseSph = cartesiano2esfera(baseDir)

    dif = [baseSph[1] - dirPivot[1], baseSph[2] - dirPivot[2]]
    
    newPoints = []
    
    for p in points:
        dirPoint = cartesiano2esfera(direccion(origin, p))
        dirPoint2 = [dirPoint[0], dirPoint[1]+dif[0], dirPoint[2]+dif[1]]
        newPoints.append(trasladar(origin, distancia(origin, p), esfera2cartesiano(dirPoint2)))
    return newPoints, [trasladar(origin, distancia(origin, pivot), baseDir), originalDir]

    

class PrintPercentage:
    percent = 0
    total = None
    partes = None
    
    def __init__(self, total, titulo:str = "", partes:int = 20):
        self.partes = partes
        self.total = total
        print("["+titulo+" "*(partes-2-len(titulo))+"]")
    
    def print(self, current):
        fit = (current / self.total) * self.partes
        fit = round(fit)
        if(fit > self.percent):
            sys.stdout.write("#" * (fit-self.percent))
            sys.stdout.flush()
        self.percent = fit
        if(fit == self.partes):
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
    solids = solids.faces()

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
        solids = aux
    print(len(solids.all()))
    lids = solids.faces("%PLANE")
    #lids = solids.faces()
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
        if len(lid.wires().all()) == 2:
            if(filtered_lids is None):
                filtered_lids = lid
            else:
                filtered_lids.add(lid)
    print("filtered",len(filtered_lids.all()))

    cylinders = []              # Faces that make up the body of the cylinder    
    
    i = 0
    printer = PrintPercentage(len(filtered_lids.all()), "Locating...")
    for lid in filtered_lids.all():
        cylinders.append(lid.edges().ancestors("Face"))         # Find the body of the cylinder that is connected to the lid
        
        origin = [0,0,0]
        normal = None
        for f in lid.faces():
            origin = list(f.Center().toTuple())             # Center of the top lid
            normal = list(lid.workplane().plane.zDir)
        
            
        centroid = get_centroid(cylinders[-1])
        pivot = trasladar(origin, 1, normal)
        
        points = []
        aux = cq.Workplane()
        for f in cylinders[-1].all():
            if(len(f.wires().all()) == 1):
                aux.add(f)
            # possible_lids = f.edges().ancestors("Face")
            # flag = 0
            # for pl in possible_lids.all():
            #     if(len(pl.wires().all())>1):
            #         flag += 1
            # if(flag > 1):
            #     aux.add(f)
            #     for pl in possible_lids.all():
            #         if(len(pl.wires().all()) > 1):
            #             for l in pl.wires(cq.selectors.LengthNthSelector(0)):
            #                 possiblePivot = list(l.Center().toTuple())
            #                 if(distancia(possiblePivot, origin) > 1):
            #                     pivot = possiblePivot                   # Center of the bottom lid         
            for pp in f.vertices().all():
                attachedF = pp.ancestors("Face")            # Faces which one of its vertex is the current point
                if(len(attachedF.all()) == len(attachedF.wires().all())): # If none faces has a hole, means the point isn't in contact with a lid, so it may be a point at the height of a rib
                    #print(len(attachedF.all()))
                    for p in pp:
                        points.append(list(p.Center().toTuple()))
        traslatedPoints, newPivot = trasladePoints(origin, pivot, points)
        height = None
        for p in traslatedPoints:
            if(height is None and p[0] != origin[0]):
                height = p[0]
            if(p[0 ] != origin[0] and distancia([p[0], origin[1], origin[2]], origin) < distancia([height, origin[1], origin[2]], origin)):
                height = p[0]
        if(height is not None):
            #height /= len(traslatedPoints)
            newCentroid = origin.copy()
            newCentroid[0] = height
            newCentroid, _ = trasladePoints(origin, newPivot[0], [newCentroid], newPivot[1])
            centroid = newCentroid[0]
        if(len(aux.faces().all()) == 0): cylinders[-1] = [cylinders[-1], centroid, 'n']
        else:
            cylinders[-1] = [aux, centroid, 's']
        i += 1
        printer.print(i)
            
            
        
    #print(len(cylinders))

    possible_heatstakes = []

    # heatstakes_workplane = None
    printer = PrintPercentage(len(filtered_lids.all()), "Extracting")
    i = 0
    # For each cylinder, calculate its centroid and add all faces within the search box
    for c in cylinders:        
        cylinder = c[0]
        centroid = c[1]
        if(c[2] == 'n'):
            possible_heatstakes.append([solids.faces(cq.selectors.BoxSelector((centroid[0] - box_size, centroid[1] - box_size, centroid[2] - box_size), (centroid[0] + box_size, centroid[1] + box_size, centroid[2] + box_size))), centroid, 'n'])
        else:
            #print(c[0].faces().all())
            possible_heatstakes.append(c)
        #print(len(possible_heatstakes))
        #print(centroid)
        # if(heatstakes_workplane is None):
        #     heatstakes_workplane = possible_heatstakes[-1][0]
        # else:
        #     heatstakes_workplane.add(possible_heatstakes[-1][0])
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

minDistance = 10000000
maxDistance = 0
meanDistance = 0
m = 0
n = 0

todoH = []
todoO = []
while(i != -1):
    try:
        a = open(prefix+str(i)+sufix,"r")
        a.close()
        
        centros = puntos[ puntos["Archivo"] == "A"+str(i) ]
        centros = centros[['X', 'Y', 'Z']]
        centros = centros.apply(limpiar, axis=1)
        m += centros.shape[0]
        #centros = pd.concat([centros, puntos[ puntos["Archivo"] == "A"+str(i * 2) ]])
        #print(centros)
        
        data = find_cylinders(prefix+str(i)+sufix)
        #data = find_cylinders("Heatstake.STEP")
        hstk = []
        otro = []
        a = set()
        completo = None
        for d in data:
            flag = False
            for c in range(centros.shape[0]):
                #print (d[1])
                #print(centros[['X', 'Y', 'Z']].iloc[c])
                if(abs(float(centros["X"].iloc[c]) - d[1][0]) < 5 and abs(float(centros["Y"].iloc[c]) - d[1][1]) < 5 and abs(float(centros["Z"].iloc[c]) - d[1][2]) < 5):
                    if(c not in a):
                        hstk.append(d[0])
                        dis = distancia([centros["X"].iloc[c], centros["Y"].iloc[c], centros["Z"].iloc[c]], d[1])
                        minDistance = min(minDistance, dis)
                        maxDistance = max(maxDistance, dis)
                        meanDistance += dis
                        n += 1
                        # print(d[2])
                        print("Real - predicted")
                        # print(abs(float(centros["X"].iloc[c]) - d[1][0]), abs(float(centros["Y"].iloc[c]) - d[1][1]), abs(float(centros["Z"].iloc[c]) - d[1][2]))
                        print(centros.iloc[c])
                        print(d[1])
                    flag = True
                    a.add(c)
                    break
            if(not flag):
                otro.append(d[0])  
            if(completo is None):
                completo = d[0]
            else:
                completo.add(d[0])
        todoH += hstk
        todoO += otro
        print("Found heatstakes in archive "+ str(i)+ ": ", len(hstk))
        print("Found junk in archive "+ str(i)+ ": ", len(otro))
        
        export_heatstakes(hstk, path="dataEntrv4/Puerta_"+str(i)+"/Heastakes", name="Heastake")
        export_heatstakes(otro, path="dataEntrv4/Puerta_"+str(i)+"/Otros")
        #export_heatstakes([completo], path="dataEntrv3/Puerta_"+str(i)+"/todo") 
        i += 1
    except Exception as e:
        print(e)
        i = -1
print("Minimal error", minDistance)
print("Max error", maxDistance)
print("Mean error", meanDistance/n)
print("Percentage of founded heatstakes", len(todoH)/m*100)
# export_heatstakes(todoH, path="dataEntrv4"+str(i)+"/Heastakes", name="Heastake")
# export_heatstakes(todoO, path="dataEntrv4"+str(i)+"/Otros")
