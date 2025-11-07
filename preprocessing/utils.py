import cadquery as cq
import sys
import math


def distance(point1,point2):
    sumatory = 0
    for i in range(len(point1)):
        sumatory += (point1[i] - point2[i])**2
    return sumatory**0.5

def direction(point1,point2):
    ab = []
    aux = []
    for i in range(len(point1)):
        ab.append(point2[i]-point1[i])
        aux.append(0)
    aux = distance(ab,aux)
    # if(aux == 0):
    #     aux = 0.000000000000000001
    for i in range(len(ab)):
        ab[i] = ab[i] / aux
    return ab

def translate(point, dist, dire):
    aux = [dire[0]*dist, dire[1]*dist, dire[2]*dist]    
    aux = [aux[0] + point[0], aux[1]+ point[1], aux[2] + point[2]]
    return aux

def translateAng(ang, dist, dire):
    aux = [dire[0]*dist, dire[1]*dist, dire[2]*dist]    
    aux = [aux[0] - ang[0], aux[1]- ang[1], aux[2] - ang[2]]
    return aux

def cartesian2sphere(point1, point2 = [0,0,0]):
    r = distance(point1, point2)
    if(r == 0):
        r = 0.000000000000000001
    if(point1[0] == 0):
        point1[0] += 0.000000000000000001
    theta = abs(math.atan(point1[1]/point1[0]))
    if(point1[0] < 0):        
        if(point1[1] < 0):
            theta = math.pi + theta
        else:
            theta = math.pi - theta
    elif(point1[1] < 0):
        theta = 2*math.pi - theta
    
    phi =  math.acos(point1[2] / r)
    
    
    return r,theta, phi

def sphere2cartesian(esf):
    return esf[0]*math.sin(esf[2])*math.cos(esf[1]), esf[0]*math.sin(esf[2])*math.sin(esf[1]), esf[0]*math.cos(esf[2])


def translatePoints(origin, pivot, points, baseDir = [1,0,0]):
    originalDir = direction(origin,pivot)
    dirPivot = cartesian2sphere(originalDir)

    baseSph = cartesian2sphere(baseDir)

    dif = [baseSph[1] - dirPivot[1], baseSph[2] - dirPivot[2]]
    
    newPoints = []
    
    for p in points:
        dirPoint = cartesian2sphere(direction(origin, p))
        dirPoint2 = [dirPoint[0], dirPoint[1]+dif[0], dirPoint[2]+dif[1]]
        newPoints.append(translate(origin, distance(origin, p), sphere2cartesian(dirPoint2)))
    return newPoints, [translate(origin, distance(origin, pivot), baseDir), originalDir]


def get_centroid(cuerpo: cq.Workplane):
    sumaX = 0
    sumaY = 0
    sumaZ = 0
    for v in cuerpo.vertices():
        vertex = v.Center().toTuple()
        sumaX = sumaX + vertex[0]
        sumaY = sumaY + vertex[1]
        sumaZ = sumaZ + vertex[2]
    sumaX = sumaX / len(cuerpo.vertices().all())
    sumaY = sumaY / len(cuerpo.vertices().all())
    sumaZ = sumaZ / len(cuerpo.vertices().all())
    return [sumaX, sumaY, sumaZ]


class PrintPercentage:
    percent = 0
    total = None
    parts = None
    
    def __init__(self, total, title:str = "", parts:int = 20):
        self.parts = parts
        self.total = total
        print("["+title+" "*(parts-2-len(title))+"]")
    
    def print(self, current):
        fit = (current / self.total) * self.parts
        fit = round(fit)
        if(fit > self.percent):
            sys.stdout.write("#" * (fit-self.percent))
            sys.stdout.flush()
        self.percent = fit
        if(fit == self.parts):
            print("")
