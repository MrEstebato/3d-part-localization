
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


def calcAng(point):
    if(point[0] == 0):
        if(point[1] > 0):
            return math.pi/2
        else:
            return 3*math.pi/2
        
    ang = abs(math.atan(point[1] / point[0]))
    if(point[0] > 0):
        if(point[1] < 0):
            ang = 2*math.pi - ang
    else:
        if(point[1] > 0):
            ang = math.pi - ang
        else:
            ang = math.pi + ang
    return ang

def calcAng2Destiny(point1, point2):
    if(point1 == [0,0]):
        return 0, 0
    ang = math.asin(point2[1]/distance([0,0], point1))
    if(ang < 0):
        ang = 2*math.pi + ang
    angO = calcAng(point1)
    difAng = ang - angO
    return difAng, angO

def rotate(point, ang):
    mag = distance([0,0], point)
    return [mag*math.cos(ang), mag*math.sin(ang)]

def translatePoints(origin, pivot, points, baseDir = [1,0,0]):
    newPivot = translate(origin, distance(origin, pivot), baseDir)
    
    originalDir = direction(origin,pivot)
    
    b = [pivot[0] - origin[0], pivot[1] - origin[1], pivot[2] - origin[2]]
    b2 = [newPivot[0] - origin[0], newPivot[1] - origin[1], newPivot[2] - origin[2]]
    
    angY = calcAng2Destiny([b[0], b[2]], [b2[0], b2[2]])
    planeY = rotate([b[0], b[2]], sum(angY))

    angZ = calcAng2Destiny([planeY[0], b[1]], [b2[0], b2[1]])
    
    newPoints = []
    
    for p in points:
        c = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]]
        angYC = calcAng([c[0], c[2]])
        planeYC = rotate([c[0], c[2]], angYC + angY[0])

        angZC = calcAng([planeYC[0], c[1]])
        planeZC = rotate([planeYC[0], c[1]], angZC + angZ[0])
        aux = [planeZC[0], planeZC[1], planeYC[1]]
        aux = [aux[0] + origin[0], aux[1] + origin[1], aux[2] + origin[2]]

        newPoints.append(aux)
    return newPoints, [translate(origin, distance(origin, pivot), baseDir), originalDir]



def get_centroid(cuerpo: cq.Workplane):
    sumatoryX = 0
    sumatoryY = 0
    sumatoryZ = 0
    for v in cuerpo.vertices():
        vertex = v.Center().toTuple()
        sumatoryX = sumatoryX + vertex[0]
        sumatoryY = sumatoryY + vertex[1]
        sumatoryZ = sumatoryZ + vertex[2]
    sumatoryX = sumatoryX / len(cuerpo.vertices().all())
    sumatoryY = sumatoryY / len(cuerpo.vertices().all())
    sumatoryZ = sumatoryZ / len(cuerpo.vertices().all())
    return [sumatoryX, sumatoryY, sumatoryZ]


class PrintPercentage:
    percent = 0
    total = None
    parts = None
    title = None
    
    def __init__(self, total, title:str = "", parts:int = 20):
        self.parts = parts
        self.total = total
        self.title = title
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
    
    def restart(self, title = None):
        if title == None:
            title = self.title
        else:
            self.title = title
        print("["+title+" "*(self.parts-2-len(title))+"]")
        self.percent = 0
