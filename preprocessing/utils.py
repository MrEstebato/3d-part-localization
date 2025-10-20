import cadquery as cq

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