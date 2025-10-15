import cadquery as cq
import time
import os

def centroide(cuerpo: cq.Workplane):
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

# Inicializar tiempo
initial_time = time.time()

# Modelo completo
solids = cq.importers.importStep("../doors/doors1.stp").solids()

# Encontrar posibles tapas
tapas = solids.edges("%CIRCLE").ancestors("Face").faces("%PLANE")

# Encontrar tapas sin geometrías raras
tapas = tapas.faces(cq.selectors.InverseSelector(cq.selectors.TypeSelector(
    ("OTHER")
    )))

aux = None

for t in tapas.all():
    pos = False
    if(len(t.edges(cq.selectors.TypeSelector("LINE")).all()) == 0):
        if(aux is None):
            aux = t
        else:
            aux.add(t)
#print(len(aux.all()))
tapas = aux

tapas_filtradas = None

#print("")

# Filtrar tapas por aquellas que constan de una figura con un agujero
for t in tapas.all():
    if len(t.wires().all()) == 2:
        if(tapas_filtradas is None):
            tapas_filtradas = t
        else:
            tapas_filtradas.add(t)
#print(len(tapas_filtradas.all()))


cilindros = []          # Caras que componen el cuerpo del cilindro

# Busca cuerpo del cilindro que esté conectado a la tapa
for t in tapas_filtradas.all():
    cilindros.append(t.edges().ancestors("Face"))
#print(len(cilindros))

data = []

heatstakesC = None
delimitador = 10

# Por cada cilindro, calcula su centroide y agrega todas las caras dentro del radio de busqueda
for c in cilindros:
    centro = centroide(c)
    data.append(solids.faces(cq.selectors.BoxSelector((centro[0] - delimitador, centro[1] -delimitador, centro[2] -delimitador), (centro[0] + delimitador, centro[1] + delimitador, centro[2] + delimitador))))
    #print(len(data))
    #print(centro)
    if(heatstakesC is None):
        heatstakesC = data[-1]
    else:
        heatstakesC.add(data[-1])
    # Construye tu grafo
    # for f in solids.faces(cq.selectors.BoxSelector((centro[0] - delimitador, centro[1] - delimitador, centro[2] -delimitador), (centro[0] + delimitador, centro[1] + delimitador, centro[2] + delimitador))).all():
    #     for v in f.vertices().all():
    #         #grafo.add(v,f)
    #         pass
        
print(time.time()-initial_time)
print(len(data))

#cuerpo = None

#for c in cilindros:
#    if(cuerpo == None):
#        cuerpo = c
#    else:
#        cuerpo.add(c)

# Ejecuta el modelo
# modelo.predict(data) -> 

path = "../doors/exportaciones/p7"
file = "/d1_e.step"
os.makedirs(path, exist_ok=True)

cq.exporters.export(heatstakesC, path+file)