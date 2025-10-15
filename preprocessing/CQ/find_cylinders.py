import cadquery as cq

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

# Modelo completo
solids = cq.importers.importStep("Heatstake_solo.step").solids().all()[0]

# Encontrar posibles tapas
tapas = solids.edges("%CIRCLE").ancestors("Face").faces("%PLANE")

# Encontrar tapas sin geometrías raras
tapas = tapas.faces(cq.selectors.InverseSelector(cq.selectors.TypeSelector("OTHER")))

tapas_filtradas = None

print("")

# Filtrar tapas por aquellas que constan de una figura con un agujero
for t in tapas.all():
    if len(t.wires().all()) == 2:
        if(tapas_filtradas is None):
            tapas_filtradas = t
        else:
            tapas_filtradas.add(t)
print(len(tapas_filtradas.all()))


cilindros = []          # Caras que componen el cuerpo del cilindro

# Busca cuerpo del cilindro que esté conectado a la tapa
for t in tapas_filtradas.all():
    cilindros.append(t.edges().ancestors("Face").faces(cq.selectors.InverseSelector(cq.selectors.TypeSelector("CIRCLE"))))
print(len(cilindros))

data = []

# Por cada cilindro, calcula su centroide y agrega todas las caras dentro del radio de busqueda
for c in cilindros:
    centro = centroide(c)
    data.append(solids.faces(cq.selectors.BoxSelector((centro[0] - 5, centro[1] - 5, centro[2] -5), (centro[0] + 5, centro[1] + 5, centro[2] + 5))))
    # Construye tu grafo
    for f in solids.faces(cq.selectors.BoxSelector((centro[0] - 5, centro[1] - 5, centro[2] -5), (centro[0] + 5, centro[1] + 5, centro[2] + 5))).all():
        for v in f.vertices().all():
            #grafo.add(v,f)
            pass
                
print(len(data))

#cuerpo = None

#for c in cilindros:
#    if(cuerpo == None):
#        cuerpo = c
#    else:
#        cuerpo.add(c)

# Ejecuta el modelo
# modelo.predict(data) -> 

#cq.exporters.export(cuerpo,"hts_solo_pr1.step")
