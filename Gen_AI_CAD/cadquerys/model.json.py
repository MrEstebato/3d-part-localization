import cadquery as cq

# Parámetros
translation_vector = (0.0, 0.0, 0.1725)
circle_center = (0.375, 0.375)
circle_radius = 0.375
extrude_depth_opposite_normal = 0.1725
sketch_scale = 0.75
output_file = "./salidasSTL/model.json.stl"

# Crear el cilindro
wp = (
    cq.Workplane("XY")
    .transformed(offset=cq.Vector(*translation_vector))
    .moveTo(circle_center[0], circle_center[1])
    .circle(circle_radius)
)

result = wp.extrude(-extrude_depth_opposite_normal)

# Escalar el sólido
scaled_result = result.val().scale(sketch_scale)

# Convertir el sólido escalado en un nuevo Workplane
scaled_wp = cq.Workplane(obj=scaled_result)


cq.exporters.export(scaled_wp, "./salidasSTL/model_scaled.step")
print("✅ Archivo STEP exportado. Ábrelo en FreeCAD para visualizarlo.")