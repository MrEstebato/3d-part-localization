import cadquery as cq

# Part 1: Cylinder
radius = 0.375
height = 0.1725
scale = 0.75
translation_z = 0.1725

cylinder = cq.Workplane("XY").moveTo(radius, radius).circle(radius).extrude(-height)

# Scaling is applied by modifying the radius in the circle definition
scaled_radius = radius * scale
cylinder = cq.Workplane("XY").moveTo(scaled_radius, scaled_radius).circle(scaled_radius).extrude(-height)


cylinder = cylinder.translate((0, 0, translation_z))

# Export to STL
file_name = "./salidasSTL/modeljson.stl"
cq.exporters.export(cylinder, file_name, tolerance=1e-3, angularTolerance=0.1)