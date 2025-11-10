import cadquery as cq
import json

cad_sequence = {
    "final_name": "Cylinder",
    "parts": {
        "part_1": {
            "coordinate_system": {
                "Translation Vector": [0.0, 0.0, 0.1725]
            },
            "sketch": {
                "face_1": {
                    "loop_1": {
                        "circle_1": {
                            "Center": [0.375, 0.375],
                            "Radius": 0.375
                        }
                    }
                }
            },
            "extrusion": {
                "extrude_depth_towards_normal": 0.0,
                "extrude_depth_opposite_normal": 0.1725,
                "sketch_scale": 0.75,
                "operation": "NewBodyFeatureOperation"
            }
        }
    }
}

result = cq.Workplane("XY")

for part_name, part_data in cad_sequence["parts"].items():
    translation_vector = part_data["coordinate_system"]["Translation Vector"]
    result = result.translate(translation_vector)

    sketch_data = part_data["sketch"]
    for face_name, face_data in sketch_data.items():
        for loop_name, loop_data in face_data.items():
            for circle_name, circle_data in loop_data.items():
                center = circle_data["Center"]
                radius = circle_data["Radius"]
                result = result.circle(radius, center=center)

    extrusion_data = part_data["extrusion"]
    extrude_depth_towards_normal = extrusion_data["extrude_depth_towards_normal"]
    extrude_depth_opposite_normal = extrusion_data["extrude_depth_opposite_normal"]
    sketch_scale = extrusion_data["sketch_scale"]
    operation = extrusion_data["operation"]

    result = result.extrude(-extrude_depth_opposite_normal)

file_name = "./salidasSTL/model.json.stl"
cq.exporters.export(result, file_name)

print(f"Model saved to {file_name}")