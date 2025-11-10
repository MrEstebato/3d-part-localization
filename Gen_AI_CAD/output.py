import json
import cadquery as cq

def create_cad_model_from_json(json_data):
    """
    Creates a CAD model using CadQuery based on the provided JSON instructions.

    Args:
        json_data (str or dict): A JSON string or a Python dictionary
                                 containing the CAD model instructions.

    Returns:
        cadquery.Workplane: The final CadQuery Workplane object representing the model.
    """
    if isinstance(json_data, str):
        instructions = json.loads(json_data)
    else:
        instructions = json_data

    final_assembly = cq.Workplane("XY")
    
    # Get the final name for the output file
    final_name = instructions.get("final_name", "model")

    for part_name, part_data in instructions.get("parts", {}).items():
        print(f"Processing part: {part_name}")

        # Initialize a workplane for the current part
        # Start with XY plane, then apply transformations
        part_workplane = cq.Workplane("XY")

        # --- Handle Coordinate System (Translation) ---
        coord_system = part_data.get("coordinate_system", {})
        translation_vector = coord_system.get("Translation Vector", [0.0, 0.0, 0.0])
        
        # Apply initial translation to the workplane
        part_workplane = part_workplane.workplane(offset=translation_vector[2]).center(translation_vector[0], translation_vector[1])

        # --- Handle Sketch ---
        sketch_data = part_data.get("sketch", {})
        for face_name, face_data in sketch_data.items():
            for loop_name, loop_data in face_data.items():
                for shape_type, shape_params in loop_data.items():
                    if shape_type.startswith("rectangle"):
                        corner1 = shape_params["Corner 1"]
                        corner2 = shape_params["Corner 2"]

                        # Calculate width, height, and center for the rectangle
                        width = abs(corner2[0] - corner1[0])
                        height = abs(corner2[1] - corner1[1])
                        center_x = (corner1[0] + corner2[0]) / 2.0
                        center_y = (corner1[1] + corner2[1]) / 2.0

                        # Move to the center of the rectangle and draw it
                        part_workplane = part_workplane.moveTo(center_x, center_y).rect(width, height)
                    # Add more shape types here if needed (e.g., circle, polyline)
                    else:
                        print(f"Warning: Unsupported sketch shape type '{shape_type}' for part '{part_name}'.")

        # --- Handle Extrusion ---
        extrusion_data = part_data.get("extrusion", {})
        if extrusion_data:
            extrude_depth_towards_normal = extrusion_data.get("extrude_depth_towards_normal", 0.0)
            extrude_depth_opposite_normal = extrusion_data.get("extrude_depth_opposite_normal", 0.0)
            # sketch_scale = extrusion_data.get("sketch_scale", 1.0) # Not directly used in simple extrude
            operation = extrusion_data.get("operation", "NewBodyFeatureOperation")

            # Calculate total extrusion depth
            total_extrude_depth = extrude_depth_towards_normal + extrude_depth_opposite_normal

            if total_extrude_depth > 0:
                # Extrude the sketch
                # If there's an opposite normal extrusion, we need to shift the solid
                # The default extrude goes in the positive normal direction from the sketch plane.
                # So, if opposite_normal > 0, the sketch plane should be considered at Z = opposite_normal
                # and extrude towards Z = opposite_normal + towards_normal
                # Or, extrude total_depth and then translate the solid down by opposite_normal.
                
                extruded_solid = part_workplane.extrude(total_extrude_depth)
                
                # If there's an extrusion in the opposite direction, translate the solid
                # so that the base of the 'towards_normal' extrusion is at the sketch plane.
                if extrude_depth_opposite_normal > 0:
                    extruded_solid = extruded_solid.translate((0, 0, -extrude_depth_opposite_normal))
                
                # For "NewBodyFeatureOperation", we just add the solid.
                # If there were other operations like "CutFeatureOperation", we'd use .cut()
                final_assembly = final_assembly.add(extruded_solid)
            else:
                print(f"Warning: Extrusion depth is zero for part '{part_name}'. No extrusion performed.")
        else:
            print(f"Warning: No extrusion data found for part '{part_name}'.")

    return final_assembly, final_name

# JSON data provided
json_input = """
{
  "final_name": "Square Box",
  "parts": {
    "part_1": {
      "coordinate_system": {
        "Translation Vector": [0.0, 0.0, 0.0]
      },
      "sketch": {
        "face_1": {
          "loop_1": {
            "rectangle_1": {
              "Corner 1": [0.0, 0.0],
              "Corner 2": [50.0, 50.0]
            }
          }
        }
      },
      "extrusion": {
        "extrude_depth_towards_normal": 50.0,
        "extrude_depth_opposite_normal": 0.0,
        "sketch_scale": 1.0,
        "operation": "NewBodyFeatureOperation"
      }
    }
  }
}
"""

# Create the model
result_model, output_name = create_cad_model_from_json(json_input)

# Export the model to an STP file
output_filename = f"{output_name.replace(' ', '_')}.step"
try:
    cq.exporters.export(result_model.val(), output_filename)
    print(f"\nSuccessfully created and exported '{output_filename}'")
except Exception as e:
    print(f"\nError exporting model: {e}")

# To view the model in a CadQuery GUI (like CQ-editor), you can uncomment the following:
# show_object(result_model)