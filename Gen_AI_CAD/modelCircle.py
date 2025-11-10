import json
import cadquery as cq
import os

def create_cad_model_from_json(json_data_string):
    """
    Creates a CAD model using CadQuery based on instructions provided in a JSON string.

    Args:
        json_data_string (str): A JSON string containing the model definition.

    Returns:
        cq.Workplane: The final CadQuery Workplane object representing the model.
    """
    data = json.loads(json_data_string)

    final_name = data["final_name"]
    parts_data = data["parts"]

    # Initialize the CadQuery model. This will hold the final combined solid.
    result_solid = None

    # Process each part defined in the JSON
    for part_name, part_info in parts_data.items():
        print(f"Processing part: {part_name}")

        # Start a new workplane for the current part's operations.
        # We'll apply translations to this workplane to position the sketch.
        wp = cq.Workplane("XY")

        # Apply translation if specified in the coordinate_system
        translation_vector = part_info["coordinate_system"].get("Translation Vector", [0.0, 0.0, 0.0])
        if translation_vector != [0.0, 0.0, 0.0]:
            # Move the workplane to the specified translation vector
            wp = wp.workplane(offset=cq.Vector(translation_vector))
            print(f"  Applied translation: {translation_vector}")

        # Process sketch definition
        sketch_info = part_info["sketch"]["face_1"]["loop_1"] # Assuming single face, single loop for simplicity

        # Currently supporting only 'rounded_rectangle' based on the example
        if "rounded_rectangle_1" in sketch_info:
            rect_params = sketch_info["rounded_rectangle_1"]
            center = rect_params.get("Center", [0.0, 0.0])
            width = rect_params["Width"]
            length = rect_params["Length"]
            radius = rect_params["CornerRadius"]

            # Create the rounded rectangle sketch on the current workplane
            # .center() moves the origin of the workplane for the next operation
            wp = wp.center(center[0], center[1]).roundedRect(width, length, radius)
            print(f"  Created rounded rectangle sketch: Center={center}, W={width}, L={length}, R={radius}")
        else:
            raise NotImplementedError(f"Sketch type not supported for part '{part_name}'. Only 'rounded_rectangle' is implemented.")

        # Process extrusion definition
        extrusion_info = part_info["extrusion"]
        extrude_towards = extrusion_info.get("extrude_depth_towards_normal", 0.0)
        extrude_opposite = extrusion_info.get("extrude_depth_opposite_normal", 0.0)
        operation = extrusion_info.get("operation", "NewBodyFeatureOperation")
        sketch_scale = extrusion_info.get("sketch_scale", 1.0) # Not used in this example, but good to parse

        # Calculate start and total amount for extrusion
        # CadQuery's extrude(amount, start) works as:
        # 'start' is the distance from the workplane to the start of the extrusion.
        # 'amount' is the total length of the extrusion.
        # If extrude_opposite is positive, the extrusion starts 'opposite' (negative) to the normal.
        # If extrude_towards is positive, the extrusion extends 'towards' (positive) the normal.

        start_depth = -extrude_opposite
        total_extrude_amount = extrude_towards + extrude_opposite

        if total_extrude_amount <= 0:
            print(f"  Warning: Part '{part_name}' has zero or negative total extrusion depth ({total_extrude_amount}). Skipping extrusion.")
            continue

        # Perform the extrusion based on the operation type
        if operation == "NewBodyFeatureOperation":
            # Create the first solid or union with existing solids
            new_part_solid = wp.extrude(amount=total_extrude_amount, start=start_depth).val()
            if result_solid is None:
                result_solid = new_part_solid
            else:
                # If a new body is specified after the first, it implies a union with the existing model
                result_solid = result_solid.union(new_part_solid)
            print(f"  Operation: NewBodyFeatureOperation. Extruded from {start_depth} to {start_depth + total_extrude_amount}.")

        elif operation == "JoinFeatureOperation":
            # Add material to the existing body
            if result_solid is None:
                raise ValueError(f"Cannot perform 'JoinFeatureOperation' for part '{part_name}' before a 'NewBodyFeatureOperation' has created a base body.")

            new_part_solid = wp.extrude(amount=total_extrude_amount, start=start_depth).val()
            result_solid = result_solid.union(new_part_solid)
            print(f"  Operation: JoinFeatureOperation. Extruded from {start_depth} to {start_depth + total_extrude_amount}.")

        elif operation == "CutFeatureOperation":
            # Remove material from the existing body
            if result_solid is None:
                raise ValueError(f"Cannot perform 'CutFeatureOperation' for part '{part_name}' before a 'NewBodyFeatureOperation' has created a base body.")

            cut_part_solid = wp.extrude(amount=total_extrude_amount, start=start_depth).val()
            result_solid = result_solid.cut(cut_part_solid)
            print(f"  Operation: CutFeatureOperation. Extruded from {start_depth} to {start_depth + total_extrude_amount}.")

        else:
            raise NotImplementedError(f"Operation '{operation}' for part '{part_name}' is not supported.")

    # Check if a final solid was created
    if result_solid is None:
        raise ValueError("No parts were processed to create a final model. Ensure at least one 'NewBodyFeatureOperation' is present and valid.")

    # Wrap the final solid in a Workplane for export
    final_model = cq.Workplane("XY").add(result_solid)
    return final_model, final_name

# --- Main execution ---
if __name__ == "__main__":
    json_input = """
    {
      "final_name": "Car Mat",
      "parts": {
        "part_1": {
          "coordinate_system": {
            "Translation Vector": [0.0, 0.0, 0.0]
          },
          "sketch": {
            "face_1": {
              "loop_1": {
                "rounded_rectangle_1": {
                  "Center": [0.0, 0.0],
                  "Width": 15.0,
                  "Length": 14.0,
                  "CornerRadius": 4.0
                }
              }
            }
          },
          "extrusion": {
            "extrude_depth_towards_normal": 3.0,
            "extrude_depth_opposite_normal": 0.0,
            "sketch_scale": 1.0,
            "operation": "NewBodyFeatureOperation"
          }
        },
        "part_2": {
          "coordinate_system": {
            "Translation Vector": [0.0, 0.0, 3.0]
          },
          "sketch": {
            "face_1": {
              "loop_1": {
                "rounded_rectangle_1": {
                  "Center": [0.0, 0.0],
                  "Width": 13.0,
                  "Length": 12.0,
                  "CornerRadius": 1.0
                }
              }
            }
          },
          "extrusion": {
            "extrude_depth_towards_normal": 0.0,
            "extrude_depth_opposite_normal": 2.0,
            "sketch_scale": 1.0,
            "operation": "CutFeatureOperation"
          }
        }
      }
    }
    """

    try:
        model, name = create_cad_model_from_json(json_input)

        output_filename = f"{name.replace(' ', '_')}.step"
        cq.exporters.export(model, output_filename)

        print(f"\nSuccessfully created and exported '{name}' to '{os.path.abspath(output_filename)}'")

    except Exception as e:
        print(f"\nAn error occurred: {e}")