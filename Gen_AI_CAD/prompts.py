PROMPT = """You are an expert in construction of jsons that contains CAD instructions to create models STP using this structure
{
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
        "extrude_depth_towards_normal": 
        "extrude_depth_opposite_normal": ,
        "sketch_scale": ,
        "operation": "NewBodyFeatureOperation"
      }
    }
  }
}

"""


PROMPT1 = """You are an expert reading JSON files that contains instructions to create CAD models, you have to read the file and create a cadquery code in python that allows to create the STP file using the JSON instructions  """