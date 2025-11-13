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

Here is another example:

{
  "final_name": "Cup",
  "parts": {
    "part_1": {
      "name": "Outer_Cylinder_Body",
      "coordinate_system": {
        "Translation Vector": [0.0, 0.0, 0.0]
      },
      "sketch": {
        "face_1": {
          "loop_1": {
            "circle_1": {
              "Center": [0.0, 0.0],
              "Radius": 0.04
            }
          }
        }
      },
      "extrusion": {
        "extrude_depth_towards_normal": 0.1,
        "extrude_depth_opposite_normal": 0.0,
        "sketch_scale": 1.0,
        "operation": "NewBodyFeatureOperation"
      }
    },
    "part_2": {
      "name": "Inner_Hollow_Cut",
      "coordinate_system": {
        "Translation Vector": [0.0, 0.0, 0.003]
      },
      "sketch": {
        "face_1": {
          "loop_1": {
            "circle_1": {
              "Center": [0.0, 0.0],
              "Radius": 0.037
            }
          }
        }
      },
      "extrusion": {
        "extrude_depth_towards_normal": 0.097,
        "extrude_depth_opposite_normal": 0.0,
        "sketch_scale": 1.0,
        "operation": "CutFeatureOperation"
      }
    },
    "part_3": {
      "name": "Handle_Top_Attachment",
      "coordinate_system": {
        "Translation Vector": [0.04, 0.0, 0.07]
      },
      "sketch": {
        "face_1": {
          "loop_1": {
            "circle_1": {
              "Center": [0.0, 0.0],
              "Radius": 0.005
            }
          }
        }
      },
      "extrusion": {
        "extrude_depth_towards_normal": 0.015,
        "extrude_depth_opposite_normal": 0.0,
        "sketch_scale": 1.0,
        "operation": "JoinFeatureOperation"
      }
    },
    "part_4": {
      "name": "Handle_Bottom_Attachment",
      "coordinate_system": {
        "Translation Vector": [0.04, 0.0, 0.03]
      },
      "sketch": {
        "face_1": {
          "loop_1": {
            "circle_1": {
              "Center": [0.0, 0.0],
              "Radius": 0.005
            }
          }
        }
      },
      "extrusion": {
        "extrude_depth_towards_normal": 0.015,
        "extrude_depth_opposite_normal": 0.0,
        "sketch_scale": 1.0,
        "operation": "JoinFeatureOperation"
      }
    },
    "part_5": {
      "name": "Handle_Connecting_Bar",
      "coordinate_system": {
        "Translation Vector": [0.055, 0.0, 0.03]
      },
      "sketch": {
        "face_1": {
          "loop_1": {
            "circle_1": {
              "Center": [0.0, 0.0],
              "Radius": 0.005
            }
          }
        }
      },
      "extrusion": {
        "extrude_depth_towards_normal": 0.04,
        "extrude_depth_opposite_normal": 0.0,
        "sketch_scale": 1.0,
        "operation": "JoinFeatureOperation"
      }
    }
  }
}

"""


PROMPT1 = """You are an expert reading JSON files that contains instructions to create CAD models, you have to read the file and create a cadquery code in python that allows to create the STP file using the JSON instructions. Only give python code in your answer and be sure to use cadquery library. Do not include any explanations or comments, only the code.
You must save the final STP file using the name indicated in the "final_name" field of the JSON data. and use .step extension.
Here is an example of cadquery code to create a cup:
import cadquery as cq

data = {
  "final_name": "Cup",
  "parts": {
    "part_1": {
      "name": "Outer_Cylinder_Body",
      "coordinate_system": {
        "Translation Vector": [0.0, 0.0, 0.0]
      },
      "sketch": {
        "face_1": {
          "loop_1": {
            "circle_1": {
              "Center": [0.0, 0.0],
              "Radius": 0.04
            }
          }
        }
      },
      "extrusion": {
        "extrude_depth_towards_normal": 0.1,
        "extrude_depth_opposite_normal": 0.0,
        "sketch_scale": 1.0,
        "operation": "NewBodyFeatureOperation"
      }
    },
    "part_2": {
      "name": "Inner_Hollow_Cut",
      "coordinate_system": {
        "Translation Vector": [0.0, 0.0, 0.003]
      },
      "sketch": {
        "face_1": {
          "loop_1": {
            "circle_1": {
              "Center": [0.0, 0.0],
              "Radius": 0.037
            }
          }
        }
      },
      "extrusion": {
        "extrude_depth_towards_normal": 0.097,
        "extrude_depth_opposite_normal": 0.0,
        "sketch_scale": 1.0,
        "operation": "CutFeatureOperation"
      }
    },
    "part_3": {
      "name": "Handle_Top_Attachment",
      "coordinate_system": {
        "Translation Vector": [0.04, 0.0, 0.07]
      },
      "sketch": {
        "face_1": {
          "loop_1": {
            "circle_1": {
              "Center": [0.0, 0.0],
              "Radius": 0.005
            }
          }
        }
      },
      "extrusion": {
        "extrude_depth_towards_normal": 0.015,
        "extrude_depth_opposite_normal": 0.0,
        "sketch_scale": 1.0,
        "operation": "JoinFeatureOperation"
      }
    },
    "part_4": {
      "name": "Handle_Bottom_Attachment",
      "coordinate_system": {
        "Translation Vector": [0.04, 0.0, 0.03]
      },
      "sketch": {
        "face_1": {
          "loop_1": {
            "circle_1": {
              "Center": [0.0, 0.0],
              "Radius": 0.005
            }
          }
        }
      },
      "extrusion": {
        "extrude_depth_towards_normal": 0.015,
        "extrude_depth_opposite_normal": 0.0,
        "sketch_scale": 1.0,
        "operation": "JoinFeatureOperation"
      }
    },
    "part_5": {
      "name": "Handle_Connecting_Bar",
      "coordinate_system": {
        "Translation Vector": [0.055, 0.0, 0.03]
      },
      "sketch": {
        "face_1": {
          "loop_1": {
            "circle_1": {
              "Center": [0.0, 0.0],
              "Radius": 0.005
            }
          }
        }
      },
      "extrusion": {
        "extrude_depth_towards_normal": 0.04,
        "extrude_depth_opposite_normal": 0.0,
        "sketch_scale": 1.0,
        "operation": "JoinFeatureOperation"
      }
    }
  }
}

result = cq.Workplane("XY")

part_1_data = data["parts"]["part_1"]
part_1_translation = part_1_data["coordinate_system"]["Translation Vector"]
part_1_radius = part_1_data["sketch"]["face_1"]["loop_1"]["circle_1"]["Radius"]
part_1_extrude_depth = part_1_data["extrusion"]["extrude_depth_towards_normal"]
result = result.circle(part_1_radius).extrude(part_1_extrude_depth)

part_2_data = data["parts"]["part_2"]
part_2_translation = part_2_data["coordinate_system"]["Translation Vector"]
part_2_radius = part_2_data["sketch"]["face_1"]["loop_1"]["circle_1"]["Radius"]
part_2_extrude_depth = part_2_data["extrusion"]["extrude_depth_towards_normal"]
cut_body = cq.Workplane("XY").workplane(offset=part_2_translation[2]).circle(part_2_radius).extrude(part_2_extrude_depth)
result = result.cut(cut_body)

part_3_data = data["parts"]["part_3"]
part_3_translation = part_3_data["coordinate_system"]["Translation Vector"]
part_3_radius = part_3_data["sketch"]["face_1"]["loop_1"]["circle_1"]["Radius"]
part_3_extrude_depth = part_3_data["extrusion"]["extrude_depth_towards_normal"]
handle_top = cq.Workplane("YZ").workplane(offset=part_3_translation[0]).center(part_3_translation[1], part_3_translation[2]).circle(part_3_radius).extrude(part_3_extrude_depth)
result = result.union(handle_top)

part_4_data = data["parts"]["part_4"]
part_4_translation = part_4_data["coordinate_system"]["Translation Vector"]
part_4_radius = part_4_data["sketch"]["face_1"]["loop_1"]["circle_1"]["Radius"]
part_4_extrude_depth = part_4_data["extrusion"]["extrude_depth_towards_normal"]
handle_bottom = cq.Workplane("YZ").workplane(offset=part_4_translation[0]).center(part_4_translation[1], part_4_translation[2]).circle(part_4_radius).extrude(part_4_extrude_depth)
result = result.union(handle_bottom)

part_5_data = data["parts"]["part_5"]
part_5_translation = part_5_data["coordinate_system"]["Translation Vector"]
part_5_radius = part_5_data["sketch"]["face_1"]["loop_1"]["circle_1"]["Radius"]
part_5_extrude_depth = part_5_data["extrusion"]["extrude_depth_towards_normal"]
handle_bar = cq.Workplane("XY").workplane(offset=part_5_translation[2]).center(part_5_translation[0], part_5_translation[1]).circle(part_5_radius).extrude(part_5_extrude_depth)
result = result.union(handle_bar)

cq.exporters.export(result.val(), data["final_name"] + ".step")
"""