from pathlib import Path
import shutil

k = 0

# for i in range(1, 6):
#     for filename in Path(
#         "C:/Users/A01369877/Documents/GM/3d-part-localization/GCN/training_data/dataEntrv3/dataEntrv3/Puerta_"
#         + str(i)
#         + "/Heastakes"
#     ).glob("*.step"):

#         print(f"Processing file: {filename}")
#         new_name = f"heatstake_{k}.step"
#         target = (
#             Path(
#                 "C:/Users/A01369877/Documents/GM/3d-part-localization/GCN/training_data/allheatstakes"
#             )
#             / new_name
#         )
#         shutil.move(str(filename), str(target))
#         print(f"Moved: {filename} -> {target}")
#         k += 1

k = 0
for i in range(1, 6):
    for filename in Path(
        "C:/Users/A01369877/Documents/GM/3d-part-localization/GCN/training_data/dataEntrv3/dataEntrv3/Puerta_"
        + str(i)
        + "/Otros"
    ).glob("*.step"):
        print(f"Processing file: {filename}")
        new_name = f"other_{k}.step"
        target = (
            Path(
                "C:/Users/A01369877/Documents/GM/3d-part-localization/GCN/training_data/allother"
            )
            / new_name
        )
        shutil.move(str(filename), str(target))
        print(f"Moved: {filename} -> {target}")
        k += 1
