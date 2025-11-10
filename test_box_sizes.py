import torch
from preprocessing.cylinders import find_cylinders
from preprocessing.graphs import create_graphs, nx_to_PyG
from torch_geometric.loader import DataLoader
from GCN.GCN import GCN3
import csv

# CONSTANTS
PATH_TO_STEP_FILE = "doors/doors4.STEP"  # Path to the STEP file to be processed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN3(feature_dim_size=14).to(device)
state_dict = torch.load(
    "GCN/good_heatstake_classifier.pth", map_location=device, weights_only=False
)
model.load_state_dict(state_dict)
model.eval()

with open("box_size_count_door_4.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Box Size (mm)", "Heatstake Count"])

for i in range(4, 20):
    BOX_SIZE = i + 1
    print(f"Testing box size: {BOX_SIZE} mm")
    try:
        cylinder_coords, cylinders = find_cylinders(PATH_TO_STEP_FILE, BOX_SIZE)
        cylinder_graphs = create_graphs(cylinders)
        PyG_graphs = nx_to_PyG(cylinder_graphs)

        loader = DataLoader(PyG_graphs, batch_size=len(PyG_graphs), shuffle=False)
        batch = next(iter(loader)).to(device)
        with torch.no_grad():
            logits = model(features=batch.x, adj=batch.edge_index, batch=batch.batch)
            preds = logits.argmax(dim=1).cpu().tolist()

        heatstake_coords = []
        for i, p in enumerate(preds):
            if p == 1:
                heatstake_coords.append(cylinder_coords[i])
        print(f"Finished box size: {BOX_SIZE} mm")
        print(f"Number of heatstake coordinates: {len(heatstake_coords)}")

        with open("box_size_count_door_4.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([BOX_SIZE, len(heatstake_coords)])
    except Exception as e:
        with open("box_size_count_door_4.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([BOX_SIZE, "0"])
