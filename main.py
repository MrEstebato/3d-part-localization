import torch
from preprocessing.cylinders import find_cylinders
from preprocessing.graphs import create_graphs, nx_to_PyG
from torch_geometric.loader import DataLoader
from GCN.GCN import GCN3
import time
import csv
from preprocessing.cylinders import get_heatstake_centroid

# CONSTANTS
PATH_TO_STEP_FILE = "doors/doors8.stp"  # Path to the STEP file to be processed
BOX_SIZE = 11  # mm, length from the centroid of the cylinder to the sides of the box

if __name__ == "__main__":

    main_start_time = time.time()
    # Find cylinders in the STEP fi1le
    start_time = time.time()
    print("Finding cylinders...")
    lids, cylinders = find_cylinders(PATH_TO_STEP_FILE, BOX_SIZE)
    print(f"Found {len(cylinders)} cylinders in {time.time() - start_time:.3f} seconds")

    # Create graphs
    start_time = time.time()
    print("Creating graphs from cylinders...")
    cylinder_graphs = create_graphs(cylinders)
    print(
        f"Created {len(cylinder_graphs)} graphs in {time.time() - start_time:.3f} seconds"
    )

    # Process Data to insert into GCN
    start_time = time.time()
    print("Encoding and converting graphs to PyG format...")
    PyG_graphs = nx_to_PyG(cylinder_graphs)
    print(
        f"Encoded and converted graphs to PyG format in {time.time() - start_time:.3f} seconds"
    )
    for g in PyG_graphs:
        print(f"Graph has {g.num_nodes} nodes and {g.num_edges} edges.")

    # Load pre-trained GCN model
    print("Loading pre-trained GCN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN3(feature_dim_size=PyG_graphs[0].num_node_features).to(device)

    state_dict = torch.load(
        "GCN/heatstake_classifier.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    start_time = time.time()
    loader = DataLoader(PyG_graphs, batch_size=len(PyG_graphs), shuffle=False)
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        logits = model(
            features=batch.x, adj=batch.edge_index, batch=batch.batch
        )  # [G, 2]
        preds = logits.argmax(dim=1).cpu().tolist()

    heatstake_coords = []
    for i, p in enumerate(preds):
        if p == 1:
            print(f"Graph {i} is classified as a heatstake.")
            heatstake_coords.append(get_heatstake_centroid(lids[i]))
        else:
            print(f"Graph {i} is NOT classified as a heatstake.")

    print(f"Classified graphs in {time.time() - start_time:.3f} seconds")
    print(f"Number of heatstake coordinates: {len(heatstake_coords)}")

    # Put all coordinates in a csv
    print("Writing into csv...")
    with open("heatstake_coordinates.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Z"])
        for coord in heatstake_coords:
            writer.writerow(
                [round(coord[0], 4), round(coord[1], 4), round(coord[2], 4)]
            )

    print(f'Wrote {len(heatstake_coords)} coordinates into "heatstake_coordinates.csv"')

    total_time = time.time() - main_start_time
    print(f"Total time taken: {total_time/60:.0f}:{total_time%60:.0f} minutes")
