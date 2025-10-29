import torch
from preprocessing.cylinders import find_cylinders
from preprocessing.graphs import create_graphs, plot_graph, nx_to_PyG
from GCN.GCN import GCN2
import time

# CONSTANTS
PATH_TO_STEP_FILE = "doors/doors1.stp"  # Path to the STEP file to be processed
BOX_SIZE = 10  # mm, length from the centroid of the cylinder to the sides of the box

if __name__ == "__main__":

    # Find cylinders in the STEP file
    start_time = time.time()
    print("Finding cylinders...")
    cylinder_coords, cylinders = find_cylinders(PATH_TO_STEP_FILE, BOX_SIZE)
    print(f"Found {len(cylinders)} cylinders in {time.time() - start_time:.3f} seconds")

    # Create graphs
    start_time = time.time()
    print("Creating graphs from cylinders...")
    cylinder_graphs = create_graphs(cylinders)
    print(
        f"Created {len(cylinder_graphs)} graphs in {time.time() - start_time:.3f} seconds"
    )
    print(
        f"Example graph has {cylinder_graphs[0].number_of_nodes()} nodes and {cylinder_graphs[0].number_of_edges()} edges"
    )

    # print("Plotting graph...")
    # plot_graph(cylinder_graphs[0])

    # Process Data to insert into GCN
    start_time = time.time()
    print("Encoding and converting graphs to PyG format...")
    PyG_graphs = nx_to_PyG(cylinder_graphs)
    print(
        f"Encoded and converted graphs to PyG format in {time.time() - start_time:.3f} seconds"
    )

    print(PyG_graphs[0])

    # Load pre-trained GCN model
    print("Loading pre-trained GCN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN2(
        feature_dim_size=PyG_graphs[0].num_node_features, num_classes=2, dropout=0.3
    ).to(device)

    state_dict = torch.load(
        "GCN/heatstake_classifier.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    # Insert each graph into GCN to tell whether it is or not a heatstake
    start_time = time.time()
    heatstake_coords = []
    for i, graph in enumerate(PyG_graphs):
        with torch.no_grad():
            graph = graph.to(device)  # move x and edge_index to the model's device
            out = model(adj=graph.edge_index, features=graph.x)
            predicted_class = out.argmax(dim=1).item()
            if predicted_class == 1:
                print(f"Graph {i} is classified as a heatstake.")
                heatstake_coords.append(cylinder_coords[i])
            else:
                print(f"Graph {i} is NOT classified as a heatstake.")
    cylinder_coords = heatstake_coords

    print(f"Classified graphs in {time.time() - start_time:.3f} seconds")
    print(f"Number of heatstake coordinates: {len(heatstake_coords)}")

    # TODO Get the correct centroid coordinates for the heatstakes

    # TODO Put all coordinates in a csv
