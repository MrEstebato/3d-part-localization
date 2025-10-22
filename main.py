# from preprocessing.OCC.get_cylinders import get_cylinders
# from preprocessing.OCC.create_graph import create_graphs
# import preprocessing.utils_OCC as utils_OCC

from preprocessing.cylinders import find_cylinders
from preprocessing.graphs import create_graphs, plot_graph, nx_to_PyG
import time

# CONSTANTS
PATH_TO_STEP_FILE = "doors/heatstake_solo.STEP"  # Path to the STEP file to be processed
BOX_SIZE = 10  # mm, length from the centroid of the cylinder to the sides of the box

if __name__ == "__main__":

    # Find cylinders in the STEP file
    start_time = time.time()
    cylinders = find_cylinders(PATH_TO_STEP_FILE, BOX_SIZE)
    print(f"Found {len(cylinders)} cylinders in {time.time() - start_time:.3f} seconds")

    # Create graphs
    start_time = time.time()
    cylinder_graphs = create_graphs(cylinders)
    print(
        f"Created {len(cylinder_graphs)} graphs in {time.time() - start_time:.3f} seconds"
    )

    plot_graph(cylinder_graphs[0])

    # Process Data to insert into GCN
    start_time = time.time()
    PyG_graphs = nx_to_PyG(cylinder_graphs)
    print(
        f"Encoded and converted graphs to PyG format in {time.time() - start_time:.3f} seconds"
    )

    print(PyG_graphs[0])

    # TODO Load pre-trained GCN model

    # TODO Insert each graph into gcn to tell whether it is or not a heatstake

    # TODO Get the correct centroid coordenates for the heatstakes

    # TODO Put all coordinates in a csv
