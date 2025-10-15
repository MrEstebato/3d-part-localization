from preprocessing.OCC.get_cylinders import get_cylinders
from preprocessing.OCC.create_graph import create_graphs
from preprocessing.CQ.find_cylinders import find_cylinders
import preprocessing.utils_OCC as utils_OCC
import time

# CONSTANTS
path_to_step_file = "doors/doors1.stp"  # Path to the STEP file to be processed
radius = 20.0  # mm, radius around cylinder center to include faces


if __name__ == "__main__":

    # Using CadQuery (preferred)
    start_time = time.time()
    cylinders = find_cylinders(path_to_step_file)
    print(f"Found {len(cylinders)} cylinders in {time.time() - start_time:.3f} seconds")

    # Using PythonOCC (probably deprecated)

    # Load the step file
    #pieze = utils_OCC.load_step(path_to_step_file)

    # Obtain all cylinders in the piece
    # start_time = time.time()
    # cylinders = get_cylinders(pieze)
    # print(
    #     f"Obtained {len(cylinders)} cylinders in {time.time() - start_time:.3f} seconds"
    # )

    # # Create graphs
    # start_time = time.time()
    # graphs = create_graphs(pieze, cylinders, radius)
    # """ for i, (G, payload) in enumerate(graphs):
    #     print(f"Graph {i}: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    #     face_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "face"]
    #     edge_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "edge"]
    #     vertex_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "vertex"]
    #     print(
    #         f"  faces={len(face_nodes)} edges={len(edge_nodes)} vertices={len(vertex_nodes)} center={payload['center']}"
    #     ) """
    # print(f"Created {len(graphs)} graphs in {time.time() - start_time:.3f} seconds")

    # TODO Insert each graph into gcn to tell whether it is or not a heatstake

    # TODO Get the correct centroid coordenates for the heatstakes

    # TODO Put all coordinates in a csv