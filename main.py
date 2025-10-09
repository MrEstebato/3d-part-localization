from preprocessing.OCC.get_cylinders import get_cylinders
from preprocessing.OCC.create_graph import create_graphs
import preprocessing.utils as utils

# CONSTANTS
path_to_step_file = "doors/heatstake_solo.STEP" # Path to the STEP file to be processed
radius = 20.0 # mm, radius around cylinder center to include faces


if __name__ == "__main__":

    # Load the step file
    pieze = utils.load_step(path_to_step_file)

    # Obtain all cylinders in the piece
    cylinders = get_cylinders(pieze)

    # Create graphs
    graphs = create_graphs(cylinders, radius)
    print("graphs:", graphs)

    # TODO Insert each graph into gcn to tell whether it is or not a heatstake

    # TODO Get the correct centroid coordenates for the heatstakes

    # TODO Put all coordinates in a csv