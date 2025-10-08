from preprocessing.OCC.get_cylinders import get_cylinders
from preprocessing.OCC.create_graph import create_graph
import preprocessing.utils as utils

if __name__ == "__main__":
    path_to_step_file = "doors/heatstake_solo.STEP"

    # Load the step file
    pieze = utils.load_step(path_to_step_file)

    # Obtain all cylinders in the piece
    cylinders = get_cylinders(pieze)

    # Create graphs
    graphs = create_graph(cylinders)

    # TODO Insert each graph into gcn to tell whether it is or not a heatstake

    # TODO Get the correct centroid coordenates for the heatstakes

    # TODO Put all coordinates in a csv