from preprocessing.get_cylinders import get_cylinders
from preprocessing.create_graph import create_graph
from OCC.Core.STEPControl import STEPControl_Reader

if __name__ == "__main__":
    path_to_step_file = "doors/heatstake_solo.STEP"

    # Load Step File
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(path_to_step_file)
    
    if status != 1:
        raise Exception(f"Error reading STEP file: {path_to_step_file}")

    # Obtain all cylinders in the piece
    cylinders = get_cylinders(step_reader)

    # Create graphs
    graphs = create_graph(cylinders)

    # TODO Insert each graph into gcn to tell whether it is or not a heatstake

    # TODO Get the correct centroid coordenates for the heatstakes

    # TODO Put all coordinates in a csv