import piece_utils as piece
import networkx as nx

if __name__ == "__main__":
    path_to_step_file = "../doors2.stp"

    # Obtain all cylinders in the piece
    cylinders = piece.get_cylinders(path_to_step_file)

    """graphs: list[nx.Graph] = []

    for cylinder in cylinders:
        local_graph = piece.create_local_graph(cylinder)
        graphs.append(local_graph)"""