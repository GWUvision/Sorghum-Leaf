import numpy as np
import networkx as nx

# create by mask not poly
def create_graph(mask, xyz_map, max_distance=10):
    leaf_graph = nx.Graph()
    contain_points_list = np.nonzero(mask)
    contain_points_list = np.stack(contain_points_list, axis=1)
    # add points to graph
    # TODO may be optimized by add_nodes
    for point in contain_points_list:
        leaf_graph.add_node(tuple(point))

    # add edges to graph
    for node in leaf_graph:
        node_x = node[0]
        node_y = node[1]
        coord_0 = np.array([xyz_map[node_x, node_y, 0],
                            xyz_map[node_x, node_y, 1],
                            xyz_map[node_x, node_y, 2]])
        neighbor_list = []
        # TODO may be optimized by add only one side neighbor
        for m in range(-max_distance, max_distance):
            for n in range(-max_distance, max_distance):
                neighbor_list.append([node[0]+m, node[1]+n])
        neighbor_list.remove([node[0], node[1]])

        for neighbor in neighbor_list:
            if tuple(neighbor) not in leaf_graph:
                continue
            x = neighbor[0]
            y = neighbor[1]
            coord_1 = np.array([xyz_map[x, y, 0],
                                xyz_map[x, y, 1],
                                xyz_map[x, y, 2]])
            weight = np.linalg.norm(coord_0 - coord_1)
            leaf_graph.add_edge(node, tuple(neighbor), weight=weight)
    return leaf_graph

def limited_pairs_longest_shortest_path_length(graph, nodes_list):
    max_length = 0
    for i in range(len(nodes_list)):
        for j in range(i+1, len(nodes_list)):
            shortest_path_length = nx.dijkstra_path_length(graph, tuple(nodes_list[i]), tuple(nodes_list[j]))
            if shortest_path_length > max_length:
                max_length = shortest_path_length
    return max_length

def leaf_length(mask, xyz_map, edge_point_list):
    leaf_graph = create_graph(mask, xyz_map, max_distance=5)
    leaf_len = limited_pairs_longest_shortest_path_length(leaf_graph, edge_point_list)
    return leaf_len