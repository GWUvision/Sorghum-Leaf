import numpy as np
import networkx as nx
import utils
from skimage import measure


def limited_pairs_longest_shortest_path_length(graph, nodes_list):
    max_length = 0
    max_length_path = None
    for i in range(len(nodes_list)):
        for j in range(i+1, len(nodes_list)):
            shortest_path = nx.dijkstra_path(graph, tuple(nodes_list[i]), tuple(nodes_list[j]))
            shortest_path_length = nx.dijkstra_path_length(graph, tuple(nodes_list[i]), tuple(nodes_list[j]))
            if shortest_path_length > max_length:
                max_length = shortest_path_length
                max_length_path = shortest_path
    return max_length_path, max_length

def mask2contour(mask, approx_factor=0.0, longest_contour_only=False):
    """Find the outlines of the mask. If the approx_factor is not 0.0, it will return approximated
    with the step length by approx_factor*total_length
    """
    contours = measure.find_contours(~mask.astype(bool), 0)
    if approx_factor != 0.0:
        approx_contours = [measure.approximate_polygon(contour, .05 * utils.contour_length(contour)) for contour in contours]
    if longest_contour_only:
        return contours[np.argmax(list(map(len, contours)))]
    else:
        return contours
    
class SorghumLeafMeasure: 
    # TODO add a method to get edge_point_list
    def __init__(self, mask, xyz_map, max_neibor_pixel=10):
        self.mask = mask
        self.xyz_map = xyz_map
        self.max_neibor_pixel = max_neibor_pixel
        self.leaf_edge_approx_factor = 0.05
        self.leaf_graph = self.create_graph()
        self._leaf_mask_for_edge_detect = np.pad(self.mask, 1, 'constant', constant_values=(0, 0))
        self.leaf_edge = mask2contour(self._leaf_mask_for_edge_detect, approx_factor=0.0, longest_contour_only=True).astype(int) - 1
        self.leaf_edge_approx = measure.approximate_polygon(self.leaf_edge, self.leaf_edge_approx_factor * utils.contour_length(self.leaf_edge))
        self.leaf_len = None
        # leaf_len_path is the midvein
        self.leaf_len_path = None
        self.leaf_width = None
        self.leaf_width_path = None
    
    def create_graph(self):
        leaf_graph = nx.Graph()
        contain_points_list = np.nonzero(self.mask)
        contain_points_list = np.stack(contain_points_list, axis=1)
        # add points to graph
        # TODO could be optimized by add_nodes()
        for point in contain_points_list:
            leaf_graph.add_node(tuple(point))

        # add edges to graph
        for node in leaf_graph:
            node_x = node[0]
            node_y = node[1]
            coord_0 = np.array([self.xyz_map[node_x, node_y, 0],
                                self.xyz_map[node_x, node_y, 1],
                                self.xyz_map[node_x, node_y, 2]])
            neighbor_list = []
            # TODO may be optimized by add only one side neighbor
            for m in range(-self.max_neibor_pixel, self.max_neibor_pixel):
                for n in range(-self.max_neibor_pixel, self.max_neibor_pixel):
                    neighbor_list.append([node[0]+m, node[1]+n])
            neighbor_list.remove([node[0], node[1]])

            for neighbor in neighbor_list:
                if tuple(neighbor) not in leaf_graph:
                    continue
                x = neighbor[0]
                y = neighbor[1]
                coord_1 = np.array([self.xyz_map[x, y, 0],
                                    self.xyz_map[x, y, 1],
                                    self.xyz_map[x, y, 2]])
                weight = np.linalg.norm(coord_0 - coord_1)
                leaf_graph.add_edge(node, tuple(neighbor), weight=weight)
        return leaf_graph
    
    def calc_leaf_length(self):
        self.leaf_len_path, self.leaf_len = limited_pairs_longest_shortest_path_length(self.leaf_graph, self.leaf_edge_approx)
        return self.leaf_len
    
    def calc_leaf_width(self, split_n=5):
        # get points of n-section
        split_length = self.leaf_len / split_n
        next_point_position = split_length
        perv_travelled = 0.0
        n_section_points_list = []
        for i in range(1, len(self.leaf_len_path) - 1):
            vector_to_next = np.array(self.leaf_len_path[i]) - np.array(self.leaf_len_path[i-1]) 
            points_distance = np.linalg.norm(vector_to_next)
            current_travelled = perv_travelled + points_distance
            while current_travelled > next_point_position:
                length_ratio = (next_point_position - perv_travelled) / points_distance
                n_section_point = np.array(self.leaf_len_path[i-1]) + length_ratio * (vector_to_next)
                n_section_points_list.append(n_section_point.round().astype(int))
                next_point_position += split_length
            if len(n_section_points_list) >= split_n:
                break
            perv_travelled = current_travelled
            
        self.leaf_width_mid_points = np.array(n_section_points_list)
        # Throw outside n_section point
        n_section_points_list = list(filter (lambda x: tuple(x) in self.leaf_graph.node and ~any(np.equal(self.leaf_edge, x).all(1)), n_section_points_list))
        
        aux_edges = []
        for edge_point in self.leaf_edge:
            aux_edges.append(('aux_point', tuple(edge_point)))
        n_section_leaf_width_list = []
        n_section_leaf_width_path_list = []
        for n_section_point in n_section_points_list:
            # find the nearest point from the point of n-section
            """
            The nearest point can be find by two method:
            1. norm
            2. add an point that connected to all the edge points with same weight
               then get the shortest path between that point and n_section_point
            """
            
            temp_graph = self.leaf_graph.copy()
            temp_graph.add_node('aux_point')
            temp_graph.add_edges_from(aux_edges)
            aux_shortest_path = nx.dijkstra_path(temp_graph, tuple(n_section_point), 'aux_point')
            nearest_point_on_edge = np.array(aux_shortest_path[-2])
            # remove vectors at same side
            point_cosine = np.dot(nearest_point_on_edge - n_section_point, (self.leaf_edge - n_section_point).T)/(np.linalg.norm(self.leaf_edge)* np.linalg.norm(n_section_point))
            self._point_dot_prod = point_cosine
            same_side_points = list(map(tuple, self.leaf_edge[np.where(point_cosine>0)]))
            self._same_side_points = np.array(same_side_points)
            temp_graph.remove_nodes_from(same_side_points)
            # find the nearest point for another sid
            aux_other_side_shortest_path = nx.dijkstra_path(temp_graph, tuple(n_section_point), 'aux_point')
            nearest_point_on_edge_other_side = np.array(aux_other_side_shortest_path[-2])
            # add two up
            leaf_width_path = nx.dijkstra_path(self.leaf_graph,
                                               tuple(nearest_point_on_edge),
                                               tuple(nearest_point_on_edge_other_side))
            leaf_width = nx.dijkstra_path_length(self.leaf_graph,
                                                 tuple(nearest_point_on_edge),
                                                 tuple(nearest_point_on_edge_other_side))
            n_section_leaf_width_list.append(leaf_width)
            n_section_leaf_width_path_list.append(leaf_width_path)
        if len(n_section_leaf_width_path_list) == 0: 
            return None
        max_idx = np.argmax(n_section_leaf_width_list)
        self.leaf_width = n_section_leaf_width_list[max_idx]
        self.leaf_width_path = n_section_leaf_width_path_list[max_idx]
        return self.leaf_width
        