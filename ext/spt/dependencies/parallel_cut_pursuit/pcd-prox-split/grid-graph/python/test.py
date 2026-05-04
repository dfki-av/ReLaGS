import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                                              "./bin"))

from grid_graph import edge_list_to_forward_star, grid_to_graph

# V = 4
# edges = np.array([[0, 1], [1, 2], [0, 2], [0, 3]], dtype = "uint16")
# edges = np.asfortranarray([[0, 1, 0, 0], [1, 2, 2, 3]])

# first_edge, adj_vertices, reindex = edge_list_to_forward_star(V, edges)

grid_shape = np.array([3, 2, 2], dtype = "int16");

fe, av, co = grid_to_graph(grid_shape, compute_connectivities = True,
    connectivity = 2, row_major_index = False);

print('fe')
print(fe)
print('')

print('av')
print(av)
print('')

print('co')
print(co)
print('')

print('vert ind')
print(np.array(range(grid_shape.prod())).reshape(grid_shape, order = 'F'))
print('')

ed = grid_to_graph(grid_shape, compute_connectivities = False,
    connectivity = 2, row_major_index = False, graph_as_forward_star = False);

print('ed')
print(ed)
print('')

V = grid_shape.prod()
fe_, av_ = edge_list_to_forward_star(V, ed)[0:2]

print('fe_')
print(fe_)
print('')

print('av_')
print(av_)
print('')
