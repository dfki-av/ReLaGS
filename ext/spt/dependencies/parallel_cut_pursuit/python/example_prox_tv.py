        #-------------------------------------------------------------#
        #  script for illustrating cp_prox_tv on denoising problem  #
        #-------------------------------------------------------------#
# References:
# H. Raguet and L. Landrieu, Preconditioning of a Generalized Forward-Backward
# Splitting and Application to Optimization on Graphs, SIAM Journal on Imaging
# Sciences, 2015, 8, 2706-2739
#
# Camille Baudoin 2019, Hugo Raguet 2023
import sys
import os
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file_path = os.path.realpath(os.path.dirname(__file__))
os.chdir(file_path)
sys.path.append(os.path.join(file_path,
    "../pcd-prox-split/grid-graph/python/bin"))
sys.path.append(os.path.join(file_path, "wrappers"))

try:
    from grid_graph import grid_to_graph
except ModuleNotFoundError:
    raise ImportError("CP prox TV example: compile grid-graph module in "
        "pcd-prox-split/grid-graph/python/")
try:
    from cp_prox_tv import cp_prox_tv 
except ModuleNotFoundError:
    raise ImportError("CP prox TV example: compile cp_prox_tv module in "
        "python/")

###  general parameters  ###
plot_results = True
image_file = '../pcd-prox-split/data/Lola.jpg'
noise_level = .2
la_tv = .5*noise_level;

###  parameters; see documentation of pfdr_prox_tv  ###
d1p = 2
K = 2
cp_dif_tol = 1e-2
pfdr_dif_tol = 1e-4
cp_it_max = 5
verbose = 1e2
max_num_threads = 0
balance_parallel_split = True;

###  initialize data  ###
print('load image and apply transformation... ', end='');
x0 = mpimg.imread(image_file)
x0 = x0[1::3, 1::3, :] # some downsampling
x0 = x0.astype('f4')/255;
y = x0 + noise_level*np.random.randn(*x0.shape).astype(x0.dtype)
print('done.');

##  plot the obervations
if plot_results:
    fig = plt.figure(1)
    fig.clear()
    ax = fig.add_subplot(131)
    plt.imshow(x0)
    plt.tick_params(axis='both', which='both', bottom=False, left=False,
        labelbottom=False, labelleft=False)
    ax.set_title('Original')
    ax = fig.add_subplot(132)
    plt.imshow(np.fmax(0, np.fmin(1, y)))
    plt.tick_params(axis='both', which='both', bottom=False, left=False,
        labelbottom=False, labelleft=False)
    ax.set_title('Noisy ({:.2f} dB)'.
        format(10*np.log10(np.sum(x0**2)/np.sum((y - x0)**2))))
    
print('generate adjacency graph... ', end='')
first_edge, adj_vertices, connectivities = grid_to_graph(
    np.array(x0.shape[:2], dtype='uint32'), 2, compute_connectivities=True)
edge_weights = np.ones(connectivities.shape, dtype=x0.dtype)
edge_weights[connectivities == 1] = la_tv
edge_weights[connectivities == 2] = la_tv/np.sqrt(2)
del connectivities
print('done.')

###  solve the optimization problem  ###
y = y.reshape((np.prod(y.shape[:2]), y.shape[2])).transpose()
start_time = time.time()
Comp, rX, Obj, Time = cp_prox_tv(y, first_edge, adj_vertices,
    edge_weights=edge_weights, d1p=d1p, K=K, cp_dif_tol=cp_dif_tol,
    cp_it_max=cp_it_max, pfdr_dif_tol=pfdr_dif_tol, verbose=verbose,
    compute_Obj=True, compute_Time=True)
exec_time = time.time() - start_time
x = rX[:, Comp] # rX is components values, Comp is components assignment
del Comp, rX
x = x.transpose().reshape(x0.shape)
y = y.transpose().reshape(x0.shape)
print("Total python wrapper execution time: {:.1f} s\n\n".format(exec_time))

if plot_results:
    fig = plt.figure(1)
    ax = fig.add_subplot(133)
    plt.imshow(np.fmax(0, np.fmin(1, x)))
    plt.tick_params(axis='both', which='both', bottom=False, left=False,
        labelbottom=False, labelleft=False)
    ax.set_title('CP prox TV ({:.2f} dB)'.
        format(10*np.log10(np.sum(x0**2)/np.sum((x - x0)**2))))

    fig = plt.figure(2)
    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(Time, Obj)
    ax.set_title("objective evolution")
    ax.set_xlabel("time (s)")
    ax.set_ylabel('$1/2 ||y - x||^2 + ||x||_{\delta_{1,2}}$');
