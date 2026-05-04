        #-------------------------------------------------------------#
        #  script for illustrating pfdr_prox_tv on denoising problem  #
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
sys.path.append(os.path.join(file_path, "../grid-graph/python/bin"))
sys.path.append(os.path.join(file_path, "wrappers"))

try:
    from grid_graph import grid_to_graph
except ModuleNotFoundError:
    raise ImportError("PFDR prox TV example: compile grid-graph module in "
        "grid-graph/python/")
try:
    from pfdr_prox_tv import pfdr_prox_tv 
except ModuleNotFoundError:
    raise ImportError("PFDR prox TV example: compile pfdr_prox_tv module in "
        "python/")

###  general parameters  ###
plot_results = True
print_results = False
image_file = '../data/Lola.jpg'
noise_level = .2
la_tv = .5*noise_level;

###  parameters; see documentation of pfdr_prox_tv  ###
d1p = 2
rho = 1.5
cond_min = 1e-1
dif_rcd = 0.0
dif_tol = 1e-4
it_max = 1e3
verbose = 1e2

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
edges, connectivities = grid_to_graph(np.array(x0.shape[:2], dtype='uint32'),
    2, compute_connectivities=True, graph_as_forward_star=False)
edge_weights = np.ones(connectivities.shape, dtype=x0.dtype)
edge_weights[connectivities == 1] = la_tv
edge_weights[connectivities == 2] = la_tv/np.sqrt(2)
del connectivities
print('done.')

###  solve the optimization problem  ###
y = y.reshape((np.prod(y.shape[:2]), y.shape[2])).transpose()
start_time = time.time()
x, Obj = pfdr_prox_tv(y, edges, edge_weights=edge_weights, d1p=d1p, rho=rho,
    cond_min=cond_min, dif_rcd=dif_rcd, dif_tol=dif_tol, it_max=it_max,
    verbose=verbose, compute_Obj=True)
exec_time = time.time() - start_time
x = x.transpose().reshape(x0.shape)
y = y.transpose().reshape(x0.shape)
print("Total python wrapper execution time: {:.1f} s\n\n".format(exec_time))

if plot_results:
    fig = plt.figure(1)
    ax = fig.add_subplot(133)
    plt.imshow(np.fmax(0, np.fmin(1, x)))
    plt.tick_params(axis='both', which='both', bottom=False, left=False,
        labelbottom=False, labelleft=False)
    ax.set_title('PFDR prox TV ({:.2f} dB)'.
        format(10*np.log10(np.sum(x0**2)/np.sum((x - x0)**2))))

    fig = plt.figure(2)
    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, exec_time, Obj.size), Obj)
    ax.set_title("objective evolution")
    ax.set_xlabel("time (s)")
    ax.set_ylabel('$1/2 ||y - x||^2 + ||x||_{\delta_{1,2}}$');
