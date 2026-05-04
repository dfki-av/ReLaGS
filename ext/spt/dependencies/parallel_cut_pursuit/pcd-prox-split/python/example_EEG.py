           #-------------------------------------------------------#
           #  script for illustrating pfdr_d1_ql1b on EEG problem  #
           #-------------------------------------------------------#
# References:
# H. Raguet, A Note on the Forward-Douglas-Rachford Splitting for Monotone 
# Inclusion and Convex Optimization, Optimization Letters, 2018, 1-24
#
# Camille Baudoin 2019, Hugo Raguet 2023
import sys
import os
import numpy as np
import scipy.io
import time

file_path = os.path.realpath(os.path.dirname(__file__))
os.chdir(file_path)
sys.path.append(os.path.join(file_path, "wrappers"))

from pfdr_d1_ql1b import pfdr_d1_ql1b 

###  general parameters  ###
plot_results = True
print_results = False

###  parameters; see documentation of pfdr_d1_ql1b  ###
dif_tol = 1e-3
it_max = 1e4;
rho = 1.5
cond_min = 1e-1
dif_rcd = 0.0
verbose = 1e3

###  initialize data  ###
# dataset courtesy of Ahmad Karfoul and Isabelle Merlet, LTSI, INSERM U1099
# Penalization parameters computed with SURE methods, heuristics adapted from
# H. Raguet: A Signal Processing Approach to Voltage-Sensitive Dye Optical
# Imaging, Ph.D. Thesis, Paris-Dauphine University, 2014
mat = scipy.io.loadmat("../data/EEG.mat", squeeze_me=True)
y = mat["y"]
Phi = mat["Phi"]
x0 = mat["x0"]
first_edge = mat["first_edge"]
adj_vertices = mat["adj_vertices"]
d1_weights = mat["d1_weights"]
l1_weights = mat["l1_weights"]
low_bnd = 0.0 
if plot_results:
    vertices = mat["mesh"].item()[0]
    faces = mat["mesh"].item()[1].astype("int")-1
del mat
# convert graph representation from forward-star to adjacency list
V = first_edge.size - 1;
edges = np.repeat(np.arange(V, dtype="uint32"),
    first_edge[1:] - first_edge[0:-1]);
edges = np.stack((edges, adj_vertices), axis=1)
del first_edge, adj_vertices

###  solve the optimization problem  ###
start_time = time.time()
x, Obj = pfdr_d1_ql1b(y, Phi, edges, d1_weights=d1_weights,
    l1_weights=l1_weights, low_bnd=low_bnd, rho=rho, dif_tol=dif_tol,
    dif_rcd=dif_rcd, it_max=it_max, verbose=verbose, compute_Obj=True)
exec_time = time.time() - start_time
print("Total python wrapper execution time: {:.1f} s\n\n".format(exec_time))

###  compute Dice scores and print results  ###
supp0 = np.array(x0 != 0, dtype="int") # ground truth support
supp = np.array(x != 0, dtype="int")
DS = 2*np.array((supp0 + supp) == 2).sum()/(supp0.sum() + supp.sum());
# support retrieved by discarding nonsignificant values with 2-means clustering
abss = np.abs(x)
sabs = np.sort(abss)
n0 = 0
n1 = x0.size # number of elements per cluster
sum0 = 0 
sum1 = sabs.sum() # sum of each cluster
m = sum1/n1
while 2*sabs[n0] < m:
    n0 = n0 + 1
    n1 = n1 - 1
    sum0 = sum0 + sabs[n0-1]
    sum1 = sum1 - sabs[n0-1]
    m = (sum0/n0 + sum1/n1)

suppa = np.array(abss > (m/2), dtype="int")
DSa = 2*np.array((supp0+suppa) == 2).sum()/(supp0.sum() + suppa.sum());
print(("Dice score: raw {0:.2f}; approx (discard less significant with "
    "2-means) {1:.2f}\n\n").format(DS, DSa))

if plot_results:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    x0min = x0.min()
    x0max = x0.max()
    numberOfColors = 256
    ## ground truth activity
    # map the color index
    xcol = np.floor((x0 - x0min)/(x0max - x0min)*numberOfColors) + 2
    xcol[supp0 == 0] = 1
    # plot figure
    fig = plt.figure(1)
    ax = plt.axes(projection="3d")
    ax.view_init(30, 90)
    cmap = plt.get_cmap("hot")
    collec = ax.plot_trisurf(vertices[:,0],vertices[:,1], vertices[:,2],
            triangles=faces, cmap=cmap)
    collec.set_array(xcol)
    plt.axis("off")
    ax.set_title("Ground truth activity")
    fig.show()
    if print_results:
        print("print ground truth... ", end="", flush=True)
        fig.savefig("EEG_ground_truth.pdf")
        print("done.\n")

    ## retrieved activity
    # map the color index
    xcol = np.floor((x - x0min)/(x0max - x0min)*numberOfColors) + 2
    xcol[supp == 0] = 1
    # plot figure
    fig = plt.figure(2)
    ax = plt.axes(projection="3d")
    ax.view_init(30, 90)
    cmap = plt.get_cmap("hot")
    collec = ax.plot_trisurf(vertices[:,0],vertices[:,1], vertices[:,2],
                             triangles=faces, cmap=cmap)
    collec.set_array(xcol)
    plt.axis("off")
    ax.set_title("Retrieved brain activity")
    fig.show()
    if print_results:
        print("print retrieved activity... ", end="", flush=True)
        fig.savefig("EEG_retrieved_activity.pdf")
        print("done.\n")

    ## print retrieved support
    # map the color index
    xcol = 1 + suppa*numberOfColors;
    fig = plt.figure(3)
    ax = plt.axes(projection="3d")
    ax.view_init(30, 90)
    cmap = plt.get_cmap("hot")
    collec = ax.plot_trisurf(vertices[:,0],vertices[:,1], vertices[:,2],
                             triangles=faces, cmap=cmap)
    collec.set_array(xcol)
    plt.axis("off")
    ax.set_title("Retrieved brain sources")
    fig.show()
    if print_results:
        print("print retrieved sources... ", end="", flush=True)
        fig.savefig("EEG_retrieved_sources.pdf")
        print("done.\n")

    fig = plt.figure(4)
    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, exec_time, Obj.size), Obj)
    ax.set_title("objective evolution")
    ax.set_xlabel("time (s)")
    ax.set_ylabel('$1/2 ||y - \Phi x||^2 + ||x||_{{\ell_1}} + '
        '||x||_{\delta_1}$');
