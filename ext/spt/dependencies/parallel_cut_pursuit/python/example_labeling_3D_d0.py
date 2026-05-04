    #--------------------------------------------------------------------%
    #  script for illustrating cp_d0_dist on labeling of 3D point cloud  %
    #--------------------------------------------------------------------%
# References:
# L. Landrieu and G. Obozinski, Cut Pursuit: fast algorithms to learn
# piecewise constant functions on general weighted graphs, SIAM Journal on
# Imaging Science, 10(4):1724-1766, 2017
#
# L. Landrieu et al., A structured regularization framework for spatially
# smoothing semantic labelings of 3D point clouds, ISPRS Journal of
# Photogrammetry and Remote Sensing, 132:102-118, 2017
#
# Camille Baudoin 2019, Hugo Raguet 2023
import sys
import os 
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt

file_path = os.path.realpath(os.path.dirname(__file__))
os.chdir(file_path)
sys.path.append(os.path.join(file_path, "wrappers"))

from cp_d0_dist import cp_d0_dist 

###  classes involved in the task  ###
classNames = ["road", "vegetation", "facade", "hardscape",
        "scanning artifacts", "cars"]
classId = np.arange(1, 7, dtype="uint8")

###  parameters; see documentations of cp_d0_dist  ###
K = 3
cp_dif_tol = 1e-3
max_num_threads = 0
balance_parallel_split = True

###  initialize data  ###
mat = scipy.io.loadmat("../pcd-prox-split/data/labeling_3D.mat",
    squeeze_me=True)
loss = mat["loss"]
y = mat["y"]
homo_d1_weight = mat["homo_d1_weight"]
ground_truth = mat["ground_truth"]
first_edge = mat["first_edge"]
adj_vertices = mat["adj_vertices"]
homo_d0_weight = 3*homo_d1_weight; # adjusted for d0 norm by trial-and-error

# compute prediction performance of random forest
ML = np.argmax(y, axis=0) + 1
F1 = np.zeros(len(classNames),)
for k in range(1,len(classNames) + 1):
    predk = np.array(ML == classId[k-1], dtype="int")
    truek = np.array(ground_truth == classId[k-1], dtype="int")
    F1[k-1] = (2*np.array((predk + truek) == 2, dtype = "int").sum()
               /(predk.sum() + truek.sum()))
print("\naverage F1 of random forest prediction: {:.2f}\n\n".format(F1.mean()))
del predk, truek

###  solve the optimization problem  ###
start_time = time.time()
Comp, rX, Obj, Time = cp_d0_dist(loss, y, first_edge, adj_vertices,
    edge_weights=homo_d0_weight, K=K, cp_dif_tol=cp_dif_tol,
    max_num_threads=max_num_threads,
    balance_parallel_split=balance_parallel_split, compute_Obj=True,
    compute_Time=True)
exec_time = time.time() - start_time
x = rX[:,Comp] # rX is components values, Comp is components assignment
del Comp, rX
print("Total python wrapper execution time {:.0f} s\n\n".format(exec_time))

# compute prediction performance of spatially regularized prediction
ML = np.argmax(x, axis=0) + 1
F1 = np.zeros(len(classNames),)
for k in range(1,len(classNames) + 1):
    predk = np.array(ML == classId[k-1], dtype="int")
    truek = np.array(ground_truth == classId[k-1], dtype="int")
    F1[k-1] = (2*np.array((predk + truek) == 2).sum()
               /(predk.sum() + truek.sum()))
print(("\naverage F1 of spatially regularized prediction: "
       "{:.2f}\n\n").format(F1.mean()))
del predk, truek

fig = plt.figure(1)
fig.clear()
ax = fig.add_subplot(111)
ax.plot(Time, Obj)
ax.set_title("objective evolution")
ax.set_xlabel("time (s)")
ax.set_ylabel('KL$^{{({:.1f})}}$(y||x) + ||x||$_{{\delta_{{0}}}}$'.
    format(loss));
