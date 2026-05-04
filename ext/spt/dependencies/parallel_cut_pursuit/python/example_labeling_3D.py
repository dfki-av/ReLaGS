     #-------------------------------------------------------------------#
     #  script for illustrating cp_d1_lsx on labeling of 3D point cloud  #
     #-------------------------------------------------------------------#
# References:
# H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
# Nonsmooth Functionals with Graph Total Variation, International Conference on
# Machine Learning, PMLR, 2018, 80, 4244-4253
#
# L. Landrieu and H. Raguet and B. Vallet and C. Mallet and M. Weinmann, A
# structured regularization framework for spatially smoothing semantic
# labelings of 3D point clouds, Journal of Photogrammetry and Remote Sensing,
# 2017, 132, 102-118 
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

from cp_d1_lsx import cp_d1_lsx 

###  classes involved in the task  ###
classNames = ["road", "vegetation", "facade", "hardscape",
        "scanning artifacts", "cars"]
classId = np.arange(1, 7, dtype="uint8")

###  parameters; see documentation of cp_d1_lsx.py  ###
cp_dif_tol = 1e-3
K = 3
max_num_threads = 0
balance_parallel_split = True

###  initialize data  ###
# For details on the data and parameters, see H. Raguet, A Note on the
# Forward-Douglas--Rachford Splitting for Monotone Inclusion and Convex
# Optimization Optimization Letters, 2018, 1-24
mat = scipy.io.loadmat("../pcd-prox-split/data/labeling_3D.mat",
    squeeze_me=True)
loss = mat["loss"]
y = mat["y"]
homo_d1_weight = mat["homo_d1_weight"]
ground_truth = mat["ground_truth"]
first_edge = mat["first_edge"]
adj_vertices = mat["adj_vertices"]
del mat

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
Comp, rX, Obj, Time = cp_d1_lsx(loss, y, first_edge, adj_vertices,
    edge_weights=homo_d1_weight, K=K, max_num_threads=max_num_threads,
    balance_parallel_split=balance_parallel_split, compute_Obj=True,
    compute_Time=True)
exec_time = time.time() - start_time
x = rX[:, Comp] # rX is components values, Comp is components assignment
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
ax.set_ylabel('KL$^{{({:.1f})}}$(y||x) + ||x||$_{{\delta_{{1,1}}}}$'.
    format(loss));
