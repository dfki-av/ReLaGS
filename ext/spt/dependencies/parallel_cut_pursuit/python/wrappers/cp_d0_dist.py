import numpy as np
import os 
import sys

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                                              "../bin"))

from cp_d0_dist_cpy import cp_d0_dist_cpy

def cp_d0_dist(loss, Y, first_edge, adj_vertices, edge_weights=None, 
               vert_weights=None, coor_weights=None, cp_dif_tol=1e-3,
               cp_it_max=10, K=2, split_iter_num=2, split_damp_ratio=1.0,
               kmpp_init_num=3, kmpp_iter_num=3, min_comp_weight=0.0,
               verbose=True, max_num_threads=0, max_split_size=None,
               balance_parallel_split=True, compute_List=False,
               compute_Graph=False, compute_Obj=False, compute_Time=False,
               compute_Dif=False):
    """
    Comp, rX, [List, Graph, Obj, Time, Dif] = cp_d0_dist(loss, Y,
        first_edge, adj_vertices, edge_weights=None, vert_weights=None,
        coor_weights=None, cp_dif_tol=1e-3, cp_it_max=10, K=2,
        split_iter_num=2, split_damp_ratio=1.0, kmpp_init_num=3,
        kmpp_iter_num=3, min_comp_weight=0.0, verbose=True, max_num_threads=0,
        max_split_size=None, balance_parallel_split=True, compute_List=False,
        compute_Obj=False, compute_Time=False, compute_Dif=False)

    Cut-pursuit algorithm with d0 (weighted contour length) penalization, with
    a loss akin to a distance:

    minimize functional over a graph G = (V, E)

        F(x) = sum_v loss(y_v, x_v) + ||x||_d0

    where for each vertex, y_v and x_v are D-dimensional vectors, the loss is
    a mix of the sum of square differences and a Kullback-Leibler divergence
    (equivalent to cross-entropy in this formulation); see the 
    'loss' attribute, and ||x||_d0 = sum_{uv in E : xu != xv} w_d0_uv ,

    using greedy cut-pursuit approach with splitting initialized with 
    k-means++.

    Available data-fidelity loss include:

    quadratic (loss = D):
        f(x) = ||y - x||_{l2,W}^2 ,
    where W is a diagonal metric (separable product along ℝ^V and ℝ^D),
    that is ||y - x||_{l2,W}^2 = sum_{v in V} w_v ||x_v - y_v||_{l2,M}^2
                               = sum_{v in V} w_v sum_d m_d (x_vd - y_vd)^2;

    Kullback-Leibler divergence (equivalent to cross-entropy) on the
    probability simplex (0 < loss < 1):
        f(x) = sum_v w_v KLs(x_v, y_v),
    with KLs(y_v, x_v) = KL(s u + (1 - s) y_v ,  s u + (1 - s) x_v), where
        KL is the regular Kullback-Leibler divergence,
        u is the uniform discrete distribution over {1,...,D}, and
        s = loss is the smoothing parameter
        m is a diagonal metric weighting the coordinates;
    it yields
        KLs(y_v, x_v) = - H(s u + (1 - s) y_v)
            - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) x_{v,d}) ,
    where H is the (weighted) entropy, that is H(s u + (1 - s) y_v)
        = - sum_d m_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) y_{v,d}) ;
    note that the choosen order of the arguments in the Kullback--Leibler
    does not favor the entropy of x (H(s u + (1 - s) y_v) is a constant),
    hence this loss is actually equivalent to cross-entropy.

    Both (1 <= loss < D): quadratic on coordinates from 1 to loss, and
    Kullback-Leibler divergence on coordinates from loss + 1 to D;
    
    note that coordinate weights makes no sense for Kullback-Leibler divergence
    alone, but should be used for weighting quadratic and KL when mixing both,
    in which case coor_weights should be of length loss + 1;
 
    INPUTS: real numeric type is either float32 or float64, not both;
            indices numeric type can be int32 or uint32.

    NOTA: by default, components are identified using uint16 identifiers;
    this can be easily changed in the module source if more than 65535 
    components are expected (recompilation is necessary)

    loss - D for quadratic, 0 < loss < 1 for smoothed Kullback-Leibler, D for
           quadratic, 1 <= loss < D for both (quadratic on the first 'loss'
           values and Kullback-Leibler on the 'D - loss' remaining)
    Y - observations, (real) D-by-V array;
        careful to the internal memory representation of multidimensional
        arrays; the C++ implementation uses column-major order (F-contiguous);
        usually numpy uses row-major order (C-contiguous), but this can often
        be taken care of without actually copying data by using transpose();
        for Kullback-Leibler loss, the value at each vertex must lie on the
        probability simplex
    first_edge, adj_vertices - forward-star graph representation:
        vertices are numeroted (start at 0) in the order they are given in Y;
            careful to the internal memory representation of multidimensional
            arrays, usually numpy uses row-major order (C-contiguous)
        edges are numeroted (start at 0) so that all edges originating
            from a same vertex are consecutive;
        for each vertex, 'first_edge' indicates the first edge starting from 
            the vertex (or, if there are none, starting from the next vertex);
            (int32 or uint32) array of length V+1, the last value is the
            total number of edges;
        for each edge, 'adj_vertices' indicates its ending vertex, (int32 or
            uint32) array of length E
    edge_weights - (real) array of length E or scalar for homogeneous weights
    vert_weights - weights on vertices (w_v in above notations)
        (real) array of length V or empty for no weights
    coor_weights - weights on coordinates (m_d in above notations)
        (real) array of length D (quadratic case) or of length loss + 1
        (quadratic + Kullback-Leibler case, the last coordinate weights the all
        Kullback-Leibler part), or empty for no weights
    cp_dif_tol - stopping criterion on iterate evolution; algorithm stops if
        relative changes (that is, relative dissimilarity measures defined by 
        the choosen loss between succesive iterates and between current iterate
        and observation) is less than dif_tol; 1e-3 is a typical value
    cp_it_max  - maximum number of iterations (graph cut, subproblem, merge)
        10 cuts solve accurately most problems
    K - number of alternative values considered in the split step
    split_iter_num - number of partition-and-update iterations in the split 
        step
    split_damp_ratio - edge weights damping for favoring splitting; edge
        weights increase in linear progression along partition-and-update
        iterations, from this ratio up to original value; real scalar between 0
        and 1, the latter meaning no damping
    kmpp_init_num - number of random k-means initializations in the split step
    kmpp_iter_num - number of k-means iterations in the split step
    min_comp_weight - minimum total weight (number of vertices if no weights
        are given on the vertices) that a component is allowed to have;
        components with smaller weights are merged with adjacent components
    verbose - if true, display information on the progress
    max_num_threads - if greater than zero, set the maximum number of threads
        used for parallelization with OpenMP
    max_split_size - maximum number of vertices allowed in connected component
        passed to a split problem; make split of very large components faster,
        but might induced suboptimal artificial cuts
    balance_parallel_split - if true, the parallel workload of the split step 
        is balanced; WARNING: this might trade off speed against optimality
    compute_List  - report the list of vertices constituting each component
    compute_Graph - get the reduced graph on the components
    compute_Obj   - compute the objective functional along iterations 
    compute_Time  - monitor elapsing time along iterations
    compute_Dif   - compute relative evolution along iterations 

    OUTPUTS: List, Graph, Obj, Time and Dif are optional, set parameters
        compute_List, compute_Graph, compute_Obj, compute_Time, or
        compute_Dif to True to request them and capture them in output
        variables in that order

    Comp - assignement of each vertex to a component, (uint16) array of
        length V 
    rX  - values of each component of the minimizer, (real) array of size
        D-by-rV; the actual minimizer is then reconstructed as X = rX[:, Comp];
    List - if requested, list of vertices constituting each component; python
        list of length rV, containing (uint32) arrays of indices
    Graph - if requested, reduced graph structure; python tuple of length 3
        representing the graph as forward-star (see input first_edge and
        adj_vertices) together with edge weights
    Obj - if requested, values of the objective functional along iterations;
        array of length actual number of cut-pursuit iterations performed + 1
    Time - if requested, the elapsed time along iterations; array of length
        actual number of cut-pursuit iterations performed + 1
    Dif - if requested, if requested, the iterate evolution along iterations;
        array of length actual number of cut-pursuit iterations performed
 
    Parallel implementation with OpenMP API.

    L. Landrieu and G. Obozinski, Cut Pursuit: fast algorithms to learn
    piecewise constant functions on general weighted graphs, SIAM Journal on
    Imaging Science, 10(4):1724-1766, 2017

    L. Landrieu et al., A structured regularization framework for spatially
    smoothing semantic labelings of 3D point clouds, ISPRS Journal of
    Photogrammetry and Remote Sensing, 132:102-118, 2017

    Baudoin Camille 2019, Raguet Hugo 2021, 2022
    """
    
    # Determine the type of float argument (real_t) 
    # real_t type is determined by the first parameter Y 
    if type(Y) != np.ndarray:
        raise TypeError("Cut-pursuit d0 distance: argument 'Y' must be a "
                        "numpy array.")

    if Y.size > 0 and Y.dtype == "float64":
        real_t = "float64" 
    elif Y.size > 0 and Y.dtype == "float32":
        real_t = "float32" 
    else:
        raise TypeError("Cut-pursuit d0 distance: argument 'Y' must be a "
                        "nonempty numpy array of type float32 or float64.") 
    # Determine V
    if Y.ndim > 1:
        V = Y.shape[1]
        D = Y.shape[0]
    else:
        V = Y.shape[0]
        D = 1
    
    # Convert in numpy array scalar entry
    if (type(first_edge) != np.ndarray
        or first_edge.dtype not in ["int32", "uint32"]):
        raise TypeError("Cut-pursuit d0 distance: argument 'first_edge' must "
                        "be a numpy array of type int32 or uint32.")

    if (type(adj_vertices) != np.ndarray
        or adj_vertices.dtype not in ["int32", "uint32"]):
        raise TypeError("Cut-pursuit d0 distance: argument 'adj_vertices' "
                        "must be a numpy array of type int32 or uint32.")

    if type(edge_weights) != np.ndarray:
        if type(edge_weights) == list:
            raise TypeError("Cut-pursuit d0 distance: argument 'edge_weights' "
                            "must be a scalar or a numpy array.")
        elif edge_weights != None:
            edge_weights = np.array([edge_weights], dtype=real_t)
        else:
            edge_weights = np.array([1.0], dtype=real_t)
        
    if type(vert_weights) != np.ndarray:
        if type(vert_weights) == list:
            raise TypeError("Cut-pursuit d0 distance: argument 'vert_weights' "
                            "must be a numpy array.")
        else:
            vert_weights = np.array([], dtype=real_t)

    if type(coor_weights) != np.ndarray:
        if type(coor_weights) == list:
            raise TypeError("Cut-pursuit d0 distance: argument 'coor_weights' "
                            "must be a numpy array.")
        else:
            coor_weights = np.array([], dtype=real_t)

    # Sanity check graph on coor weights and loss 
    if 0.0 < loss and loss < 1.0 and coor_weights.size > 0:
        raise ValueError("Cut-pursuit d0 distance: with Kullback-Leibler loss "
            "(0 < loss < 1), 'coor_weights' should be empty; weighting "
            "coordinates of the probability space makes no sense.")
    elif (1 <= loss and loss < D and coor_weights.size > 0 and
        coor_weights.size != loss + 1):
        raise ValueError("Cut-pursuit d0 distance: for weighting quadratic and"
            " Kullback-Leibler parts, 'coor_weights' should be of size loss + "
            "1 = {0}, because argument 'loss' indicates the number of "
            "coordinates involved in the quadratic part and the last weight is"
            " for KL (size {1} given)".format(loss + 1, coor_weights.size))
    elif loss == D and coor_weights.size > 0 and coor_weights.size != D:
        raise ValueError("Cut-pursuit d0 distance: with quadratic loss (D),"
            " argument 'coor_weights' should be empty or of size D = {0} "
            "(size {1} given)".format(D, coor_weights.size))

    # Sanity check graph on graph structure 
    if first_edge.size != V + 1 :
        raise ValueError("Cut-pursuit d0 distance: argument 'first_edge'"
                         "should contain |V| + 1 = {0} elements, "
                         "but {1} are given".format(V + 1, first_edge.size))
 
    # Check type of all numpy.array arguments of type float
    for name, ar_args in zip(
            ["Y", "edge_weights", "vert_weights", "coor_weights"],
            [ Y ,  edge_weights ,  vert_weights ,  coor_weights ]):
        if ar_args.dtype != real_t:
            raise TypeError("Cut-pursuit d0 distance: argument '{0}' must be "
                            "of type '{1}'".format(name, real_t))

    # Check fortran continuity of all multidimensional numpy array arguments
    if not(Y.flags["F_CONTIGUOUS"]):
        raise TypeError("Cut-pursuit d0 distance: argument 'Y' must be in "
                        "column-major order (F-contigous).")

    # Convert in float64 all float arguments
    loss = float(loss)
    split_damp_ratio = float(split_damp_ratio)
    cp_dif_tol = float(cp_dif_tol)
     
    # Convert all int arguments
    cp_it_max = int(cp_it_max)
    K = int(K)
    split_iter_num = int(split_iter_num)
    kmpp_init_num = int(kmpp_init_num)
    kmpp_iter_num = int(kmpp_iter_num)
    max_num_threads = int(max_num_threads)
    if max_split_size is None:
        max_split_size = V
    else:
        max_split_size = int(max_split_size)

    # Check type of all booleen arguments
    for name, b_args in zip(
        ["verbose", "balance_parallel_split", "compute_List", "compute_Graph",
         "compute_Obj", "compute_Time", "compute_Dif"],
        [ verbose ,  balance_parallel_split ,  compute_List ,  compute_Graph ,
          compute_Obj ,  compute_Time ,  compute_Dif ]):
        if type(b_args) != bool:
            raise TypeError("Cut-pursuit d0 distance: argument '{0}' must be "
                            "boolean".format(name))

    # Call wrapper python in C  
    return cp_d0_dist_cpy(loss, Y, first_edge, adj_vertices, edge_weights,
        vert_weights, coor_weights, cp_dif_tol, cp_it_max, K, split_iter_num,
        split_damp_ratio, kmpp_init_num, kmpp_iter_num, min_comp_weight,
        verbose, max_num_threads, max_split_size, balance_parallel_split,
        real_t == "float64", compute_List, compute_Graph, compute_Obj,
        compute_Time, compute_Dif)
