import numpy as np
import os 
import sys

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                                              "../bin"))

from cp_d1_lsx_cpy import cp_d1_lsx_cpy

def cp_d1_lsx(loss, Y, first_edge, adj_vertices, edge_weights=None, 
    loss_weights=None, d11_metric=None, cp_dif_tol=1e-3, cp_it_max=10,
    K=2, split_iter_num=1, split_damp_ratio=1.0, split_values_init_num=2,
    split_values_iter_num=2, pfdr_rho=1., pfdr_cond_min=1e-2, pfdr_dif_rcd=0.,
    pfdr_dif_tol=None, pfdr_it_max=int(1e4), verbose=int(1e2),
    max_num_threads=0, max_split_size=None, balance_parallel_split=True,
    compute_List=False, compute_Graph=False, compute_Obj=False, 
    compute_Time=False, compute_Dif=False):
    """
    Comp, rX, [List, Graph, Obj, Time, Dif] = cp_d1_lsx(loss, Y,
        first_edge, adj_vertices, edge_weights=None, loss_weights=None,
        d11_metric=None, cp_dif_tol=1e-3, cp_it_max=10,  K=2,
        split_iter_num=1, split_damp_ratio=1.0, split_values_init_num=2,
        split_values_iter_num=2, pfdr_rho=1.0, pfdr_cond_min=1e-2,
        pfdr_dif_rcd=0.0, pfdr_dif_tol=1e-2*cp_dif_tol, pfdr_it_max=1e4,
        verbose=1e2, max_num_threads=0, balance_parallel_split=True,
        compute_List=False, compute_Graph=False, compute_Obj=False,
        compute_Time=False, compute_Dif=False)

    Cut-pursuit algorithm with d1 (total variation) penalization, with a 
    separable loss term and simplex constraints:

    minimize functional over a graph G = (V, E)

        F(x) = f(x) + ||x||_d1 + i_{simplex}(x)

    where for each vertex, x_v is a D-dimensional vector,
          f is a separable data-fidelity loss
          ||x||_d1 = sum_{uv in E} w_d1_uv (sum_d w_d1_d |x_ud - x_vd|),
    and i_{simplex} is the standard D-simplex constraint over each vertex,
        i_{simplex} = 0 for all v, (for all d, x_vd >= 0) and sum_d x_vd = 1,
                    = infinity otherwise;

    using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
    splitting algorithm.

    Available separable data-fidelity loss include:

    linear
        f(x) = - <x, y> ,  with  <x, y> = sum_{v,d} x_{v,d} y_{v,d} ;

    quadratic
        f(x) = 1/2 ||y - x||_{l2,w}^2 ,
    with  ||y - x||_{l2,w}^2 = sum_{v,d} w_v (y_{v,d} - x_{v,d})^2 ;

    smoothed Kullback-Leibler divergence (cross-entropy)
        f(x) = sum_v w_v KLs(x_v, y_v),
    with KLs(y_v, x_v) = KL(s u + (1 - s) y_v ,  s u + (1 - s) x_v), where
        KL is the regular Kullback-Leibler divergence,
        u is the uniform discrete distribution over {1,...,D}, and
        s = loss is the smoothing parameter ;
    it yields
        KLs(y_v, x_v) = - H(s u + (1 - s) y_v)
            - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) x_{v,d}) ,
    where H is the entropy, that is H(s u + (1 - s) y_v)
          = - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) y_{v,d}) ;
    note that the choosen order of the arguments in the Kullback-Leibler
    does not favor the entropy of x (H(s u + (1 - s) y_v) is a constant),
    hence this loss is actually equivalent to cross-entropy.

    INPUTS: real numeric type is either float32 or float64, not both;
            indices numeric type can be int32 or uint32.

    NOTA: by default, components are identified using uint16 identifiers; 
    this can be easily changed in the wrapper source if more than 65535
    components are expected (recompilation is necessary)

    loss - 0 for linear, 1 for quadratic, 0 < loss < 1 for smoothed 
        Kullback-Leibler (see above)
    Y - observations, (real) D-by-V array;
        for Kullback-Leibler loss, the value at each vertex must lie on the
        probability simplex
        careful to the internal memory representation of multidimensional
        arrays; the C++ implementation uses column-major order (F-contiguous);
        usually numpy uses row-major order (C-contiguous), but this can often 
        be taken care of without actually copying data by using transpose();
    first_edge, adj_vertices - graph forward-star representation:
        vertices are numeroted (start at 0) in the order they are given in Y;
            careful to the internal memory representation of multidimensional
            arrays, usually numpy uses row-major order (C-contiguous)
        edges are numeroted (start at 0) so that all edges originating
            from a same vertex are consecutive;
        for each vertex, 'first_edge' indicates the first edge starting from 
            the vertex (or, if there are none, starting from the next vertex);
            (int 32 or uint32) array of length V + 1, the last value is the
            total number of edges;
        for each edge, 'adj_vertices' indicates its ending vertex, (int32 or
            uint32) array of length E
    edge_weights - weights on the edges (w_d1_uv in the above notations);
        (real) array of length E, or scalar for homogeneous weights
    loss_weights - weights on vertices (w_v in the above notations);
        (real) array of length V
    d11_metric - diagonal metric on the d11 penalization (w_d1_d above);
        (real) array of length D; all weights must be strictly positive, and it
        is advised to normalize the weights so that the first value is unity
    cp_dif_tol - stopping criterion on iterate evolution; algorithm stops if
        relative changes (in Euclidean norm) is less than dif_tol;
        1e-3 is a typical value; a lower one can give better precision
        but with longer computational time and more final components
    cp_it_max - maximum number of iterations (graph cut and subproblem)
        10 cuts solve accurately most problems
    K - number of alternative descent directions considered in the split step
    split_iter_num - number of partition-and-update iterations in the split 
        step
    split_damp_ratio - edge weights damping for favoring splitting; edge
        weights increase in linear progression along partition-and-update
        iterations, from this ratio up to original value; real scalar between 0
        and 1, the latter meaning no damping
    split_values_init_num - number of random initializations when looking for
        descent directions in the split step
    split_values_iter_num - number of refining iterations when looking for
        descent directions in the split step
    pfdr_rho - relaxation parameter, 0 < rho < 2
        1 is a conservative value; 1.5 often speeds up convergence
    pfdr_cond_min - stability of preconditioning; 0 < cond_min < 1;
        corresponds roughly the minimum ratio to the maximum descent metric;
        1e-2 is typical; a smaller value might enhance preconditioning
    pfdr_dif_rcd - reconditioning criterion on iterate evolution;
        a reconditioning is performed if relative changes of the iterate drops
        below dif_rcd; WARNING: reconditioning might temporarily draw minimizer
        away from the solution set and give bad subproblem solutions
    pfdr_dif_tol - stopping criterion on iterate evolution; algorithm stops if
        relative changes (in Euclidean norm) is less than dif_tol
        1e-2*cp_dif_tol is a conservative value
    pfdr_it_max - maximum number of iterations
        1e4 iterations provides enough precision for most subproblems
    verbose - if nonzero, display information on the progress, every 'verbose'
        PFDR iterations
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

    H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing Nonsmooth
    Functionals with Graph Total Variation, International Conference on Machine
    Learning, PMLR, 2018, 80, 4244-4253

    H. Raguet, A Note on the Forward-Douglas--Rachford Splitting for Monotone 
    Inclusion and Convex Optimization Optimization Letters, 2018, 1-24

    Baudoin Camille 2019, Raguet Hugo 2021
    """
    
    # Determine the type of float argument (real_t) 
    # real_t type is determined by the first parameter Y 
    if Y.size > 0 and Y.dtype == "float64":
        real_t = "float64" 
    elif Y.size > 0 and Y.dtype == "float32":
        real_t = "float32" 
    else:
        raise TypeError("Cut-pursuit d1 loss simplex: argument 'Y' must be a "
                        "nonempty numpy array of type float32 or float64.") 
    
    # Check numpy arrays
    if type(Y) != np.ndarray:
        raise TypeError("Cut-pursuit d1 loss simplex: argument 'Y' must be a "
                        "numpy array.")

    if (type(first_edge) != np.ndarray
        or first_edge.dtype not in ["int32", "uint32"]):
        raise TypeError("Cut-pursuit d1 loss simplex: argument 'first_edge' "
                        "must be a numpy array of type int32 or uint32.")

    if (type(adj_vertices) != np.ndarray
        or adj_vertices.dtype not in ["int32", "uint32"]):
        raise TypeError("Cut-pursuit d1 loss simplex: argument 'adj_vertices' "
                        "must be a numpy array of type int32 or uint32.")

    if type(edge_weights) != np.ndarray:
        if type(edge_weights) == list:
            raise TypeError("Cut-pursuit d1 loss simplex: argument "
                            "'edge_weights' must be a scalar or a numpy array")
        elif edge_weights != None:
            edge_weights = np.array([edge_weights], dtype=real_t)
        else:
            edge_weights = np.array([1.0], dtype=real_t)
        
    if type(loss_weights) != np.ndarray:
        if loss_weights != None:
            raise TypeError("Cut-pursuit d1 loss simplex: argument "
                "'loss_weights' must be a a numpy array")
        else:
            loss_weights = np.array([], dtype=real_t)

    if type(d11_metric) != np.ndarray:
        if d11_metric != None:
            raise TypeError("Cut-pursuit d1 loss simplex: argument "
                "'d11_metric' must be a numpy array.")
        else:
            d11_metric = np.array([], dtype=real_t)

    # Check graph structure 
    if first_edge.size != Y.shape[1] + 1 :
        raise ValueError("Cut-pursuit d1 loss simplex: argument 'first_edge'"
                         "should contain |V| + 1 = {0} elements, but {1} are "
                         "given".format(Y.shape[1] + 1, first_edge.size))
 
    # Check type of all numpy.array arguments of type float
    for name, ar_args in zip(
        ["Y", "edge_weights", "loss_weights", "d11_metric"],
        [ Y ,  edge_weights ,  loss_weights ,  d11_metric ]):
        if ar_args.dtype != real_t:
            raise TypeError("Cut-pursuit d1 loss simplex: argument '{0}' must "
                "be of type '{1}'".format(name, real_t))

    # Check fortran continuity of all multidimensional numpy.array arguments
    if not(Y.flags["F_CONTIGUOUS"]):
        raise TypeError("Cut-pursuit d1 loss simplex: argument 'Y' must be in "
            "column-major order (F-contigous).")

    # Convert in float64 all float arguments
    loss = float(loss)
    split_damp_ratio = float(split_damp_ratio)
    cp_dif_tol = float(cp_dif_tol)
    if pfdr_dif_tol is None:
        pfdr_dif_tol = 1e-2*cp_dif_tol
    pfdr_rho = float(pfdr_rho)
    pfdr_cond_min = float(pfdr_cond_min)
    pfdr_dif_rcd = float(pfdr_dif_rcd)
    pfdr_dif_tol = float(pfdr_dif_tol)
     
    # Convert all int arguments
    cp_it_max = int(cp_it_max)
    K = int(K)
    split_iter_num = int(split_iter_num)
    split_values_init_num = int(split_values_init_num)
    split_values_iter_num = int(split_values_iter_num)
    pfdr_it_max = int(pfdr_it_max)
    verbose = int(verbose)
    max_num_threads = int(max_num_threads)
    if max_split_size is None:
        max_split_size = Y.shape[1]
    else:
        max_split_size = int(max_split_size)

    # Check type of all booleen arguments
    for name, b_args in zip(
        ["balance_parallel_split", "compute_List", "compute_Graph",
         "compute_Obj", "compute_Time", "compute_Dif"],
        [ balance_parallel_split ,  compute_List ,  compute_Graph ,
          compute_Obj ,  compute_Time ,  compute_Dif ]):
        if type(b_args) != bool:
            raise TypeError("Cut-pursuit d1 loss simplex: argument '{0}' must "
                            "be boolean".format(name))

    # Call wrapper python in C  
    return cp_d1_lsx_cpy(loss, Y, first_edge, adj_vertices, edge_weights,
        loss_weights, d11_metric, cp_dif_tol, cp_it_max, K, split_iter_num,            split_damp_ratio, split_values_init_num, split_values_iter_num,
        pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max,
        verbose, max_num_threads, max_split_size, balance_parallel_split,
        real_t == "float64", compute_List, compute_Graph, compute_Obj,
        compute_Time, compute_Dif)
