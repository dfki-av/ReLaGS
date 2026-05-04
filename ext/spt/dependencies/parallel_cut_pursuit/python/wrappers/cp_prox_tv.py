import numpy as np
import os 
import sys

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                                              "../bin"))

from cp_prox_tv_cpy import cp_prox_tv_cpy

def cp_prox_tv(Y, first_edge, adj_vertices, l22_metric=None, edge_weights=None,
    d1p=2, d1p_metric=None, cp_dif_tol=1e-4, cp_it_max=10, K=2,
    split_iter_num=1, split_damp_ratio=1.0, split_values_init_num=2,
    split_values_iter_num=2, pfdr_rho=1., pfdr_cond_min=1e-2,
    pfdr_dif_rcd=0., pfdr_dif_tol=None, pfdr_it_max=int(1e4), verbose=int(1e3),
    max_num_threads=0, max_split_size=None, balance_parallel_split=True,
    compute_List=False, compute_Graph=False, compute_Obj=False,
    compute_Time=False, compute_Dif=False):
    """
    Comp, rX, [List, Graph, Obj, Time, Dif] = cp_prox_tv(Y, first_edge,
        adj_vertices, l22_metric=None, edge_weights=1.0, d1p=2,
        d1p_metric=None, cp_dif_tol=1e-4, cp_it_max=10, K=2, split_iter_num=1,
        split_damp_ratio=1.0, split_values_init_num=2, split_values_iter_num=2,
        pfdr_rho=1.0, pfdr_cond_min=1e-2, pfdr_dif_rcd=0.0,
        pfdr_dif_tol=1e-2*cp_dif_tol, pfdr_it_max=int(1e4), verbose=int(1e3),
        max_num_threads=0, max_split_size=None, balance_parallel_split=True,
        compute_List=False, compute_Graph=False, compute_Obj=False,
        compute_Time=False, compute_Dif=False)

    Compute the proximal operator of the graph total variation penalization:
                                                                              
     minimize functional F defined over a graph G = (V, E)
    
     F: R^{D-by-V} -> R
            x      -> 1/2 ||y - x||_Ms^2 + ||x||_d1p
    
     where y in R^{D-by-V}, Ms is a diagonal metric in R^{VxD-by-VxD} so that
    
          ||y - x||_Ms^2 = sum_{v in V} sum_d ms_v_d (y_v - x_v)^2 ,
    
     and
     
          ||x||_d1p = sum_{uv in E} w_uv ||x_u - x_v||_{Md,p} ,
    
     where Md is a diagonal metric and p can be 1 or 2,
      ||x_v||_{M,1} = sum_d md_d |x_v|  (weighted l1-norm) or
      ||x_v||_{M,2} = sqrt(sum_d md_d x_v^2)  (weighted l2-norm)
                                                                              
    using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
    splitting algorithm.

    INPUTS: real numeric type is either float32 or float64, not both;
            indices numeric type can be int32 or uint32.

    NOTA: by default, components are identified using uint16 identifiers; 
    this can be easily changed in the wrapper source if more than 65535
    components are expected (recompilation is necessary)

    Y - observations, (real) array of length V;
        careful to the internal memory representation of multidimensional
        arrays; the C++ implementation uses column-major order (F-contiguous);
        usually numpy uses row-major order (C-contiguous), but this can often
        be taken care of without actually copying data by using transpose();
    first_edge, adj_vertices - graph forward-star representation:
        vertices are numeroted (start at 0) in the order given in Y
            careful to the internal memory representation of multidimensional
            arrays, usually numpy uses row-major order (C-contiguous)
        edges are numeroted (start at 0) so that all edges originating
            from a same vertex are consecutive;
        for each vertex, 'first_edge' indicates the first edge starting 
            from the vertex (or, if there are none, starting from the next 
            vertex); (int32 or uint32) array of length V + 1, the last value
            is the total number of edges;
        for each edge, 'adj_vertices' indicates its ending vertex, (int32 or
            uint32) array of length E
    l22_metric - diagonal metric on squared l2-norm (Ms in above notations);
        array or length V for weights depending only on vertices, D-by-V array
        otherwise; all weights must be strictly positive
    edge_weights - weights on the edges (w_uv above);
        (real) array of length E, or scalar for homogeneous weights
    d1p - define the total variation as the l11- (d1p = 1) or l12- (d1p = 2)
        norm of the finite differences
    d1p_metric - diagonal metric on d1p penalisation (Md in above notations);
        all weights must be strictly positive, and it is advised to normalize
        the weights so that the first value is unity for computation stability
    cp_dif_tol - stopping criterion on iterate evolution; algorithm stops if
        relative changes (in Euclidean norm) is less than dif_tol;
        1e-4 is a typical value; a lower one can give better precision
        but with longer computational time and more final components
    cp_it_max - maximum number of iterations (graph cut and subproblem)
        10 cuts solve accurately most problems
    K - number of alternative descent directions considered in the split step;
        makes sense only for D > 1
    split_iter_num - number of partition-and-update iterations in the split 
        step; makes sense only for D > 1
    split_damp_ratio - edge weights damping for favoring splitting; edge
        weights increase in linear progression along partition-and-update
        iterations, from this ratio up to original value; real scalar between 0
        and 1, the latter meaning no damping; makes sense only for D > 1
    split_values_init_num - number of random initializations when looking for
        descent directions in the split step; makes sense only for D > 1
    split_values_iter_num - number of refining iterations when looking for
        descent directions in the split step; makes sense only for D > 1
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
    pfdr_it_max - maximum number of iterations; 1e4 iterations provides enough
        precision for most subproblems
    verbose - if nonzero, display information on the progress, every 'verbose'
        PFDR iterations
    max_num_threads - if greater than zero, set the maximum number of threads
        used for parallelization with OpenMP
    max_split_size - maximum number of vertices allowed in connected component
        passed to a split problem; make split of very large components faster,
        but might induced suboptimal artificial cuts
    balance_parallel_split - if true, the parallel workload of the split step 
        is balanced; WARNING: this might trade off speed against optimality
    compute_List - report the list of vertices constituting each component
    compute_Graph - get the reduced graph on the components
    compute_Obj - compute the objective functional along iterations 
    compute_Time - monitor elapsing time along iterations
    compute_Dif - compute relative evolution along iterations 

    OUTPUTS: List, Graph, Obj, Time and Dif are optional, set parameters
        compute_List, compute_Graph, compute_Obj, compute_Time or compute_Dif
        to True to request them and capture them in output variables in that
        order

    Comp - assignement of each vertex to a component, (uint16) array of
        length V
    rX - values of each component of the minimizer, (real) array of length rV;
        the actual minimizer is then reconstructed as X = rX[Comp];
    List - if requested, list of vertices constituting each component; python
        list of length rV, containing (uint32) arrays of indices
    Graph - if requested, reduced graph structure; python tuple of length 3
        representing the graph as forward-star (see input first_edge and
        adj_vertices) together with edge weights
    Obj - if requested, the values of the objective functional along
        iterations, up to the constant 1/2||Y||^2;
        array of length the actual number of iterations performed + 1;
    Time - if requested, the elapsed time along iterations;
        array of length the actual number of iterations performed + 1
    Dif  - if requested, the iterate evolution along iterations;
        array of length the actual number of iterations performed

    Parallel implementation with OpenMP API.

    L. Landrieu and G. Obozinski, Cut Pursuit: Fast Algorithms to Learn
    Piecewise Constant Functions on General Weighted Graphs, SIIMS, 10, 4,
    1724–1766, 2017.
 
    Hugo Raguet 2021, 2022
    """

    # Check numpy arrays
    if type(Y) != np.ndarray:
        raise TypeError("Cut-pursuit prox TV: argument 'Y' must be a numpy "
                        "array.")

    # real type is determined by Y
    real_t = Y.dtype
    if real_t not in ["float32", "float64"]:
        raise TypeError("Cut-pursuit prox TV: currently, the real numeric type"
                        " must be float32 or float64.")

    if (type(first_edge) != np.ndarray
        or first_edge.dtype not in ["int32", "uint32"]):
        raise TypeError("Cut-pursuit prox TV: argument 'first_edge' must be a "
                        "numpy array of type int32 or uint32.")

    if (type(adj_vertices) != np.ndarray
        or adj_vertices.dtype not in ["int32", "uint32"]):
        raise TypeError("Cut-pursuit prox TV: argument 'adj_vertices' must be "
                        "a numpy array of type int32 or uint32.")

    if type(l22_metric) != np.ndarray:
        if l22_metric != None:
            raise TypeError("Cut-pursuit prox TV: argument 'l22_metric' must"
                "be a a numpy array")
        else:
            l22_metric = np.array([], dtype=real_t)

    if type(edge_weights) != np.ndarray:
        if type(edge_weights) == list:
            raise TypeError("Cut-pursuit prox TV: argument 'edge_weights' must"
                            " be a scalar or a numpy array.")
        elif edge_weights != None:
            edge_weights = np.array([edge_weights], dtype=real_t)
        else:
            edge_weights = np.array([1.0], dtype=real_t)

    if type(d1p_metric) != np.ndarray:
        if d1p_metric != None:
            raise TypeError("Cut-pursuit prox TV: argument 'd1p_metric' must"
                "be a a numpy array")
        else:
            d1p_metric = np.array([], dtype=real_t)

    # Determine V and check the graph structure
    if first_edge.size != Y.shape[1] + 1:
        raise ValueError("Cut-pursuit prox TV: argument 'first_edge' should "
                         "contain |V| + 1 = {0} elements, but {1} are given."
                         .format(V + 1, first_edge.size))
 
    # Check type of all numpy.array arguments of type float
    for name, ar_args in zip(
        ["Y", "l22_metric", "edge_weights", "d1p_metric"],
        [ Y,   l22_metric ,  edge_weights ,  d1p_metric ]):
        if ar_args.dtype != real_t:
            raise TypeError("Cut-pursuit prox TV: argument '{0}' must be of "
                "type '{1}'".format(name, real_t))

    # Check fortran continuity of all multidimensional numpy array arguments
    if not(Y.flags["F_CONTIGUOUS"]):
        raise TypeError("Cut-pursuit prox TV: argument 'Y' must be in "
            "column-major order (F-contigous).")

    # ensure scalar arguments of the correct type
    cp_dif_tol = float(cp_dif_tol)
    if pfdr_dif_tol is None:
        pfdr_dif_tol = 1e-2*cp_dif_tol
    pfdr_rho = float(pfdr_rho)
    pfdr_cond_min = float(pfdr_cond_min)
    pfdr_dif_rcd = float(pfdr_dif_rcd)
    pfdr_dif_tol = float(pfdr_dif_tol)
     
    # Convert all int arguments 
    d1p = int(d1p)
    cp_it_max = int(cp_it_max)
    if Y.shape[0] == 1:
        K = 2
        split_iter_num = 1
        split_damp_ratio = 1.0
        split_values_init_num = 1
        split_values_iter_num = 1
    else:
        K = int(K)
        split_iter_num = int(split_iter_num)
        split_damp_ratio = float(split_damp_ratio)
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
            raise TypeError("Cut-pursuit prox TV: argument '{0}' must be "
                "boolean".format(name))
    
    # Call wrapper python in C  
    return cp_prox_tv_cpy(Y, first_edge, adj_vertices, l22_metric,
        edge_weights, d1p, d1p_metric, cp_dif_tol, cp_it_max, K,
        split_iter_num, split_damp_ratio, split_values_init_num,
        split_values_iter_num, pfdr_rho, pfdr_cond_min, pfdr_dif_rcd,
        pfdr_dif_tol, pfdr_it_max, verbose, max_num_threads, max_split_size,
        balance_parallel_split, real_t == "float64", compute_List,
        compute_Graph, compute_Obj, compute_Time, compute_Dif) 
