import numpy as np
import os 
import sys

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                              "../bin"))

from pfdr_prox_tv_cpy import pfdr_prox_tv_cpy

def pfdr_prox_tv(Y, edges, l22_metric=None, edge_weights=None, d1p=2,
    d1p_metric=None, rho=1.5, cond_min=1e-2, dif_rcd=0.0, dif_tol=1e-4,
    dif_it=32, it_max=1e3, verbose=1e2, compute_Obj=False, compute_Dif=False):

    """
    X, [Obj, Dif] = pfdr_prox_tv(Y, edges, l22_metric=None, edge_weights=None,
        d1p=2, d1p_metric=None, rho=1.0, cond_min=1e-2, dif_rcd=0.0,
        dif_tol=1e-3, dif_it=32, it_max=1e3, verbose=1e2, compute_Obj=False,
        compute_Dif=False)

    Compute the proximal operator of the d1 (total variation) penalization:

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

    using preconditioned forward-Douglas-Rachford splitting algorithm.

    INPUTS: real numeric type is either float32 or float64, not both;
            indices numeric type can be int32 or uint32.

    Y - observations, (real) D-by-V array, column-major (F-contigous) format;
        careful to the internal memory representation of multidimensional
        arrays, usually numpy uses row-major (C-contiguous) format
        (convert to F_CONTIGUOUS without copying data by using transpose);
    edges - list of edges, (int32 or uint32) array of length 2E;
        vertices are numeroted (start at 0) in the order they are given in Y;
        edge number e connects vertices given by edges[2*e] and edges[2*e + 1];
        every vertex should belong to at least one edge with a nonzero
        penalization coefficient. If it is not the case, a workaround is to add
        an edge from the isolated vertex to itself with a small nonzero weight
    l22_metric - diagonal metric on squared l2-norm (matrix Ms above);
        array or length V for weights depending only on vertices,
        D-by-V array otherwise; weights must be strictly positive
    edge_weights - weights on the edges (w_uv in the above notations);
        (real) array of length E or scalar for homogeneous weights
    d1p - define the total variation as the l11- (d1p = 1) or l12- (d1p = 2)
        norm of the finite differences
    d1p_metric - diagonal metric on d1p penalisation (Md in above notations);
        all weights must be strictly positive, and it is advised to normalize
        the weights so that the first value is unity for computation stability
    rho - relaxation parameter, 0 < rho < 2
        1 is a conservative value; 1.5 often speeds up convergence
    cond_min - stability of preconditioning; 0 < cond_min < 1;
        corresponds roughly the minimum ratio to the maximum descent metric;
        1e-2 is typical; a smaller value might enhance preconditioning
    dif_rcd - reconditioning criterion on iterate evolution;
        a reconditioning is performed if relative changes of the iterate drops
        below dif_rcd;
        10*dif_tol is a typical value, 1e2*dif_tol or 1e3*dif_tol might speed
        up convergence; WARNING: reconditioning might temporarily draw
        minimizer away from solution, so it is advised to monitor objective
        value when using reconditioning
    dif_tol - stopping criterion on iterate evolution; algorithm stops if
        relative changes (in l1 norm) is less than dif_tol;
        1e-3 is a typical value; a lower one can give better precision but with
        longer computational time
    dif_it - number of iterations between iterates for evolution measure
    it_max - maximum number of iterations
        usually depends on the size of the problems in relation to the
        available computational budget
    verbose - if nonzero, display information on the progress, every 'verbose'
        PFDR iterations
    compute_Obj - compute the objective functional along iterations 
    compute_Dif - compute relative evolution along iterations 

    OUTPUTS: Obj, Dif are optional outputs, set optional input
        compute_Obj, compute_Dif to True to get them 

    X - final minimizer, array of length V (real)
    Obj - if requested ,the values of the objective functional along iterations
          (array of length number of iterations + 1)
    Dif - if requested, the iterate evolution along iterations
           (array of length number of iterations)
     
    Parallel implementation with OpenMP API.

    H. Raguet and L. Landrieu, Preconditioning of a Generalized
    Forward-Backward Splitting and Application to Optimization on Graphs, SIAM
    Journal on Imaging Sciences, 2015, 8, 2706-2739

    Baudoin Camille 2019, Raguet Hugo 2022, 2023
    """
    
    # Determine the type of float argument (real_t) 
    # real_t type is determined by the first parameter Y 
    if type(Y) != np.ndarray:
        raise TypeError("PFDR prox TV: argument 'Y' must be a numpy array")

    if Y.any() and Y.dtype == "float64":
        real_t = "float64" 
    elif Y.any() and Y.dtype == "float32":
        real_t = "float32" 
    else:
        raise TypeError("PFDR prox TV: argument 'Y' must be a nonempty numpy "
            "array of type float32 or float64") 
    
    # Convert in numpy array scalar entry: Y, edges, edge_weights, d1p_metric
    # and define float numpy array argument with the right float type, if empty
    if type(edges) != np.ndarray or edges.dtype not in ["int32", "uint32"]:
        raise TypeError("PFDR prox TV: argument 'edges' must be a numpy array"
            "of type int 32 or uint32")

    if type(l22_metric) != np.ndarray:
        if l22_metric != None:
            raise TypeError("PFDR prox TV: argument 'l22_metric' must be a "
                "numpy array")
        else:
            l22_metric = np.array([], dtype=real_t)

    if type(edge_weights) != np.ndarray:
        if type(edge_weights) == list:
            raise TypeError("PFDR prox TV: argument 'edge_weights' must be a"
                " scalar or a numpy array")
        elif edge_weights != None:
            edge_weights = np.array([edge_weights], dtype=real_t)
        else:
            edge_weights = np.array([1.0], dtype=real_t)
        
    if type(d1p_metric) != np.ndarray:
        if d1p_metric != None:
            raise TypeError("PFDR prox TV: argument 'd1p_metric' must be a "
                "numpy array")
        else:
            d1p_metric = np.array([], dtype=real_t)

    # Check type of all numpy.array arguments of type float (Y, edge_weights,
    # d1p_metric) 
    for name, ar_args in zip(
            ["Y", "edge_weights", "d1p_metric"],
            [Y, edge_weights, d1p_metric]):
        if ar_args.dtype != real_t:
            raise TypeError("PFDR prox TV: argument '{0}' must be of type "
                "'{1}'".format(name, real_t))

    # Check fortran continuity of all multidimensional numpy.array arguments
    if not(Y.flags["F_CONTIGUOUS"]):
        raise TypeError("PFDR prox TV: argument 'Y' must be F_CONTIGUOUS")

    # Convert in float64 all float arguments if needed (rho, cond_min, dif_rcd,
    # dif_tol) 
    rho = float(rho)
    cond_min = float(cond_min)
    dif_rcd = float(dif_rcd)
    dif_tol = float(dif_tol)
     
    # Convert all int arguments (it_max, verbose) in int
    d1p = int(d1p)
    dif_it = int(dif_it)
    it_max = int(it_max)
    verbose = int(verbose)

    # Check type of all booleen arguments (compute_Obj, compute_Dif)
    for name, b_args in zip(
        ["compute_Obj", "compute_Dif"],
        [compute_Obj, compute_Dif]):
        if type(b_args) != bool:
            raise TypeError("PFDR prox TV: argument '{0}' must be boolean"
                .format(name))

    # Call wrapper python in C  
    return pfdr_prox_tv_cpy(Y, edges, l22_metric, edge_weights, d1p,
        d1p_metric, rho, cond_min, dif_rcd, dif_tol, dif_it, it_max, verbose,
        real_t == "float64", compute_Obj, compute_Dif)

