import numpy as np
import os 
import sys

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                              "../bin"))

from pfdr_d1_lsx_cpy import pfdr_d1_lsx_cpy

def pfdr_d1_lsx(loss, Y, edges, edge_weights=None, loss_weights=None, 
                d11_metric=None, rho=1.5, cond_min=1e-2, 
                dif_rcd=0.0, dif_tol=1e-2, dif_it=16, it_max=1e3,
                verbose=int(1e1), compute_Obj=False, compute_Dif=False):

    """
    X, [Obj, Dif] = pfdr_d1_lsx(
            loss, Y, edges, edge_weights=None, loss_weights=None, 
            d11_metric=None, rho=1.0, cond_min=1e-2, 
            dif_rcd=0.0, dif_tol=1e-2, dif_it=16, it_max=1e3, verbose=1e1, 
            compute_Obj=False, compute_Dif=False)

    PFDR with d1 (total variation) penalization, with a separable loss term and
    simplex constraints:

    minimize functional over a graph G = (V, E)

        F(x) = f(x) + ||x||_d1 + i_{simplex}(x)

    where for each vertex, x_v is a D-dimensional vector,
          f is a separable data-fidelity loss
          ||x||_d1 = sum_{uv in E} w_d1_uv (sum_d w_d1_d |x_ud - x_vd|),
    and i_{simplex} is the standard D-simplex constraint over each vertex,
        i_{simplex} = 0 for all v, (for all d, x_vd >= 0) and sum_d x_vd = 1,
                    = infinity otherwise;

    using preconditioned forward-Douglas-Rachford splitting algorithm.

    Available separable data-fidelity loss include:

    linear
        f(x) = - <x, y> ,  with  <x, y> = sum_{v,d} x_{v,d} y_{v,d};

    quadratic
        f(x) = 1/2 ||y - x||_{l2,w}^2 ,
    with  ||y - x||_{l2,w}^2 = sum_{v,d} w_v (y_{v,d} - x_{v,d})^2;

    smoothed Kullback-Leibler divergence (cross-entropy)
        f(x) = sum_v w_v KLs(x_v, y_v),
    with KLs(y_v, x_v) = KL(s u + (1 - s) y_v ,  s u + (1 - s) x_v), where
        KL is the regular Kullback-Leibler divergence,
        u is the uniform discrete distribution over {1,...,D}, and
        s = loss is the smoothing parameter;
    it yields
        KLs(y_v, x_v) = - H(s u + (1 - s) y_v)
            - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) x_{v,d}) ,
    where H is the entropy, that is H(s u + (1 - s) y_v)
          = - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) y_{v,d}) ;
    note that the choosen order of the arguments in the Kullback-Leibler
    does not favor the entropy of x (H(s u + (1 - s) y_v) is a constant),
    hence this loss is actually equivalent to cross-entropy.

    INPUTS: real numeric type is either float32 or float64, not both;
            indices numeric type is uint32.

    loss - 0 for linear, 1 for quadratic, 0 < loss < 1 for smoothed 
        Kullback-Leibler (see above)
    Y - observations, (real) D-by-V array, column-major (F-contigous) format;
        careful to the internal memory representation of multidimensional
        arrays, usually numpy uses row-major (C-contiguous) format
        (convert to F_CONTIGUOUS without copying data by using transpose);
        for Kullback-Leibler loss, the value at each vertex must lie on the
        probability simplex
    edges - list of edges, (uint32) array of length 2E;
        vertices are numeroted (start at 0) in the order they are given in Y;
        edge number e connects vertices given by edges[2*e] and edges[2*e + 1];
        every vertex should belong to at least one edge with a nonzero
        penalization coefficient. If it is not the case, a workaround is to add
        an edge from the isolated vertex to itself with a small nonzero weight
    edge_weights - weights on the edges (w_d1_uv in the above notations);
        (real) array of length E or scalar for homogeneous weights
    loss_weights - weights on vertices (w_v in the above notations);
        (real) array of length V
    d11_metric - diagonal metric on the d11 penalization (w_d1_d above);
        (real) array of length D; all weights must be strictly positive, and it
        is advised to normalize the weights so that the first value is unity
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

    H. Raguet, A Note on the Forward-Douglas--Rachford Splitting for Monotone 
    Inclusion and Convex Optimization Optimization Letters, 2018, 1-24

    Baudoin Camille 2019, Raguet Hugo 2022, 2023
    """
    
    # Determine the type of float argument (real_t) 
    # real_t type is determined by the first parameter Y 
    if type(Y) != np.ndarray:
        raise TypeError("argument 'Y' must be a numpy array")

    if Y.any() and Y.dtype == "float64":
        real_t = "float64" 
    elif Y.any() and Y.dtype == "float32":
        real_t = "float32" 
    else:
        raise TypeError("argument 'Y' must be a nonempty numpy array of type "
                        "float32 or float64") 
    
    # Convert in numpy array scalar entry: Y, edges, 
    # edge_weights, loss_weights, d11_metric and define float numpy array
    # argument with the right float type, if empty:
    if type(edges) != np.ndarray or edges.dtype != "uint32":
        raise TypeError("argument 'edges' must be a numpy array of type"
                        "uint32")

    if type(edge_weights) != np.ndarray:
        if type(edge_weights) == list:
            raise TypeError("argument 'edge_weights' must be a scalar or a "
                            "numpy array")
        elif edge_weights != None:
            edge_weights = np.array([edge_weights], dtype=real_t)
        else:
            edge_weights = np.array([1.0], dtype=real_t)
        
    if type(loss_weights) != np.ndarray:
        if type(loss_weights) == list:
            raise TypeError("argument 'loss_weights' must be a scalar or a "
                            "numpy array")
        elif loss_weights != None:
            loss_weights = np.array([loss_weights], dtype=real_t)
        else:
            loss_weights = np.array([], dtype=real_t)

    if type(d11_metric) != np.ndarray:
        if d11_metric != None:
            raise TypeError("argument 'd11_metric' must be a numpy array")
        else:
            d11_metric = np.array([], dtype=real_t)

    # Check type of all numpy.array arguments of type float (Y, edge_weights,
    # loss_weights, d11_metric) 
    for name, ar_args in zip(
            ["Y", "edge_weights", "loss_weights", "d11_metric"],
            [Y, edge_weights, loss_weights, d11_metric]):
        if ar_args.dtype != real_t:
            raise TypeError("argument '{0}' must be of type '{1}'"
                            .format(name, real_t))

    # Check fortran continuity of all multidimensional numpy.array arguments
    if not(Y.flags["F_CONTIGUOUS"]):
        raise TypeError("argument 'Y' must be F_CONTIGUOUS")

    # Convert in float64 all float arguments if needed (rho,
    # cond_min, dif_rcd, dif_tol) 
    loss = float(loss)
    rho = float(rho)
    cond_min = float(cond_min)
    dif_rcd = float(dif_rcd)
    dif_tol = float(dif_tol)
     
    # Convert all int arguments (it_max, verbose) in ints: 
    dif_it = int(dif_it)
    it_max = int(it_max)
    verbose = int(verbose)

    # Check type of all booleen arguments (compute_Obj, compute_Dif)
    for name, b_args in zip(
        ["compute_Obj", "compute_Dif"],
        [compute_Obj, compute_Dif]):
        if type(b_args) != bool:
            raise TypeError("argument '{0}' must be boolean".format(name))

    # Call wrapper python in C  
    return pfdr_d1_lsx_cpy(loss, Y, edges, edge_weights, loss_weights,
        d11_metric, rho, cond_min, dif_rcd, dif_tol, dif_it, it_max,
        verbose, real_t == "float64", compute_Obj, compute_Dif)

