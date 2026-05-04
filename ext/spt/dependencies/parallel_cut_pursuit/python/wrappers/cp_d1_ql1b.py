import numpy as np
import os 
import sys

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), 
                                              "../bin"))

from cp_d1_ql1b_cpy import cp_d1_ql1b_cpy

def cp_d1_ql1b(Y, A, first_edge, adj_vertices, edge_weights=None, 
               Yl1=None, l1_weights=None, low_bnd=None, upp_bnd=None, 
               cp_dif_tol=1e-4, cp_it_max=10, pfdr_rho=1., pfdr_cond_min=1e-2,
               pfdr_dif_rcd=0., pfdr_dif_tol=None, pfdr_it_max=int(1e4),
               verbose=int(1e3), max_num_threads=0, max_split_size=None,
               balance_parallel_split=True, Gram_if_square=True,
               compute_List=False, compute_Graph=False, compute_Obj=False,
               compute_Time=False, compute_Dif=False):
    """
    Comp, rX, [List, Graph, Obj, Time, Dif] = cp_d1_ql1b(Y | AtY, A | AtA,
            first_edge, adj_vertices, edge_weights=None, Yl1=None,
            l1_weights=None, low_bnd=None, upp_bnd=None, cp_dif_tol=1e-4,
            cp_it_max=10, pfdr_rho=1.0, pfdr_cond_min=1e-2, pfdr_dif_rcd=0.0,
            pfdr_dif_tol=1e-2*cp_dif_tol, pfdr_it_max=int(1e4),
            verbose=int(1e3), Gram_if_square=True, max_num_threads=0,
            max_split_size=None, balance_parallel_split=True,
            compute_List=False, compute_Graph=False, compute_Obj=False,
            compute_Time=False, compute_Dif=False)

    Cut-pursuit algorithm with d1 (total variation) penalization, with a 
    quadratic functional, l1 penalization and box constraints:

    minimize functional over a graph G = (V, E)

        F(x) = 1/2 ||y - A x||^2 + ||x||_d1 + ||yl1 - x||_l1 + i_[m,M](x)

    where y in R^N, x in R^V, A in R^{N-by-|V|}
          ||x||_d1 = sum_{uv in E} w_d1_uv |x_u - x_v|,
          ||x||_l1 = sum_{v  in V} w_l1_v |x_v|,
          and the convex indicator
          i_[m,M] = infinity if it exists v in V such that x_v < m_v or 
          x_v > M_v
                  = 0 otherwise;

    using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
    splitting algorithm.

    It is easy to introduce a SDP metric weighting the squared l2-norm
    between y and A x. Indeed, if M is the matrix of such a SDP metric,
    ||y - A x||_M^2 = ||Dy - D A x||^2, with D = M^(1/2).
    Thus, it is sufficient to call the method with Y <- Dy, and A <- D A.
    Moreover, when A is the identity and M is diagonal (weighted square l2 
    distance between x and y), one should call on the precomposed version 
    (see below) with Y <- DDy = My and A <- D2 = M.

    INPUTS: real numeric type is either float32 or float64, not both;
            indices numeric type can be int32 or uint32.

    NOTA: by default, components are identified using uint16 identifiers; 
    this can be easily changed in the wrapper source if more than 65535
    components are expected (recompilation is necessary)

    Y - observations, (real) array of length N (direct matricial case) or
                             array of length V (left-premult. by A^t), or
                             empty matrix (for all zeros)
    A - matrix, (real) N-by-V array (direct matricial case), or
                       V-by-V array (premultiplied to the left by A^t), or
                       V-by-1 array (_square_ diagonal of A^t A = A^2), or
                       nonzero scalar (for identity matrix), or
                       zero scalar (for no quadratic part);
        for an arbitrary scalar matrix, use identity and scale observations
        and penalizations accordingly
        if N = V in a direct matricial case, the last argument 'Gram_if_square'

      careful to the internal memory representation of multidimensional
      arrays; the C++ implementation uses column-major order (F-contiguous);
      usually numpy uses row-major order (C-contiguous), but this can often be
      taken care of without actually copying data by using transpose();

    first_edge, adj_vertices - graph forward-star representation:
        vertices are numeroted (start at 0) in the order given in Y or A
            careful to the internal memory representation of multidimensional
            arrays, usually numpy uses row-major order (C-contiguous)
        edges are numeroted (start at 0) so that all edges originating
            from a same vertex are consecutive;
        for each vertex, 'first_edge' indicates the first edge starting 
            from the vertex (or, if there are none, starting from the next 
            vertex); (int 32 or uint32) array of length V + 1, the last value
            is the total number of edges;
        for each edge, 'adj_vertices' indicates its ending vertex, (int 32 or 
            uint32) array of length E
    edge_weights - (real) array of length E or a scalar for homogeneous weights
    Yl1 - offset for l1 penalty, (real) array of length V
    l1_weights - (real) array of length V or scalar for homogeneous weights
    low_bnd - (real) array of length V or scalar
    upp_bnd - (real) array of length V or scalar
    cp_dif_tol - stopping criterion on iterate evolution; algorithm stops if
        relative changes (in Euclidean norm) is less than dif_tol;
        1e-4 is a typical value; a lower one can give better precision
        but with longer computational time and more final components
    cp_it_max - maximum number of iterations (graph cut and subproblem)
        10 cuts solve accurately most problems
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
    pfdr_it_max - maximum number of iterations 1e4 iterations provides enough
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
    Gram_if_square - if A is square, set to false for direct matricial case
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
    rX - values of each component of the minimizer, (real) array of length rV;
        the actual minimizer is then reconstructed as X = rX[Comp];
    List - if requested, list of vertices constituting each component; python
        list of length rV, containing (uint32) arrays of indices
    Graph - if requested, reduced graph structure; python tuple of length 3
        representing the graph as forward-star (see input first_edge and
        adj_vertices) together with edge weights
    Obj - if requested, values of the objective functional along iterations;
        array of length actual number of cut-pursuit iterations performed + 1;
        NOTA: in the precomputed A^t A version, a constant 1/2||Y||^2 in the
        quadratic part is omited
    Time - if requested, the elapsed time along iterations; array of length
        actual number of cut-pursuit iterations performed + 1
    Dif - if requested, the iterate evolution along iterations; array of length
        actual number of cut-pursuit iterations performed

    Parallel implementation with OpenMP API.

    H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing Nonsmooth
    Functionals with Graph Total Variation, International Conference on Machine
    Learning, PMLR, 2018, 80, 4244-4253

    H. Raguet, A Note on the Forward-Douglas--Rachford Splitting for Monotone 
    Inclusion and Convex Optimization Optimization Letters, 2018, 1-24

    Baudoin Camille 2019, Hugo Raguet 2021
    """

    # Determine the type of float argument (real_t) 
    # real type is determined by Y or Yl1
    if type(Y) == np.ndarray and Y.size > 0:
        real_t = Y.dtype
    elif type(Yl1) == np.ndarray and Yl1.size > 0:
        real_t = Yl1.dtype
    else:
        raise TypeError("Cut-pursuit d1 quadratic l1 bounds: at least one of "
                        "arguments 'Y' or 'Yl1' must be provided as a nonempty"
                        " numpy array.")

    if real_t not in ["float32", "float64"]:
        raise TypeError("Cut-pursuit d1 quadratic l1 bounds: currently, the "
                        "real numeric type must be float32 or float64.")

    # Check numpy arrays: Y, A, first_edge, adj_vertices, edge_weights, Yl1,
    # l1_weights, low_bnd, upp_bnd, and define float numpy array argument with
    # the right float type if necessary
    if type(Y) != np.ndarray:
        if Y == None:
            Y = np.array([], dtype=real_t)
        else:
            raise TypeError("Cut-pursuit d1 quadratic l1 bounds: argument 'Y' "
                            "must be a numpy array.")

    if type(A) != np.ndarray:
        if type(A) == list:
            raise TypeError("Cut-pursuit d1 quadratic l1 bounds: argument 'A' "
                            "must be a scalar or a numpy array.")
        else:
            A = np.array([A], real_t)

    if (type(first_edge) != np.ndarray
        or first_edge.dtype not in ["int32", "uint32"]):
        raise TypeError("Cut-pursuit d1 quadratic l1 bounds: argument "
            "'first_edge' must be a numpy array of type int32 or uint32.")

    if (type(adj_vertices) != np.ndarray
        or adj_vertices.dtype not in ["int32", "uint32"]):
        raise TypeError("Cut-pursuit d1 quadratic l1 bounds: argument "
            "'adj_vertices' must be a numpy array of type int32 or uint32.")

    if type(edge_weights) != np.ndarray:
        if type(edge_weights) == list:
            raise TypeError("Cut-pursuit d1 quadratic l1 bounds: argument "
                    "'edge_weights' must be a scalar or a numpy array.")
        elif edge_weights != None:
            edge_weights = np.array([edge_weights], dtype=real_t)
        else:
            edge_weights = np.array([1.0], dtype=real_t)

    if type(Yl1) != np.ndarray:
        if Yl1 == None:
            Yl1 = np.array([], dtype=real_t)
        else:
            raise TypeError("Cut-pursuit d1 quadratic l1 bounds: argument "
                            "'Yl1' must be a numpy array.")

    if type(l1_weights) != np.ndarray:
        if type(l1_weights) == list:
            raise TypeError("Cut-pursuit d1 quadratic l1 bounds: argument "
                            "'l1_weights' must be a scalar or a numpy array.")
        elif l1_weights != None:
            l1_weights = np.array([l1_weights], dtype=real_t)
        else:
            l1_weights = np.array([0.0], dtype=real_t)

    if type(low_bnd) != np.ndarray:
        if type(low_bnd) == list:
            raise TypeError("Cut-pursuit d1 quadratic l1 bounds: argument "
                            "'low_bnd' must be a scalar or a numpy array.")
        elif low_bnd != None:
            low_bnd = np.array([low_bnd], dtype=real_t)
        else: 
            low_bnd = np.array([-np.inf], dtype=real_t)

    if type(upp_bnd) != np.ndarray:
        if type(upp_bnd) == list:
            raise TypeError("Cut-pursuit d1 quadratic l1 bounds: argument "
                            "'upp_bnd' must be a scalar or a numpy array.")
        elif upp_bnd != None:
            upp_bnd = np.array([upp_bnd], dtype=real_t)
        else: 
            upp_bnd = np.array([np.inf], dtype=real_t)

    # Determine V and check the graph structure
    if A.ndim > 1 and A.shape[1] != 1:
        V = A.shape[1] 
    elif A.shape[0] == 1:
        if Y.size > 0:
            V = Y.size
        elif Yl1.size > 0:
            V = Yl1.size
    else:
        V = A.shape[0]
    
    if first_edge.size != V + 1:
        raise ValueError("Cut-pursuit d1 quadratic l1 bounds: argument "
                         "'first_edge' should contain |V| + 1 = {0} elements, "
                         "but {1} are given.".format(V + 1, first_edge.size))
 
    # Check type of all numpy.array arguments of type float
    for name, ar_args in zip(
            ["Y", "A", "edge_weights", "Yl1", "l1_weights", "low_bnd",
             "upp_bnd"],
            [ Y ,  A ,  edge_weights ,  Yl1 ,  l1_weights ,  low_bnd ,
              upp_bnd ]):
        if ar_args.dtype != real_t:
            raise TypeError("argument '{0}' must be of type '{1}'"
                            .format(name, real_t))

    # Check fortran continuity of all multidimensional numpy array arguments
    if not(Y.flags["F_CONTIGUOUS"]):
        raise TypeError("Cut-pursuit d1 quadratic l1 bounds: argument 'Y' "
                        "must be in column-major order (F-contigous).")
    if not(A.flags["F_CONTIGUOUS"]):
        raise TypeError("Cut-pursuit d1 quadratic l1 bounds: argument 'A' "
                        "must be in column-major order (F-contigous).")

    # Convert in float64 all float arguments
    if pfdr_dif_tol is None:
        pfdr_dif_tol = 1e-2*cp_dif_tol
    cp_dif_tol = float(cp_dif_tol)
    pfdr_rho = float(pfdr_rho)
    pfdr_cond_min = float(pfdr_cond_min)
    pfdr_dif_rcd = float(pfdr_dif_rcd)
    pfdr_dif_tol = float(pfdr_dif_tol)
     
    # Convert all int arguments
    cp_it_max = int(cp_it_max)
    pfdr_it_max = int(pfdr_it_max)
    verbose = int(verbose)
    max_num_threads = int(max_num_threads)
    if max_split_size is None:
        max_split_size = V
    else:
        max_split_size = int(max_split_size)

    # Check type of all booleen arguments
    for name, b_args in zip(
            ["Gram_if_square", "balance_parallel_split", "compute_List",
             "compute_Graph", "compute_Obj", "compute_Time", "compute_Dif"],
            [ Gram_if_square ,  balance_parallel_split ,  compute_List ,
              compute_Graph ,  compute_Obj ,  compute_Time ,  compute_Dif ]):
        if type(b_args) != bool:
            raise TypeError("Cut-pursuit d1 quadratic l1 bounds: argument "
                            "'{0}' must be boolean".format(name))
    
    # Call wrapper python in C  
    return cp_d1_ql1b_cpy(Y, A, first_edge, adj_vertices, edge_weights,
        Yl1, l1_weights, low_bnd, upp_bnd, cp_dif_tol, cp_it_max, pfdr_rho,
        pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max, verbose,
        max_num_threads, max_split_size, balance_parallel_split,
        Gram_if_square, real_t == "float64", compute_List, compute_Graph,
        compute_Obj, compute_Time, compute_Dif)
