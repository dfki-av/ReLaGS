function varargout = cp_prox_tv_mex(Y, first_edge, adj_vertices, options)
%
%       [Comp, rX, [List, Graph, Obj, Time, Dif]] = cp_prox_tv_mex(Y,
%           first_edge, adj_vertices, [options])
%
% Compute the proximal operator of the graph total variation penalization:
%
%  minimize functional F defined over a graph G = (V, E)
% 
%  F: R^{D-by-V} -> R
%         x      -> 1/2 ||y - x||_Ms^2 + ||x||_d1p
% 
%  where y in R^{D-by-V}, Ms is a diagonal metric in R^{VxD-by-VxD} so that
% 
%       ||y - x||_Ms^2 = sum_{v in V} sum_d ms_v_d (y_v - x_v)^2 ,
% 
%  and
%  
%       ||x||_d1p = sum_{uv in E} w_uv ||x_u - x_v||_{Md,p} ,
% 
%  where Md is a diagonal metric and p can be 1 or 2,
%   ||x_v||_{M,1} = sum_d md_d |x_v|  (weighted l1-norm) or
%   ||x_v||_{M,2} = sqrt(sum_d md_d x_v^2)  (weighted l2-norm)
%
% using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
% splitting algorithm.
%
% NOTA: by default, components are identified using uint16 identifiers; this
% can be easily changed in the mex source if more than 65535 components are
% expected (recompilation is necessary)
% 
% INPUTS: real numeric type is either single or double, not both;
%         indices start at 0, type uint32
%
% Y - observations, (real) D-by-V array
% first_edge, adj_vertices - forward-star graph representation:
%     vertices are numeroted (start at 0) in the order they are given in Y or A
%         (careful to the internal memory representation of multidimensional
%          arrays, usually Octave and Matlab use column-major format)
%     edges are numeroted (start at 0) so that all vertices originating
%         from a same vertex are consecutive;
%     for each vertex, first_edge indicates the first edge starting from the
%         vertex (or, if there are none, starting from the next vertex);
%         (uint32) array of length V + 1, the first value is always zero and
%         the last value is always the total number of edges E;
%     for each edge, adj_vertices indicates its ending vertex, (uint32) array
%         of length E
% options - structure with any of the following fields [with default values]:
%     l22_metric [none], edge_weights [1.0], d1p [2], d1p_metric [none],
%     cp_dif_tol [1e-4], cp_it_max [10], K [2], split_iter_num [1],
%     split_damp_ratio [1.0], split_values_init_num [2],
%     split_values_iter_num [2], pfdr_rho [1.0], pfdr_cond_min [1e-2],
%     pfdr_dif_rcd [0.0], pfdr_dif_tol [1e-2*cp_dif_tol], pfdr_it_max [1e4],
%     verbose [1e3], max_num_threads [none], max_split_size [none],
%     balance_parallel_split [true], compute_List [false], compute_Obj [false],
%     compute_Time [false], compute_Dif [false]
% l22_metric - diagonal metric on squared l2-norm (Ms in above notations);
%     array or length V for weights depending only on vertices, D-by-V array
%     otherwise; all weights must be strictly positive
% edge_weights - weights on the edges (w_uv above);
%     (real) array of length E, or scalar for homogeneous weights
% d1p - define the total variation as the l11- (d1p = 1) or l12- (d1p = 2) norm
%     of the finite differences
% d1p_metric - diagonal metric on d1p penalisation (Md in above notations);
%     all weights must be strictly positive, and it is advised to
%     normalize the weights so that the first value is unity for computation
%     stability
% cp_dif_tol - stopping criterion on iterate evolution; algorithm stops if
%     relative changes (in Euclidean norm) is less than dif_tol;
%     1e-4 is a typical value; a lower one can give better precision but with
%     longer computational time and more final components
% cp_it_max - maximum number of iterations (graph cut and subproblem);
%     10 cuts solve accurately most problems
% K - number of alternative descent directions considered in the split step
% split_iter_num - number of partition-and-update iterations in the split step
% split_damp_ratio - edge weights damping for favoring splitting; edge
%     weights increase in linear progression along partition-and-update
%     iterations, from this ratio up to original value; real scalar between 0
%     and 1, the latter meaning no damping
% split_values_init_num - number of random initializations when looking for
%     descent directions in the split step
% split_values_iter_num - number of refining iterations when looking for
%     descent directions in the split step
% pfdr_rho - relaxation parameter, 0 < rho < 2;
%     1 is a conservative value; 1.5 often speeds up convergence
% pfdr_cond_min - stability of preconditioning; 0 < cond_min < 1;
%     corresponds roughly the minimum ratio to the maximum descent metric;
%     1e-2 is typical, a smaller value might enhance preconditioning
% pfdr_dif_rcd - reconditioning criterion on iterate evolution;
%     a reconditioning is performed if relative changes of the iterate drops
%     below dif_rcd; WARNING: reconditioning might temporarily draw minimizer
%     away from the solution, and give bad subproblem solutions
% pfdr_dif_tol - stopping criterion on iterate evolution; algorithm stops if
%     relative changes (in Euclidean norm) is less than dif_tol;
%     1e-2*cp_dif_tol is a conservative value
% pfdr_it_max - maximum number of iterations;
%     1e4 iterations provides enough precision for most subproblems
% verbose - if nonzero, display information on the progress, every 'verbose'
%     PFDR iterations
% max_num_threads - if greater than zero, set the maximum number of threads
%     used for parallelization with OpenMP
% max_split_size - maximum number of vertices allowed in connected component
%     passed to a split problem; make split of very large components faster,
%     but might induced suboptimal artificial cuts
% balance_parallel_split - if true, the parallel workload of the split step 
%     is balanced; WARNING: this might trade off speed against optimality
% compute_List  - report the list of vertices constituting each component
% compute_Graph - get the reduced graph on the components
% compute_Obj   - compute the objective functional along iterations 
% compute_Time  - monitor elapsing time along iterations
% compute_Dif   - compute relative evolution along iterations 
%
% OUTPUTS: indices start at 0
%
% Comp - assignement of each vertex to a component, (uint16) array of length V
% rX - values of eachcomponents of the minimizer, (real) array of length rV;
%     the actual minimizer can be reconstructed with X = rX(Comp + 1);
% List - if requested, list of vertices constituting each component; cell array
%     of length rV, containing (uint32) arrays of indices
% Graph - if requested, reduced graph structure; cell array of length 3
%     representing the graph as forward-star (see input first_edge and
%     adj_vertices) together with edge weights
% Obj - the values of the objective functional along iterations; array of
%     length number of iterations performed + 1;
% Time - if requested, the elapsed time along iterations;
%     array of length number of iterations performed + 1
% Dif  - if requested, the iterate evolution along iterations;
%     array of length number of iterations performed
%
% TODO: implement subgradients retrieval
% Gtv - subgradients of the total variation penalization at solution; (real)
%     array of length E; if e is the edge (u, v), the subgradient of the
%     total variation penalization at vertices (u, v) is (-Gd1(e), Gd1(e))
% 
% Parallel implementation with OpenMP API.
%
% L. Landrieu and G. Obozinski, Cut Pursuit: Fast Algorithms to Learn
% Piecewise Constant Functions on General Weighted Graphs, SIIMS, 10, 4,
% 1724–1766, 2017.
%
% Hugo Raguet 2023
