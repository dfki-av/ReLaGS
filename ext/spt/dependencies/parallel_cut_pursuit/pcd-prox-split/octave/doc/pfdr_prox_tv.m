function [X, Obj, Dif] = pfdr_prox_tv(Y, edges, options)
%
%        [X, Obj, Dif] = pfdr_prox_tv(Y, edges, [options])
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
% using preconditioned forward-Douglas-Rachford splitting algorithm.
%
% INPUTS: real numeric type is either single or double, not both;
%         indices start at 0, type uint32
%
% Y - observations, (real) D-by-V array
% edges - list of edges, (uint32) array of length 2E;
%     vertices are numeroted (start at 0) in the order they are given in Y;
%     edge number e connects vertices given by edges(2*e - 1) and edges(2*e);
%     every vertex should belong to at least one edge with a nonzero
%     penalization coefficient. If it is not the case, a workaround is to add
%     an edge from the isolated vertex to itself with a small nonzero weight
% options - structure with any of the following fields [with default values]:
%     l22_metric [none], edge_weights [1.0], d1p [2], d1p_metric [none],
%     rho [1.5], cond_min [1e-2], dif_rcd [0.0], dif_tol [1e-4],
%     dif_it [32], it_max [1e3], verbose [1e2]
% l22_metric - diagonal metric on squared l2-norm (Ms in above notations);
%     array or length V for weights depending only on vertices,
%     D-by-V array otherwise; weights must be strictly positive
% edge_weights - weights on the edges (w_uv above);
%     (real) array of length E or scalar for homogeneous weights
% d1p - define the total variation as the l11- (d1p = 1) or l12- (d1p = 2) norm
%     of the finite differences
% d1p_metric - diagonal metric on d1p penalisation (Md in above notations);
%     all weights must be strictly positive, and it is advised to normalize
%     the weights so that the first value is unity for computation stability
% rho - relaxation parameter, 0 < rho < 2;
%     1 is a conservative value; 1.5 often speeds up convergence
% cond_min - stability of preconditioning; 0 < cond_min < 1;
%     corresponds roughly to the minimum ratio between different directions of
%     the descent metric; 1e-2 is a typical value;
%     smaller values might enhance preconditioning but might also make it
%     unstable; increase this value if iteration steps seem to get too small
% dif_rcd - reconditioning criterion on iterate evolution;
%     a reconditioning is performed if relative changes of the iterate drops
%     below dif_rcd; it is then divided by 10;
%     10*dif_tol is a typical value, 1e2*dif_tol or 1e3*dif_tol might speed up
%     convergence;
%     WARNING: reconditioning might temporarily draw minimizer away from
%     solution, it is advised to monitor objective value when using
%     reconditioning
% dif_tol - stopping criterion on iterate evolution; algorithm stops if
%     relative changes (in Euclidean norm) is less than dif_tol;
%     1e-5 is a typical value; a lower one can give better precision but with
%     longer computational time
% dif_it - number of iterations between iterates for evolution measure
% it_max - maximum number of iterations;
%     usually depends on the size of the problems in relation to the available
%     computational budget
% verbose - if nonzero, display information on the progress, every 'verbose'
%     iterations
%
% OUTPUTS:
%
% X - final minimizer, array of length V (real)
% Obj - the values of the objective functional along iterations;
%     array of length number of iterations + 1;
% Dif  - if requested, the iterate evolution along iterations;
%     array of length number of iterations
% 
% Parallel implementation with OpenMP API.
%
% H. Raguet and L. Landrieu, Preconditioning of a Generalized Forward-Backward
% Splitting and Application to Optimization on Graphs, SIAM Journal on Imaging
% Sciences, 2015, 8, 2706-2739
%
% Hugo Raguet 2023
