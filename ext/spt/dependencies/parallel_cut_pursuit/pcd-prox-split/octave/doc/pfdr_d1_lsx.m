function [X, Obj, Dif] = pfdr_d1_lsx(loss, Y, edges, options)
%
%        [X, Obj, Dif] = pfdr_d1_lsx(loss, Y, edges, [options])
%
% Minimize functional over a graph G = (V, E)
%
%         F(x) = f(x) + ||x||_d1 + i_{simplex}(x)
%
%  where for each vertex, x_v is a D-dimensional vector,
%        f is a separable data-fidelity loss
%        ||x||_d1 = sum_{uv in E} w_d1_uv (sum_d w_d1_d |x_ud - x_vd|),
%  and i_{simplex} is the standard D-simplex constraint over each vertex,
%      i_{simplex} = 0 for all v, (for all d, x_vd >= 0) and sum_d x_vd = 1,
%                  = infinity otherwise;
%
% using preconditioned forward-Douglas-Rachford splitting algorithm.
%
% Available separable data-fidelity loss include:
%
% linear
%       f(x) = - <x, y>_w ,
%   with  <x, y>_w = sum_{v,d} w_d x_{v,d} y_{v,d} ;
%
% quadratic
%       f(x) = 1/2 ||y - x||_{l2,w}^2 ,
%   with  ||y - x||_{l2,w}^2 = sum_{v,d} w_v (y_{v,d} - x_{v,d})^2 ;
%
% smoothed Kullback-Leibler divergence (cross-entropy)
%       f(x) = sum_v w_v KLs(x_v, y_v),
%   with KLs(y_v, x_v) = KL(s u + (1 - s) y_v ,  s u + (1 - s) x_v), where
%     KL is the regular Kullback-Leibler divergence,
%     u is the uniform discrete distribution over {1,...,D}, and
%     s = loss is the smoothing parameter ;
%   it yields
%
%     KLs(y_v, x_v) = H(s u + (1 - s) y_v)
%         - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) x_{v,d}) ,
%
%   where H is the entropy, that is H(s u + (1 - s) y_v)
%       = - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) y_{v,d}) ;
%   note that the choosen order of the arguments in the Kullback--Leibler
%   does not penalize the entropy of x_v (H(s u + (1 - s) y_v) is a constant),
%   hence this loss is actually equivalent to cross-entropy.
%
% INPUTS: real numeric type is either single or double, not both;
%         indices are start at 0, type uint32
%
% loss - 0 for linear, 1 for quadratic, 0 < loss < 1 for smoothed
%     Kullback-Leibler (see above)
% Y    - observations, (real) D-by-V array, column-major format;
%     the value at each vertex is supposed to lie on the probability simplex
% edges - list of edges, (uint32) array of length 2E;
%     vertices are numeroted (start at 0) in the order they are given in Y;
%     edge number e connects vertices given by edges(2*e - 1) and edges(2*e);
%     every vertex should belong to at least one edge with a nonzero
%     penalization coefficient. If it is not the case, a workaround is to add
%     an edge from the isolated vertex to itself with a small nonzero weight
% options - structure with any of the following fields [with default values]:
%     edge_weights [1.0], loss_weights [none], d11_metric [none],
%     rho [1.5], cond_min [1e-2], dif_rcd [0.0], dif_tol [1e-2], dif_it [16],
%     it_max [1e3], verbose [1e1]
% edge_weights - weights on the edges (w_d1_uv in the above notations);
%     (real) array of length E, or scalar for homogeneous weights
% loss_weights - weights on vertices (w_v in the above notations);
%     (real) array of length V
% d11_metric - diagonal metric on the d11 penalization (w_d1_d above);
%     (real) array of length D; all weights must be strictly positive, and it
%     is advised to normalize the weights so that the first value is unity
% rho - relaxation parameter, 0 < rho < 2;
%     1 is a conservative value; 1.5 often speeds up convergence
% cond_min - stability of preconditioning; 0 < cond_min < 1;
%     corresponds roughly to the minimum ratio to the maximum descent metric;
%     1e-2 is a typical value; a smaller value might enhance preconditioning
% dif_rcd - reconditioning criterion on iterate evolution;
%     a reconditioning is performed if relative changes of the iterate drops
%     below dif_rcd;
%     10*dif_tol is a typical value, 1e2*dif_tol or 1e3*dif_tol might speed up
%     convergence; WARNING: reconditioning might temporarily draw minimizer
%     away from solution, so it is advised to monitor objective value when
%     using reconditioning
% dif_tol - stopping criterion on iterate evolution; algorithm stops if
%     relative changes (in l1 norm) is less than dif_tol;
%     1e-3 is a typical value; a lower one can give better precision but with
%     longer computational time
% dif_it - number of iterations between iterates for evolution measure
% it_max - maximum number of iterations;
%     usually depends on the size of the problems in relation to the available
%     computational budget
% verbose  - if nonzero, display information on the progress, every 'verbose'
%     iterations
%
% OUTPUTS:
%
% X - final minimizer, (real) array of length V
% Obj - the values of the objective functional along iterations;
%   array of length number of iterations + 1
% Dif - if requested, the iterate evolution along iterations;
%   array of length number of iterations
% 
% Parallel implementation with OpenMP API.
%
% H. Raguet and L. Landrieu, Preconditioning of a Generalized Forward-Backward
% Splitting and Application to Optimization on Graphs, SIAM Journal on Imaging
% Sciences, 2015, 8, 2706-2739
%
% H. Raguet, A Note on the Forward-Douglas-Rachford Splitting for Monotone 
% Inclusion and Convex Optimization, Optimization Letters, 2018, 1-24
%
% Hugo Raguet 2016, 2018
