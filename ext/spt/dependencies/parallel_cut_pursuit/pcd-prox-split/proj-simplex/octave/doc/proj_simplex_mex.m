function X = proj_simplex_mex(Y, A, M)
%
%        X = proj_simplex_metric_mex(Y, A = [], M = [])
% 
% Orthogonal projection over the simplex:
%
%      for all d, x_d >= 0; and sum_d x_d = a
%      
% possibly within a diagonal metric defined by 1./m as
%
%      <x, y>_{1/m} = <x, diag(1./m) y> = sum_d x_d y_d / m_d
%
% i.e. m is the vector of the _inverses_ of the diagonal entries of the 
% matrix of the desired metric.
%
% Based on Condat's modification (simple version) of Michelot (1986)'s algorithm
% Work possibly on different vectors in parallel with OpenMP API
%
% INPUTS: (warning: real numeric type is either single or double, not both)
% Y - array of N D-dimensionnal vectors, D-by-N array (real)
% A - total sums of the simplices (unity yields the standard simplex)
%     can be different for each input vector (array of length N; real)
%     or common to all (scalar; real)
% M - (inverse terms of) a diagonal metric
%     can be different for each input verctor (D-by-N array; real)
%     or common to all (array of length D; real)
%
% OUTPUTS:
% X - the results of the projection, D-by-N array (real)
%
% Reference:
% L. Condat, Fast Projection onto the Simplex and the l1 Ball, Mathematical
% Programming, 2016, 158, 575-585
% 
% Hugo Raguet 2016, 2018
