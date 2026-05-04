function wth = wth_element(values, wrk, weights)
%
%        wth = wth_element(values, wrk = length(values)/2, weights = [])
%
% W-th element: weighted n-th element
%
% The weighted rank of an index i is the interval
%
%      [wsum(i), wsum(i) + weight(i)[
%
% where wsum(i) is the cumulative sum of the weights of all indices comparing
% lower to i, and weight(i) is the weight associated to it.
% The w-th element is the index whose weighted rank contains the target wrk;
% note that if all weights are equal to unity, the w-th element with target n
% reduces to the n-th element (starting count at 0).
%
% Use quickselect algorithm
%
% INPUTS:
% values - array of values according to which the ordering is made
% wrk - the target weighted rank
% weights - weights associated
%
% OUTPUTS:
% wth - the element in 'values' whose weighted rank is wrk
%
% Hugo Raguet 2018
