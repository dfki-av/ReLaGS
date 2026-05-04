function [first_edge, adj_vertices, reindex] = ...
                    edge_list_to_forward_star(V, edges)
%
%        [first_edge, adj_vertices, reindex] =
%                   edge_list_to_forward_star(V, edges)
% 
% Convert the graph representation given by the edge list 'edges' into a
% forward-star representation 'first_edge', 'adj_vertices'.
% 
% A graph with V vertices and E edges is represented either as edge list
% (array of E edges given as ordered pair of vertices), or as forward star,
% where edges are numeroted so that all edges originating from a same vertex
% are consecutive, and represented by the following parameters:
% first_edge - array of length V + 1, indicating for each vertex, the first
%      edge starting from the vertex (or, if there are none, starting from
%      the next vertex); the first value is always zero and the last value is
%      always the total number of edges
% adj_vertices - array of length E, indicating for each edge, its ending
%      vertex
% 
% NOTA: edges and vertices identifiers are integers (start at 0), whose type
% is either uint16, uint32 or uint64, depending on the type of 'edges' input.
% 
% INPUTS:
%  V - the number of vertices in the graph (usually max(edge(:)) + 1)
%  edges - list of edges, 2-by-E array of integers, each edges being given by
%          two consecutive vertex identifiers
% 
% OUTPUTS:
%  first_edge, adj_vertices - see above.
%  reindex - for all edges originating from a same vertex being consecutive,
%            they must be reordered; reindex keep track of this permutation
%            so that edge number 'e' in the original list becomes edge number
%            reindex[e] in the forward-star structure; in practice, in order to
%            have the same order in both structures, permute the original list
%            with `edges(:, reindex + 1) = edges`.
% 
% Hugo Raguet 2020
