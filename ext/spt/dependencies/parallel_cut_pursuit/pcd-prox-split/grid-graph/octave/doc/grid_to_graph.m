function [first_edge, adj_vertices, connectivities] = grid_to_graph( ...
            shape, connectivity, graph_as_forward_star)
%
%        [first_edge, adj_vertices, [connectivities]] = grid_to_graph
%           (shape, [connectivity = 1], [graph_as_forward_star = true])
% or
%
%        [edges, [connectivities]] = grid_to_graph_mex(shape, ...
%           connectivity, graph_as_forward_star = false)
% 
% Compute the graph structure defined by local connectivity on a
% multidimensional grid.
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
% is either uint16, uint32 or uint64, depending on the type of 'shape' input.
% 
% INPUTS:
%  shape - integer array of length D, giving the grid size in each dimension
%  connectivity - defines the neighboring relationship;
%        corresponds to the _square_ of the maximum Euclidean distance
%        between two neighbors;
%        if less than 4, it defines the number of coordinates allowed
%        to simultaneously vary (+1 or -1) to define a neighbor; in that
%        case, each level l of connectivity in dimension D adds
%        binom(D, l)*2^l neighbors;
%        corresponding number of neighbors for D = 2 and 3:
% 
%        connectivity |  1   2   3 
%        --------------------------
%                  2D |  4   8  (8)
%                  3D |  6  18  26 
% 
%        note that a connectivity of 4 or more includes neighbors whose
%        coordinates might differ by 2 or more from the coordinates of the
%        considered vertex. Interestingly, in dimension 4 or more, including
%        all surrounding vertices would then also include vertices from a
%        "more distant" surround: the neighbor v + (2, 0, 0, 0) is at the
%        same distance as the neighbor v + (1, 1, 1, 1).
%  graph_as_forward_star - representation of the output graph; if set to true
%        (the default), forward-star as described above in 2 output arrays;
%        otherwise, edge list in 1 output array
% 
% OUTPUTS:
%       Vertices of the grid are indexed in _column-major_ order, that is
%       indices increase first along the first dimension specified in 'shape'
%       array (in the usual convention in 2D, this corresponds to columns), and
%       then along the second dimension, and so on up to the last dimension;
%       indexing in _row-major_ order (indices increase first along the last
%       dimension and so on up to the first) can be obtained by simply
%       reverting the order of the grid dimensions in 'shape' array (in 2D,
%       this amounts to transposition)
%  first_edge, adj_vertices - if graph_as_forward_star is true, forward-star
%        representation of the resulting graph, as described above
%  edges - if graph_as_forward_star is false, edge list representation of the
%        resulting graph, integers array of size 2-by-E
%  connectivities - if requested, the connectivities (square Euclidean
%        lengths) of the edges, array of uint8 of length E
% 
% Hugo Raguet 2019, 2020
