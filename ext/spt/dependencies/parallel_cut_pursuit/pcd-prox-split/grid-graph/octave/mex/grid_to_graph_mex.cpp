/*=============================================================================
 * [first_edge, adj_vertices, connectivities] = grid_to_graph(shape,
 *      [connectivity = 1], [graph_as_forward_star = true])
 *
 * or
 *
 * [edges, connectivities] = grid_to_graph(shape, [connectivity = 1],
 *      graph_as_forward_star = false)
 * 
 * If graph_as_forward_star is true (the default), then the two first outputs 
 * first_edge, adj_vertices constitutes the forward star representation of the
 * graph. If graph_as_forward_star is set to false, the first output
 * constitutes an edge list representation.
 * 
 *  Hugo Raguet 2019
 *===========================================================================*/
#include <cstdint>
#include <limits>
#include "mex.h"
#include "grid_graph.hpp"

typedef uint8_t conn_t;
# define mxCONN_CLASS mxUINT8_CLASS
# define mxCONN_ID "uint8"

/* template for handling several index types */
template <typename index_t, mxClassID mxINDEX_CLASS>
static void grid_to_graph_mex(int nlhs, mxArray **plhs, int nrhs,
    const mxArray **prhs)
{
    /* retrieve inputs */
    size_t D = mxGetNumberOfElements(prhs[0]);
    index_t* shape = (index_t*) mxGetData(prhs[0]);
    conn_t connectivity = nrhs > 1 ? mxGetScalar(prhs[1]) : 1;

    bool graph_as_forward_star = nrhs > 2 ?
        mxIsLogicalScalarTrue(prhs[2]) : true;

    /* retrieve sizes */
    size_t E = num_edges_grid_graph<index_t, conn_t>(D, shape, connectivity);
    index_t V = 1;
    for (size_t d = 0; d < D; d++){ V *= shape[d]; }

    if (V > std::numeric_limits<index_t>::max()){
        mexErrMsgIdAndTxt("MEX", "Grid to graph mex: the number of "
            "vertices cannot be represented by the given integer type "
            "(%lu vertices requested, no more than %lu allowed)", V,
            (size_t) std::numeric_limits<index_t>::max());
    }
    if (graph_as_forward_star && E > std::numeric_limits<index_t>::max()){
        mexErrMsgIdAndTxt("MEX", "Grid to graph mex: the number of "
            "edges cannot be represented by the given integer type "
            "(%lu edges requested, no more than %lu allowed)", E,
            (size_t) std::numeric_limits<index_t>::max());
    }

    /* compute edges and connectivities */
    index_t* edges = (index_t*) mxMalloc(sizeof(index_t)*2*E);

    conn_t* connectivities = nullptr;
    if (!graph_as_forward_star && nlhs > 1){
        plhs[1] = mxCreateNumericMatrix(1, E, mxCONN_CLASS, mxREAL);
        connectivities = (conn_t*) mxGetData(plhs[1]);
    }else if (graph_as_forward_star && nlhs > 2){
        plhs[2] = mxCreateNumericMatrix(1, E, mxCONN_CLASS, mxREAL);
        connectivities = (conn_t*) mxGetData(plhs[2]);
    }

    edge_list_grid_graph<index_t, conn_t>(D, shape, connectivity, edges,
        connectivities); 

    if (!graph_as_forward_star){
        plhs[0] = mxCreateNumericMatrix(0, 0, mxINDEX_CLASS, mxREAL);
        mxSetM(plhs[0], 2);
        mxSetN(plhs[0], E);
        mxSetData(plhs[0], (void*) edges);
        return;
    }

    /* convert to forward star representation */
    plhs[0] = mxCreateNumericMatrix(1, V + 1, mxINDEX_CLASS, mxREAL);
    index_t* first_edge = (index_t*) mxGetData(plhs[0]);

    index_t* reindex = (index_t*) mxMalloc(sizeof(index_t)*E);

    edge_list_to_forward_star<index_t, index_t>(V, E, edges, first_edge,
        reindex);

    /* override source vertices with permuted end vertices */
    for (size_t e = 0; e < E; e++){
        edges[(size_t) 2*reindex[e]] = edges[2*e + 1];
    }
    /* store resulting adjacent vertices in first half of edges storage */
    index_t* adj_vertices = edges;
    for (size_t e = 0; e < E; e++){ adj_vertices[e] = edges[2*e]; }
    
    /* permute connectivity correspondingly and set to output if requested */
    if (nlhs > 2){ 
        index_t* buf = edges + E; // reuse second half of edges storage
        for (size_t e = 0; e < E; e++){ buf[reindex[e]] = connectivities[e]; }
        for (size_t e = 0; e < E; e++){ connectivities[e] = buf[e]; }
    }

    /* free unused second half of edges and permutation indices */
    mxFree((void*) reindex);
    adj_vertices = (index_t*)
        mxRealloc((void*) adj_vertices, sizeof(index_t)*E);

    plhs[1] = mxCreateNumericMatrix(0, 0, mxINDEX_CLASS, mxREAL);
    mxSetM(plhs[1], 1);
    mxSetN(plhs[1], E);
    mxSetData(plhs[1], (void*) adj_vertices);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    if (mxGetClassID(prhs[0]) == mxINT16_CLASS ||
        mxGetClassID(prhs[0]) == mxUINT16_CLASS){
        grid_to_graph_mex<uint16_t, mxUINT16_CLASS>(nlhs, plhs, nrhs, prhs);
    }else if (mxGetClassID(prhs[0]) == mxINT32_CLASS ||
              mxGetClassID(prhs[0]) == mxUINT32_CLASS){
        grid_to_graph_mex<uint32_t, mxUINT32_CLASS>(nlhs, plhs, nrhs, prhs);
    }else if (mxGetClassID(prhs[0]) == mxINT64_CLASS ||
              mxGetClassID(prhs[0]) == mxUINT64_CLASS){
        grid_to_graph_mex<uint64_t, mxUINT64_CLASS>(nlhs, plhs, nrhs, prhs);
    }else{
        mexErrMsgIdAndTxt("MEX", "Grid to graph: argument 'shape' must be of "
            "an integer class encoded over 16, 32 or 64 bits (%s given).",
            mxGetClassName(prhs[0]));
    }
}
