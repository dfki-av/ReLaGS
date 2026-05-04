/*=============================================================================
 * [first_edge, adj_vertices, reindex] = edge_list_to_forward_star(V, edges)
 * 
 *  Hugo Raguet 2019
 *===========================================================================*/
#include <cstdint>
#include <limits>
#include "mex.h"
#include "grid_graph.hpp"

/* template for handling several index types */
template <typename index_t, mxClassID mxINDEX_CLASS>
static void edge_list_to_forward_star_mex(int nlhs, mxArray **plhs, int nrhs,
    const mxArray **prhs)
{
    size_t V = mxGetScalar(prhs[0]);
    size_t E = mxGetNumberOfElements(prhs[1])/2;
    const index_t* edges = (index_t*) mxGetData(prhs[1]);

    if (V > std::numeric_limits<index_t>::max()){
        mexErrMsgIdAndTxt("MEX", "Edge list to forward star mex: the number "
            "of vertices 'V' cannot be represented by the given integer type "
            "(%lu provided, no more than %lu allowed)", V,
            (size_t) std::numeric_limits<index_t>::max());
    }
    if (E > std::numeric_limits<index_t>::max()){
        mexErrMsgIdAndTxt("MEX", "Edge list to forward star mex: the number "
            "of edges cannot be represented by the given integer type "
            "(%lu provided, no more than %lu allowed)", E,
            (size_t) std::numeric_limits<index_t>::max());
    }


    plhs[0] = mxCreateNumericMatrix(1, V + 1, mxINDEX_CLASS, mxREAL);
    index_t* first_edge = (index_t*) mxGetData(plhs[0]);

    index_t* reindex = (index_t*) mxMalloc(sizeof(index_t)*E);

    edge_list_to_forward_star<index_t, index_t>(V, E, edges, first_edge,
        reindex);

    plhs[1] = mxCreateNumericMatrix(1, E, mxINDEX_CLASS, mxREAL);
    index_t* adj_vertices = (index_t*) mxGetData(plhs[1]);
    for (size_t e = 0; e < E; e++){
        adj_vertices[reindex[e]] = edges[2*e + 1];
    }

    if (nlhs > 2){
        plhs[2] = mxCreateNumericMatrix(0, 0, mxINDEX_CLASS, mxREAL);
        mxSetM(plhs[2], 1);
        mxSetN(plhs[2], E);
        mxSetData(plhs[2], (void*) reindex);
    }else{
        mxFree((void*) edges);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    if (mxGetClassID(prhs[1]) == mxINT16_CLASS ||
        mxGetClassID(prhs[1]) == mxUINT16_CLASS){
        edge_list_to_forward_star_mex<uint16_t, mxUINT16_CLASS>(nlhs, plhs,
            nrhs, prhs);
    }else if (mxGetClassID(prhs[1]) == mxINT32_CLASS ||
              mxGetClassID(prhs[1]) == mxUINT32_CLASS){
        edge_list_to_forward_star_mex<uint32_t, mxUINT32_CLASS>(nlhs, plhs,
            nrhs, prhs);
    }else if (mxGetClassID(prhs[1]) == mxINT64_CLASS ||
              mxGetClassID(prhs[1]) == mxUINT64_CLASS){
        edge_list_to_forward_star_mex<uint64_t, mxUINT64_CLASS>(nlhs, plhs,
            nrhs, prhs);
    }else{
        mexErrMsgIdAndTxt("MEX", "Edge list to forward star: argument 'edges' "
            "must be of an integer class encoded over 16, 32 or 64 bits "
            "(%s given).", mxGetClassName(prhs[1]));
    }
}
