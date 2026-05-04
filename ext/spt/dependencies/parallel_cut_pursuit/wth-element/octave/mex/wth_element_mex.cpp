/*=============================================================================
 * wth = wth_element(values, wrk = size/2, weights = []);
 *
 * Hugo Raguet 2016, 2018
 *===========================================================================*/
#include <cstdint>
#include "mex.h"
#include "../../include/wth_element.hpp"

/* template for handling both single and double precisions */
template <typename value_t, typename index_t, typename rank_t = index_t,
    typename weight_t = index_t>
static void wth_element_mex(int nlhs, mxArray **plhs, int nrhs,
    const mxArray **prhs)
{
    index_t size = mxGetNumberOfElements(prhs[0]);

    const rank_t wrk = nrhs > 1 ? mxGetScalar(prhs[1]) : size/2;

    plhs[0] = mxCreateNumericMatrix(1, 1, mxGetClassID(prhs[0]), mxREAL);
    value_t* wth = (value_t*) mxGetData(plhs[0]);

    if (nrhs < 3){ // nth element
        mxArray* cpyvalues = mxDuplicateArray(prhs[0]);
        value_t* values = (value_t*) mxGetData(cpyvalues);
        *wth = nth_element(values, size, wrk);
        mxDestroyArray(cpyvalues);
    }else{ // weighted version
        const weight_t* weights = (weight_t*) mxGetData(prhs[2]);
        const value_t* values = (value_t*) mxGetData(prhs[0]);
        index_t* indices = (index_t*) mxMalloc(sizeof(index_t)*size);
        for (index_t i = 0; i < size; i++){ indices[i] = i; }
        *wth = wth_element(indices, values, size, wrk, weights);
        mxFree(indices);
    }
}

template <typename value_t, typename index_t>
static void wth_element_mex_weight_t(int nlhs, mxArray **plhs, int nrhs,
    const mxArray **prhs)
{
    if (nrhs < 3){
        wth_element_mex<value_t, index_t>(nlhs, plhs, nrhs, prhs);
    }else if (mxIsDouble(prhs[2])){
        wth_element_mex<value_t, index_t, double, double>
            (nlhs, plhs, nrhs, prhs);
    }else if (mxIsSingle(prhs[2])){
        wth_element_mex<value_t, index_t, double, float>
            (nlhs, plhs, nrhs, prhs);
    }else if (mxIsUint8(prhs[2])){
        wth_element_mex<value_t, index_t, uint32_t, uint8_t>
            (nlhs, plhs, nrhs, prhs);
    }else if (mxIsUint16(prhs[2])){
        wth_element_mex<value_t, index_t, uint32_t, uint16_t>
            (nlhs, plhs, nrhs, prhs);
    }else if (mxIsUint32(prhs[2])){
        wth_element_mex<value_t, index_t, uint64_t, uint32_t>
            (nlhs, plhs, nrhs, prhs);
    }else if (mxIsUint64(prhs[2])){
        wth_element_mex<value_t, index_t, uint64_t, uint64_t>
            (nlhs, plhs, nrhs, prhs);
    }else{
        mexErrMsgIdAndTxt("MEX", "W-th element: not implemented for input "
            "weights of class %s.", mxGetClassName(prhs[2]));
    }
}

template <typename value_t>
static void wth_element_mex_index_t(int nlhs, mxArray **plhs, int nrhs,
    const mxArray **prhs)
{
    size_t size = mxGetNumberOfElements(prhs[0]);
    if (size <= ((uint8_t) -1)){
        wth_element_mex_weight_t<value_t, uint8_t>(nlhs, plhs, nrhs, prhs);
    }else if (size <= ((uint16_t) -1)){
        wth_element_mex_weight_t<value_t, uint16_t>(nlhs, plhs, nrhs, prhs);
    }else if (size <= ((uint32_t) -1)){
        wth_element_mex_weight_t<value_t, uint32_t>(nlhs, plhs, nrhs, prhs);
    }else if (size <= ((uint64_t) -1)){
        wth_element_mex_weight_t<value_t, uint64_t>(nlhs, plhs, nrhs, prhs);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    if (mxIsDouble(prhs[0])){
        wth_element_mex_index_t<double>(nlhs, plhs, nrhs, prhs);
    }else if (mxIsSingle(prhs[0])){
        wth_element_mex_index_t<float>(nlhs, plhs, nrhs, prhs);
    }else if (mxIsUint8(prhs[0])){
        wth_element_mex_index_t<uint8_t>(nlhs, plhs, nrhs, prhs);
    }else if (mxIsUint16(prhs[0])){
        wth_element_mex_index_t<uint16_t>(nlhs, plhs, nrhs, prhs);
    }else if (mxIsUint32(prhs[0])){
        wth_element_mex_index_t<uint32_t>(nlhs, plhs, nrhs, prhs);
    }else if (mxIsUint64(prhs[0])){
        wth_element_mex_index_t<uint64_t>(nlhs, plhs, nrhs, prhs);
    }else if (mxIsInt8(prhs[0])){
        wth_element_mex_index_t<int8_t>(nlhs, plhs, nrhs, prhs);
    }else if (mxIsInt16(prhs[0])){
        wth_element_mex_index_t<int16_t>(nlhs, plhs, nrhs, prhs);
    }else if (mxIsInt32(prhs[0])){
        wth_element_mex_index_t<int32_t>(nlhs, plhs, nrhs, prhs);
    }else if (mxIsInt64(prhs[0])){
        wth_element_mex_index_t<int64_t>(nlhs, plhs, nrhs, prhs);
    }else{
        mexErrMsgIdAndTxt("MEX", "W-th element: not implemented for input "
            "values of class %s.", mxGetClassName(prhs[0]));
    }
}
