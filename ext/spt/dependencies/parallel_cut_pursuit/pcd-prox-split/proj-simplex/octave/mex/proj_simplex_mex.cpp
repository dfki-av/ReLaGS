/*=============================================================================
 *  Y = proj_simplex_mex(X, A = [], M = []);
 *
 *  Hugo Raguet 2016, 2018
 *===========================================================================*/
#include "mex.h"
#include "proj_simplex.hpp"

/* functions for checking arguments type */
static void check_args(int nrhs, const mxArray *prhs[], const int* args,
    int n, mxClassID id, const char* id_name)
{
    for (int i = 0; i < n; i++){
        if (nrhs > args[i] && mxGetClassID(prhs[args[i]]) != id
            && !mxIsEmpty(prhs[args[i]])){
            mexErrMsgIdAndTxt("MEX", "projection on simplex: argument %d is of"
                " class  %s, but class %s is expected.", args[i] + 1,
                mxGetClassName(prhs[args[i]]), id_name);
        }
    }
}

/* arrays with arguments type */
static const int args_real_t[] = {0, 1, 2};
static const int n_real_t = 3;

/* template for handling both single and double precisions */
template<typename real_t>
static void proj_simplex_mex(int nlhs, mxArray **plhs, int nrhs,
    const mxArray **prhs)
{
    proj_simplex::index_t D = mxGetM(prhs[0]);
    proj_simplex::index_t N = mxGetN(prhs[0]);

    const real_t *A = (nrhs > 1 && mxGetNumberOfElements(prhs[1]) > 1) ?
        (real_t*) mxGetData(prhs[1]) : nullptr;
    real_t a = (nrhs > 1 && mxGetNumberOfElements(prhs[1]) == 1) ?
        mxGetScalar(prhs[1]) : 1.;

    const real_t *M = (nrhs > 2 && mxGetNumberOfElements(prhs[2]) > D) ?
        (real_t*) mxGetData(prhs[2]) : nullptr;
    const real_t *m = (nrhs > 2 && mxGetNumberOfElements(prhs[2]) == D) ?
        (real_t*) mxGetData(prhs[2]) : nullptr;

    if (A && mxGetNumberOfElements(prhs[1]) != N){
        mexErrMsgIdAndTxt("MEX", "The simplex sum argument A has %d elements "
            "but should have %d elements like the the number of columns of "
            "the input vectors argument X", mxGetNumberOfElements(prhs[1]), N);
    }
    if (M && mxGetM(prhs[2]) != D){
        mexErrMsgIdAndTxt("MEX", "The metric argument M has %d rows "
            "but should have %d rows like the input vectors argument X",
            mxGetM(prhs[2]), D);
    }
    if (M && mxGetN(prhs[2]) != N){
        mexErrMsgIdAndTxt("MEX", "The metric argument M has %d columns "
            "but should have %d columns like the input vectors argument X",
            mxGetN(prhs[2]), N);
    }

    plhs[0] = mxDuplicateArray(prhs[0]);
    real_t *X = (real_t*) mxGetData(plhs[0]);
    proj_simplex::proj_simplex<real_t>(X, D, N, A, a, M, m);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    if (mxIsDouble(prhs[0])){
        check_args(nrhs, prhs, args_real_t, n_real_t, mxDOUBLE_CLASS,
            "double");
        proj_simplex_mex<double>(nlhs, plhs, nrhs, prhs);
    }else{
        check_args(nrhs, prhs, args_real_t, n_real_t, mxSINGLE_CLASS,
            "single");
        proj_simplex_mex<float>(nlhs, plhs, nrhs, prhs);
    }
}
