/*=============================================================================
 * [X, Obj, Dif] = pfdr_d1_lsx(loss, Y, edges, [options])
 *
 * options is a struct with any of the following fields [with default values]:
 *
 *      edge_weights [1.0], loss_weights [none], d11_metric [none],
 *      rho [1.5], cond_min [1e-2], dif_rcd [0.0], dif_tol [1e-2],
 *      dif_it [16], it_max [1e3], verbose [1e1]
 * 
 *  Hugo Raguet 2016, 2018
 *===========================================================================*/
#include <cstring>
#include <cstdint>
#include "mex.h"
#include "pfdr_d1_lsx.hpp"

using namespace std;

/* vertex_t is an integer type able to represent the number of vertices */
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned iterator in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
    typedef int32_t vertex_t;
    # define mxVERTEX_CLASS mxINT32_CLASS
    # define VERTEX_T_STRING "int32"
#else
    typedef uint32_t vertex_t;
    # define mxVERTEX_CLASS mxUINT32_CLASS
    # define VERTEX_T_STRING "uint32"
#endif


/* function for checking optional parameters */
static void check_opts(const mxArray* options)
{
    if (!options){ return; }

    if (!mxIsStruct(options)){
        mexErrMsgIdAndTxt("MEX", "PFDR graph d1 loss simplex: "
            "fourth parameter 'options' should be a structure, (%s given).",
            mxGetClassName(options));
    }

    const int num_allow_opts = 10;
    const char* opts_names[] = {"edge_weights", "loss_weights",
        "d11_metric", "rho", "cond_min", "dif_rcd", "dif_tol", "dif_it",
        "it_max", "verbose"};

    const int num_given_opts = mxGetNumberOfFields(options);

    for (int given_opt = 0; given_opt < num_given_opts; given_opt++){
        const char* opt_name = mxGetFieldNameByNumber(options, given_opt);
        int allow_opt;
        for (allow_opt = 0; allow_opt < num_allow_opts; allow_opt++){
            if (strcmp(opt_name, opts_names[allow_opt]) == 0){ break; }
        }
        if (allow_opt == num_allow_opts){
            mexErrMsgIdAndTxt("MEX", "PFDR graph d1 loss simplex: "
                "option '%s' unknown.", opt_name);
        }
    }
}

/* function for checking parameter type */
static void check_arg_class(const mxArray* arg, const char* arg_name,
    mxClassID class_id, const char* class_name)
{
    if (mxGetNumberOfElements(arg) > 1 && mxGetClassID(arg) != class_id){
        mexErrMsgIdAndTxt("MEX", "PFDR graph d1 loss simplex: "
            "parameter '%s' should be of class %s (%s given).",
            arg_name, class_name, mxGetClassName(arg), class_name);
    }
}

/* resize memory buffer allocated by mxMalloc and create a row vector */
template <typename type_t>
static mxArray* resize_and_create_mxRow(type_t* buffer, size_t size,
    mxClassID id)
{
    mxArray* row = mxCreateNumericMatrix(0, 0, id, mxREAL);
    if (size){
        mxSetM(row, 1);
        mxSetN(row, size);
        buffer = (type_t*) mxRealloc((void*) buffer, sizeof(type_t)*size);
        mxSetData(row, (void*) buffer);
    }else{
        mxFree((void*) buffer);
    }
    return row;
}

/* template for handling both single and double precisions */
template<typename real_t, mxClassID mxREAL_CLASS>
static void pfdr_d1_lsx_mex(int nlhs, mxArray **plhs, int nrhs,
    const mxArray **prhs)
{
    /***  get inputs  ***/

    const char* real_class_name = mxREAL_CLASS == mxDOUBLE_CLASS ?
        "double" : "single";

    /**  sizes and loss  **/

    real_t loss = mxGetScalar(prhs[0]);
    size_t D = mxGetM(prhs[1]);
    vertex_t V = mxGetN(prhs[1]);
    const real_t *Y = (real_t*) mxGetData(prhs[1]);

    /**  graph structure  **/

    check_arg_class(prhs[2], "edges", mxVERTEX_CLASS, VERTEX_T_STRING);

    size_t E = mxGetNumberOfElements(prhs[2])/2;
    if (mxGetN(prhs[2]) == 2 && E > 2){
        mexErrMsgIdAndTxt("MEX", "PFDR graph d1 quadratic l1 bounds: edges "
            "ends must lie consecutively in memory; since GNU Octave and "
            "Matlab use column-major layout, parameter 'edges' should have "
            "size 2-by-%i (%i-by-2 given)", E, E);
    }
    const vertex_t* edges = (vertex_t*) mxGetData(prhs[2]);

    /**  optional parameters  **/

    const mxArray* options = nrhs > 3 ? prhs[3] : nullptr;

    check_opts(options);
    const mxArray* opt;


    /* penalizations */
    #define GET_REAL_OPT(NAME) \
        const real_t* NAME = nullptr; \
        if (opt = mxGetField(options, 0, #NAME)){ \
            check_arg_class(opt, #NAME, mxREAL_CLASS, real_class_name); \
            NAME = (real_t*) mxGetData(opt); \
        }

    GET_REAL_OPT(loss_weights)
    GET_REAL_OPT(d11_metric)
    GET_REAL_OPT(edge_weights)
    real_t homo_edge_weight = 1.0;
    if (opt && mxGetNumberOfElements(opt) == 1){
        edge_weights = nullptr;
        homo_edge_weight = mxGetScalar(opt);
    }

    /* algorithmic parameters */
    #define GET_SCAL_OPT(NAME, DFLT) \
        NAME = (opt = mxGetField(options, 0, #NAME)) ? mxGetScalar(opt) : DFLT;

    real_t GET_SCAL_OPT(rho, 1.5);
    real_t GET_SCAL_OPT(cond_min, 1e-2);
    real_t GET_SCAL_OPT(dif_rcd, 0.0);
    real_t GET_SCAL_OPT(dif_tol, 1e-2);
    int GET_SCAL_OPT(dif_it, 16);
    int GET_SCAL_OPT(it_max, 1e3);
    int GET_SCAL_OPT(verbose, 1e1);

    /***  prepare output  ***/

    plhs[0] = mxCreateNumericMatrix(D, V, mxREAL_CLASS, mxREAL);
    real_t *X = (real_t*) mxGetPr(plhs[0]);

    real_t* Obj = nlhs > 1 ?
        (real_t*) mxMalloc(sizeof(real_t)*(it_max + 1)) : nullptr;
    real_t *Dif = nlhs > 2 ?
        (real_t*) mxMalloc(sizeof(real_t)*it_max) : nullptr;

    /**  preconditioned forward-Douglas-Rachford  **/

    Pfdr_d1_lsx<real_t, vertex_t> *pfdr =
        new Pfdr_d1_lsx<real_t, vertex_t>
            (V, E, edges, loss, D, Y, d11_metric);

    pfdr->set_edge_weights(edge_weights, homo_edge_weight);
    pfdr->set_loss(loss_weights);
    pfdr->set_conditioning_param(cond_min, dif_rcd);
    pfdr->set_relaxation(rho);
    pfdr->set_algo_param(dif_tol, dif_it, it_max, verbose);
    pfdr->set_monitoring_arrays(Obj, Dif);
    pfdr->set_iterate(X);
    pfdr->initialize_iterate();

    int it = pfdr->precond_proximal_splitting();

    pfdr->set_iterate(nullptr); // prevent X to be free()'d
    delete pfdr;

    /**  resize monitoring arrays and assign to outputs  **/
    if (nlhs > 1){
        plhs[1] = resize_and_create_mxRow(Obj, it + 1, mxREAL_CLASS);
    }
    if (nlhs > 2){
        plhs[2] = resize_and_create_mxRow(Dif, it, mxREAL_CLASS);
    }

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    /* real type is determined by the second parameter Y */
    if (mxIsDouble(prhs[1])){
        pfdr_d1_lsx_mex<double, mxDOUBLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }else{
        pfdr_d1_lsx_mex<float, mxSINGLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }
}
