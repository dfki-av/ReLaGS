/*=============================================================================
 * [X, Obj, Dif] = pfdr_d1_ql1b(Y | AtY, A | AtA, edges, [options])
 *
 * options is a struct with any of the following fields [with default values]:
 *
 *      d1_weights [1.0], Yl1 [none], l1_weights [0.0], low_bnd [-Inf],
 *      upp_bnd [Inf], L [none], rho [1.5], cond_min [1e-2], dif_rcd [0.0],
 *      dif_tol [1e-4], dif_it [32], it_max [1e3], verbose [1e2],
 *      Gram_if_square [true]
 * 
 *  Hugo Raguet 2016, 2018
 *===========================================================================*/
#include <cstdint>
#include <cstring>
#include "mex.h"
#include "pfdr_d1_ql1b.hpp"

using namespace std;

/* vertex_t must be able to represent the numbers of vertices */
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
        mexErrMsgIdAndTxt("MEX", "PFDR graph d1 quadratic l1 bounds: "
            "fourth parameter 'options' should be a structure, (%s given).",
            mxGetClassName(options));
    }

    const int num_allow_opts = 13;
    const char* opts_names[] = {"d1_weights", "Yl1", "l1_weights", "low_bnd",
        "upp_bnd", "L", "rho", "cond_min", "dif_rcd", "dif_tol", "it_max",
        "verbose", "Gram_if_square"};

    const int num_given_opts = mxGetNumberOfFields(options);

    for (int given_opt = 0; given_opt < num_given_opts; given_opt++){
        const char* opt_name = mxGetFieldNameByNumber(options, given_opt);
        int allow_opt;
        for (allow_opt = 0; allow_opt < num_allow_opts; allow_opt++){
            if (strcmp(opt_name, opts_names[allow_opt]) == 0){ break; }
        }
        if (allow_opt == num_allow_opts){
            mexErrMsgIdAndTxt("MEX", "PFDR graph d1 quadratic l1 bounds: "
                "option '%s' unknown.", opt_name);
        }
    }
}

/* function for checking parameter type */
static void check_arg_class(const mxArray* arg, const char* arg_name,
    mxClassID class_id, const char* class_name)
{
    if (mxGetNumberOfElements(arg) > 1 && mxGetClassID(arg) != class_id){
        mexErrMsgIdAndTxt("MEX", "PFDR graph d1 quadratic l1 bounds: "
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
static void pfdr_d1_ql1b_mex(int nlhs, mxArray **plhs, int nrhs,
    const mxArray **prhs)
{
    /***  get inputs  ***/

    const char* real_class_name = mxREAL_CLASS == mxDOUBLE_CLASS ?
        "double" : "single";

    /**  quadratic functional  **/
    size_t N = mxGetM(prhs[1]);
    vertex_t V = mxGetN(prhs[1]);

    if (N == 0 && V == 0){
        mexErrMsgIdAndTxt("MEX", "PFDR graph d1 quadratic l1 bounds: "
            "argument A cannot be empty.");
    }

    check_arg_class(prhs[0], "Y", mxREAL_CLASS, real_class_name);
    check_arg_class(prhs[1], "A", mxREAL_CLASS, real_class_name);

    const real_t* Y = !mxIsEmpty(prhs[0]) ?
        (real_t*) mxGetData(prhs[0]) : nullptr;
    const real_t* A = (N == 1 && V == 1) ?
        nullptr : (real_t*) mxGetData(prhs[1]);
    const real_t a = (N == 1 && V == 1) ?  mxGetScalar(prhs[1]) : 1.0;

    if (V == 1){ /* quadratic functional is only weighted square difference */
        if (N == 1){
            if (!mxIsEmpty(prhs[0])){ /* fidelity is square l2 */
                V = mxGetNumberOfElements(prhs[0]);
            }else{ /* fidelity is only l1; optional Yl1 has been checked */
                V = mxGetNumberOfElements(mxGetField(prhs[3], 0, "Yl1"));
            }
        }else{ /* A is given V-by-1, representing a diagonal V-by-V */
            V = N;
        }
        N = Pfdr_d1_ql1b<real_t, vertex_t>::Gram_diag();
    }else if (V == N && (nrhs < 5 || !mxGetField(prhs[3], 0, "Gram_if_square")
        || mxIsLogicalScalarTrue(mxGetField(prhs[3], 0, "Gram_if_square")))){
        /* A and Y are left-premultiplied by A^t */
        N = Pfdr_d1_ql1b<real_t, vertex_t>::Gram_full();
    }

    /**  graph structure  **/

    check_arg_class(prhs[2], "edges", mxVERTEX_CLASS, VERTEX_T_STRING);

    size_t E = mxGetNumberOfElements(prhs[2])/2;
    if (mxGetN(prhs[2]) == 2 && E > 2){
        mexErrMsgIdAndTxt("MEX", "PFDR prox TV: edges ends must lie "
            "consecutively in memory; since GNU Octave and Matlab use "
            "column-major layout, parameter 'edges' should have size "
            "2-by-%i (%i-by-2 given)", E, E);
    }
    const vertex_t* edges = (vertex_t*) mxGetData(prhs[2]);

    /**  optional parameters  **/

    const mxArray* options = nrhs > 3 ? prhs[3] : nullptr;

    check_opts(options);
    const mxArray* opt;

    /* penalizations */
    #define GET_REAL_OPT(NAME, DFLT) \
        const real_t* NAME = nullptr; \
        real_t homo_ ## NAME = DFLT; \
        if (options && (opt = mxGetField(options, 0, #NAME))){ \
            check_arg_class(opt, #NAME, mxREAL_CLASS, real_class_name); \
            if (mxGetNumberOfElements(opt) > 1){ \
                NAME = (real_t*) mxGetData(opt); \
            }else{ \
                homo_ ## NAME = mxGetScalar(opt); \
            } \
        }

    GET_REAL_OPT(d1_weights, 1.0)
    GET_REAL_OPT(l1_weights, 0.0)
    GET_REAL_OPT(low_bnd, (-Pfdr_d1_ql1b<real_t, vertex_t>::real_inf()))
    GET_REAL_OPT(upp_bnd, (Pfdr_d1_ql1b<real_t, vertex_t>::real_inf()))

    const real_t* Yl1 = nullptr;
    if (opt = mxGetField(options, 0, "Yl1")){
        check_arg_class(opt, "Yl1", mxREAL_CLASS, real_class_name);
        Yl1 = (real_t*) mxGetData(opt);
    }

    const real_t* L = nullptr;
    real_t l = -1.0;
    if (opt = mxGetField(options, 0, "L")){
        check_arg_class(opt, "L", mxREAL_CLASS, real_class_name);
        if (mxGetNumberOfElements(opt) > 1){
            L = (real_t*) mxGetData(opt);
        }else{
            l = mxGetScalar(opt);
        }
    }
   
    /* algorithmic parameters */
    #define GET_SCAL_OPT(NAME, DFLT) \
        NAME = (opt = mxGetField(options, 0, #NAME)) ? mxGetScalar(opt) : DFLT;

    real_t GET_SCAL_OPT(rho, 1.0);
    real_t GET_SCAL_OPT(cond_min, 1e-2);
    real_t GET_SCAL_OPT(dif_rcd, 0.0);
    real_t GET_SCAL_OPT(dif_tol, 1e-4);
    int GET_SCAL_OPT(dif_it, 32);
    int GET_SCAL_OPT(it_max, 1e3);
    int GET_SCAL_OPT(verbose, 1e2);

    /***  prepare output  ***/

    plhs[0] = mxCreateNumericMatrix(V, 1, mxREAL_CLASS, mxREAL);
    real_t *X = (real_t*) mxGetPr(plhs[0]);

    real_t* Obj = nlhs > 1 ?
        (real_t*) mxMalloc(sizeof(real_t)*(it_max + 1)) : nullptr;
    real_t *Dif = nlhs > 2 ?
        (real_t*) mxMalloc(sizeof(real_t)*it_max) : nullptr;

    /***  preconditioned forward-Douglas-Rachford  ***/

    Pfdr_d1_ql1b<real_t, vertex_t> *pfdr =
        new Pfdr_d1_ql1b<real_t, vertex_t>(V, E, edges);

    pfdr->set_edge_weights(d1_weights, homo_d1_weights);
    pfdr->set_quadratic(Y, N, A, a);
    pfdr->set_l1(l1_weights, homo_l1_weights, Yl1);
    pfdr->set_bounds(low_bnd, homo_low_bnd, upp_bnd, homo_upp_bnd);
    if (L || l >= 0.0){
        pfdr->set_lipschitz_param(L, l, L ?
            Pfdr<real_t, vertex_t>::MONODIM : Pfdr<real_t, vertex_t>::SCALAR);
    }
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
    /* real type is determined by the first parameter Y
     * or by the optional parameter Yl1 */
    if (mxIsEmpty(prhs[0]) && (nrhs < 4 || !mxGetField(prhs[3], 0, "Yl1"))){
        mexErrMsgIdAndTxt("MEX", "PFDR graph d1 quadratic l1 bounds: "
            "parameter Y and optional parameter Yl1 cannot be both empty.");
    }

    if ((!mxIsEmpty(prhs[0]) && mxIsDouble(prhs[0])) ||
        (mxIsEmpty(prhs[0]) && mxIsDouble(mxGetField(prhs[3], 0, "Yl1")))){
        pfdr_d1_ql1b_mex<double, mxDOUBLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }else{
        pfdr_d1_ql1b_mex<float, mxSINGLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }
}
