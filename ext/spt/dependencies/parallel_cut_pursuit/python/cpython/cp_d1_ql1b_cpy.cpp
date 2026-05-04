/*=============================================================================
 * Comp, rX, [List, Graph, Obj, Time, Dif] = cp_d1_ql1b_cpy(Y, A, first_edge,
 *  adj_vertices, edge_weights, Yl1, l1_weights, low_bnd, upp_bnd, cp_dif_tol,
 *  cp_it_max, pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol,
 *  pfdr_it_max, verbose, max_num_threads, max_split_size,
 *  balance_parallel_split, Gram_if_square, real_is_double, compute_Obj,
 *  compute_Time, compute_Dif)
 * 
 *  Baudoin Camille 2019, Raguet Hugo 2022, 2023
 *===========================================================================*/
#include <cstdint>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "cp_d1_ql1b.hpp" 

using namespace std;

/* index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph */
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned iterator in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
    typedef int32_t index_t;
    #define NPY_IND NPY_INT32
    #ifndef COMP_T_ON_32_BITS
        typedef int16_t comp_t;
        #define NPY_COMP NPY_INT16
    #else /* more than 32767 components are expected */
        typedef int32_t comp_t;
        #define NPY_COMP NPY_INT32
    #endif
#else
    typedef uint32_t index_t;
    #define NPY_IND NPY_UINT32
    #ifndef COMP_T_ON_32_BITS
        typedef uint16_t comp_t;
        #define NPY_COMP NPY_UINT16
    #else /* more than 65535 components are expected */
        typedef uint32_t comp_t;
        #define NPY_COMP NPY_UINT32
    #endif
#endif

/* template for handling both single and double precisions */
template<typename real_t, NPY_TYPES NPY_REAL>
static PyObject* cp_d1_ql1b(PyArrayObject* py_Y,
    PyArrayObject* py_A, PyArrayObject* py_first_edge,
    PyArrayObject* py_adj_vertices, PyArrayObject* py_edge_weights,
    PyArrayObject* py_Yl1, PyArrayObject* py_l1_weights,
    PyArrayObject* py_low_bnd, PyArrayObject* py_upp_bnd, real_t cp_dif_tol,
    int cp_it_max, real_t pfdr_rho, real_t pfdr_cond_min, real_t pfdr_dif_rcd,
    real_t pfdr_dif_tol, int pfdr_it_max, int verbose, int max_num_threads, 
    index_t max_split_size, int balance_parallel_split, int Gram_if_square,
    int compute_List, int compute_Graph, int compute_Obj, int compute_Time,
    int compute_Dif)
{
    /**  get inputs  **/

    /* quadratic functional */
    npy_intp* py_A_dims = PyArray_DIMS(py_A);
    size_t N = py_A_dims[0];
    index_t V = PyArray_NDIM(py_A) > 1 ? py_A_dims[1] : 1;

    const real_t* Y = PyArray_SIZE(py_Y) > 0 ?
        (real_t*) PyArray_DATA(py_Y) : nullptr;
    const real_t* A = (N == 1 && V == 1) ?
        nullptr : (real_t*) PyArray_DATA(py_A); 
    real_t * ptr_A = (real_t*) PyArray_DATA(py_A);
    const real_t a = (N == 1 && V == 1) ?
        ptr_A[0] : 1.0; 

    if (V == 1){ /* quadratic functional is only weighted square difference */
        if (N == 1){
            if (PyArray_SIZE(py_Y) > 0){ /* fidelity is square l2 */
                V = PyArray_SIZE(py_Y);
            }else if (PyArray_SIZE(py_Yl1) > 0){
                /* fidelity is only l1 */
                V = PyArray_SIZE(py_Yl1);
            }
        }else{ /* A is given V-by-1, representing a diagonal V-by-V */
            V = N;
        }
        N = Cp_d1_ql1b<real_t, index_t, comp_t>::Gram_diag();
    }else if (V == N && Gram_if_square){
        /* A and Y are left-premultiplied by A^t */
        N = Cp_d1_ql1b<real_t, index_t, comp_t>::Gram_full(); 
    }

    /* graph structure */
    index_t E = PyArray_SIZE(py_adj_vertices);
    const index_t* first_edge = (index_t*) PyArray_DATA(py_first_edge); 
    const index_t* adj_vertices = (index_t*) PyArray_DATA(py_adj_vertices); 

    /* penalizations */
    const real_t* edge_weights = (real_t*) PyArray_DATA(py_edge_weights);
    real_t homo_edge_weight = PyArray_SIZE(py_edge_weights) == 1 ? 
        edge_weights[0] : 1.0;
    if (PyArray_SIZE(py_edge_weights) <= 1){ edge_weights = nullptr; }

    const real_t* Yl1 = PyArray_SIZE(py_Yl1) > 0 ?
        (real_t*) PyArray_DATA(py_Yl1) : nullptr;

    const real_t* l1_weights = (real_t*) PyArray_DATA(py_l1_weights);
    real_t homo_l1_weight = PyArray_SIZE(py_l1_weights) == 1 ?
        l1_weights[0] : 0.0;
    if (PyArray_SIZE(py_l1_weights) <= 1){ l1_weights = nullptr; }

    const real_t* low_bnd = (real_t*) PyArray_DATA(py_low_bnd);
    real_t homo_low_bnd = PyArray_SIZE(py_low_bnd) == 1 ? 
        low_bnd[0] : -Cp_d1_ql1b<real_t, index_t, comp_t>::real_inf();
    if (PyArray_SIZE(py_low_bnd) <= 1){ low_bnd = nullptr; }

    const real_t* upp_bnd = (real_t*) PyArray_DATA(py_upp_bnd);
    real_t homo_upp_bnd = PyArray_SIZE(py_upp_bnd) == 1 ?
        upp_bnd[0] : Cp_d1_ql1b<real_t, index_t, comp_t>::real_inf();
    if (PyArray_SIZE(py_upp_bnd) <= 1){ upp_bnd = nullptr; }

    /* number of threads */ 
    if (max_num_threads <= 0){ max_num_threads = omp_get_max_threads(); }

    /**  prepare output; rX is created later  **/
    /* NOTA: no check for successful allocations is performed */

    npy_intp size_py_Comp[] = {V};
    PyArrayObject* py_Comp = (PyArrayObject*) PyArray_ZEROS(1, size_py_Comp,
        NPY_COMP, 1);
    comp_t* Comp = (comp_t*) PyArray_DATA(py_Comp); 

    real_t* Obj = nullptr;
    if (compute_Obj){ Obj = (real_t*) malloc(sizeof(real_t)*(cp_it_max + 1)); }

    double* Time = nullptr;
    if (compute_Time){
        Time = (double*) malloc(sizeof(double)*(cp_it_max + 1));
    }

    real_t* Dif = nullptr;
    if (compute_Dif){ Dif = (real_t*) malloc(sizeof(real_t)*cp_it_max); }

    /**  cut-pursuit with preconditioned forward-Douglas-Rachford  **/

    Cp_d1_ql1b<real_t, index_t, comp_t>* cp =
        new Cp_d1_ql1b<real_t, index_t, comp_t>
            (V, E, first_edge, adj_vertices);

    cp->set_edge_weights(edge_weights, homo_edge_weight);
    cp->set_quadratic(Y, N, A, a);
    cp->set_l1(l1_weights, homo_l1_weight, Yl1);
    cp->set_bounds(low_bnd, homo_low_bnd, upp_bnd, homo_upp_bnd);
    cp->set_cp_param(cp_dif_tol, cp_it_max, verbose);
    cp->set_split_param(max_split_size);
    cp->set_pfdr_param(pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_it_max,
        pfdr_dif_tol);
    cp->set_parallel_param(max_num_threads, balance_parallel_split);
    cp->set_monitoring_arrays(Obj, Time, Dif);
    cp->set_components(0, Comp); // use the preallocated component array Comp

    int cp_it = cp->cut_pursuit();

    /* get number of components and their lists of indices if necessary */
    const index_t* first_vertex;
    const index_t* comp_list;
    comp_t rV = cp->get_components(nullptr, &first_vertex, &comp_list);

    PyObject* py_List = nullptr;
    if (compute_List){
        py_List = PyList_New(rV); // list of arrays
        for (comp_t rv = 0; rv < rV; rv++){
            index_t comp_size = first_vertex[rv+1] - first_vertex[rv];
            npy_intp size_py_List_rv[] = {comp_size};
            PyArrayObject* py_List_rv = (PyArrayObject*) PyArray_ZEROS(1,
                size_py_List_rv, NPY_IND, 1);
            index_t* List_rv = (index_t*) PyArray_DATA(py_List_rv);
            for (index_t i = 0; i < comp_size; i++){
                List_rv[i] = comp_list[first_vertex[rv] + i];
            }
            PyList_SetItem(py_List, rv, (PyObject*) py_List_rv);
        }
    }

    /* copy reduced values */
    const real_t* cp_rX = cp->get_reduced_values();
    npy_intp size_py_rX[] = {rV};
    PyArrayObject* py_rX = (PyArrayObject*) PyArray_ZEROS(1, size_py_rX,
        NPY_REAL, 1);
    real_t* rX = (real_t*) PyArray_DATA(py_rX);
    for (comp_t rv = 0; rv < rV; rv++){ rX[rv] = cp_rX[rv]; }

    /* retrieve reduced graph structure */
    PyObject* py_Graph = nullptr;
    if (compute_Graph){
        const comp_t* reduced_edge_list;
        const real_t* reduced_edge_weights;
        size_t rE;
        /* get reduced edge list */
        rE = cp->get_reduced_graph(&reduced_edge_list, &reduced_edge_weights);

        /* numpy arrays for forward-star representation and weights */
        npy_intp size_py_red_first_edge[] = {rV + 1};
        PyArrayObject* py_red_first_edge = (PyArrayObject*)
            PyArray_ZEROS(1, size_py_red_first_edge, NPY_IND, 1);
        index_t* red_first_edge = (index_t*) PyArray_DATA(py_red_first_edge);

        npy_intp size_py_red_adj_vertices[] = {(npy_intp/* supp. warning*/)rE};
        PyArrayObject* py_red_adj_vertices = (PyArrayObject*)
            PyArray_ZEROS(1, size_py_red_adj_vertices, NPY_COMP, 1);
        comp_t* red_adj_vertices = (comp_t*) PyArray_DATA(py_red_adj_vertices);

        npy_intp size_py_red_edge_weights[] = {(npy_intp/* supp. warning*/)rE};
        PyArrayObject* py_red_edge_weights = (PyArrayObject*)
            PyArray_ZEROS(1, size_py_red_edge_weights, NPY_REAL, 1);
        real_t* red_edge_weights = (real_t*) PyArray_DATA(py_red_edge_weights);

        /* reduced edge list is guaranteed to be in increasing order of
         * starting component; conversion to forward-star is straightforward */
        comp_t rv = 0;
        size_t re = 0;
        while (re < rE || rv < rV){
            red_first_edge[rv] = re;
            while (re < rE && reduced_edge_list[2*re] == rv){
                red_adj_vertices[re] = reduced_edge_list[2*re + 1];
                red_edge_weights[re] = reduced_edge_weights[re];
                re++;
            }
            rv++;
        }
        red_first_edge[rV] = rE;

        /* gather forward-star representation and weights in python tuple */
        py_Graph = PyTuple_New(3);
        PyTuple_SET_ITEM(py_Graph, 0, (PyObject*) py_red_first_edge);
        PyTuple_SET_ITEM(py_Graph, 1, (PyObject*) py_red_adj_vertices);
        PyTuple_SET_ITEM(py_Graph, 2, (PyObject*) py_red_edge_weights);
    }

    /* retrieve monitoring arrays */
    PyArrayObject* py_Obj = nullptr;
    if (compute_Obj){
        npy_intp size_py_Obj[] = {cp_it + 1};
        py_Obj = (PyArrayObject*) PyArray_ZEROS(1, size_py_Obj, NPY_REAL, 1);
        real_t* Obj_ = (real_t*) PyArray_DATA(py_Obj);
        for (int i = 0; i < size_py_Obj[0]; i++){ Obj_[i] = Obj[i]; }
        free(Obj);
    }

    PyArrayObject* py_Time = nullptr;
    if (compute_Time){
        npy_intp size_py_Time[] = {cp_it + 1};
        py_Time = (PyArrayObject*) PyArray_ZEROS(1, size_py_Time, NPY_FLOAT64,
            1);
        double* Time_ = (double*) PyArray_DATA(py_Time);
        for (int i = 0; i < size_py_Time[0]; i++){ Time_[i] = Time[i]; }
        free(Time);
    }

    PyArrayObject* py_Dif = nullptr;
    if (compute_Dif){
        npy_intp size_py_Dif[] = {cp_it};
        py_Dif = (PyArrayObject*) PyArray_ZEROS(1, size_py_Dif, NPY_REAL, 1);
        real_t* Dif_ = (real_t*) PyArray_DATA(py_Dif);
        for (int i = 0; i < size_py_Dif[0]; i++){ Dif_[i] = Dif[i]; }
        free(Dif);
    }

    cp->set_components(0, nullptr); // prevent Comp to be free()'d
    delete cp;

    /* build output according to optional output specified */
    Py_ssize_t nout = 2;
    if (compute_List){ nout++; }
    if (compute_Graph){ nout++; }
    if (compute_Obj){ nout++; }
    if (compute_Time){ nout++; }
    if (compute_Dif){ nout++; }
    PyObject* py_Out = PyTuple_New(nout);
    PyTuple_SET_ITEM(py_Out, 0, (PyObject*) py_Comp);
    PyTuple_SET_ITEM(py_Out, 1, (PyObject*) py_rX);
    nout = 2;
    if (compute_List){ PyTuple_SET_ITEM(py_Out, nout++, py_List); }
    if (compute_Graph){ PyTuple_SET_ITEM(py_Out, nout++, py_Graph); }
    if (compute_Obj){ PyTuple_SET_ITEM(py_Out, nout++, (PyObject*) py_Obj); }
    if (compute_Time){ PyTuple_SET_ITEM(py_Out, nout++, (PyObject*) py_Time); }
    if (compute_Dif){ PyTuple_SET_ITEM(py_Out, nout++, (PyObject*) py_Dif); }

    return py_Out;
}

/* actual interface */
#if PY_VERSION_HEX >= 0x03040000 // Py_UNUSED suppress warning from 3.4
static PyObject* cp_d1_ql1b_cpy(PyObject* Py_UNUSED(self), PyObject* args)
{
#else
static PyObject* cp_d1_ql1b_cpy(PyObject* self, PyObject* args)
{   (void) self; // suppress unused parameter warning
#endif
    /* INPUT */ 
    PyArrayObject *py_Y, *py_A, *py_first_edge, *py_adj_vertices,
        *py_edge_weights, *py_Yl1, *py_l1_weights, *py_low_bnd, *py_upp_bnd; 
    double cp_dif_tol, pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol;
    int cp_it_max, pfdr_it_max, verbose, max_num_threads, max_split_size,
        balance_parallel_split, Gram_if_square, real_is_double, compute_List,
        compute_Graph, compute_Obj, compute_Time, compute_Dif; 
    
    /* parse the input, from Python Object to C PyArray, double, or int type */
    if(!PyArg_ParseTuple(args, "OOOOOOOOOdiddddiiiiiiiiiiii", &py_Y, &py_A,
        &py_first_edge, &py_adj_vertices, &py_edge_weights, &py_Yl1,
        &py_l1_weights, &py_low_bnd, &py_upp_bnd, &cp_dif_tol, &cp_it_max,
        &pfdr_rho, &pfdr_cond_min, &pfdr_dif_rcd, &pfdr_dif_tol, &pfdr_it_max,
        &verbose, &max_num_threads, &max_split_size, &balance_parallel_split,
        &Gram_if_square, &real_is_double, &compute_List, &compute_Graph,
        &compute_Obj, &compute_Time, &compute_Dif)){
        return NULL;
    }

    if (real_is_double){ /* real_t type is double */
        return cp_d1_ql1b<double, NPY_FLOAT64>(py_Y, py_A, py_first_edge,
            py_adj_vertices, py_edge_weights, py_Yl1, py_l1_weights,
            py_low_bnd, py_upp_bnd, cp_dif_tol, cp_it_max, pfdr_rho,
            pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max, verbose,
            max_num_threads, max_split_size, balance_parallel_split,
            Gram_if_square, compute_List, compute_Graph, compute_Obj,
            compute_Time, compute_Dif);
    }else{ /* real_t type is float */
        return cp_d1_ql1b<float, NPY_FLOAT32>(py_Y, py_A, py_first_edge,
            py_adj_vertices, py_edge_weights, py_Yl1, py_l1_weights,
            py_low_bnd, py_upp_bnd, cp_dif_tol, cp_it_max, pfdr_rho,
            pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max, verbose,
            max_num_threads, max_split_size, balance_parallel_split,
            Gram_if_square, compute_List, compute_Graph, compute_Obj,
            compute_Time, compute_Dif);
    }
}

static PyMethodDef cp_d1_ql1b_methods[] = {
    {"cp_d1_ql1b_cpy", cp_d1_ql1b_cpy, METH_VARARGS,
        "wrapper for parallel cut-pursuit quadratic d1 l1 bounds"},
    {NULL, NULL, 0, NULL}
};

/* module initialization */
#if PY_MAJOR_VERSION >= 3
/* Python version 3 */
static struct PyModuleDef cp_d1_ql1b_module = {
    PyModuleDef_HEAD_INIT,
    "cp_d1_ql1b_cpy", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    cp_d1_ql1b_methods,
    NULL, /* multi-phase initialization, may be null */
    NULL, /* traversal function, may be null */
    NULL, /* clearing function, may be null */
    NULL  /* freeing function, may be null */
};

PyMODINIT_FUNC
PyInit_cp_d1_ql1b_cpy(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */
    return PyModule_Create(&cp_d1_ql1b_module);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC
initcp_d1_ql1b_cpy(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */
    (void) Py_InitModule("cp_d1_ql1b_cpy", cp_d1_ql1b_methods);
}

#endif
