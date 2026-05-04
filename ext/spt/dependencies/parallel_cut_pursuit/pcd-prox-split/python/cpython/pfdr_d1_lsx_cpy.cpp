/*=============================================================================
 * X, [Obj, Dif] = pfdr_d1_lsx_cpy(loss, Y, edges, edge_weights, loss_weights,
 *      d11_metric, rho, cond_min, dif_rcd, dif_tol, dif_it, it_max, verbose,
 *      real_t_double, compute_Obj, compute_Dif)
 * 
 * Baudoin Camille 2019, Raguet Hugo 2022
 *===========================================================================*/
#include <cstdint>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "pfdr_d1_lsx.hpp"

using namespace std;

/* vertex_t must be able to represent the number of vertices and of
 * (undirected) edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph, as well as the dimension D */
typedef uint32_t vertex_t;

/* template for handling both single and double precisions */
template<typename real_t, NPY_TYPES NPY_REAL>
static PyObject* pfdr_d1_lsx(real_t loss, PyArrayObject* py_Y,
    PyArrayObject* py_edges, PyArrayObject* py_edge_weights, 
    PyArrayObject* py_loss_weights, PyArrayObject* py_d11_metric,
    real_t rho, real_t cond_min, real_t dif_rcd, real_t dif_tol, int dif_it,
    int it_max, int verbose, int compute_Obj, int compute_Dif)
{
    /**  get inputs  **/
    /* sizes and loss */
    npy_intp * py_Y_size = PyArray_DIMS(py_Y);
    size_t D = py_Y_size[0];
    vertex_t V = py_Y_size[1]; 

    const real_t *Y = (real_t*) PyArray_DATA(py_Y);
    const real_t *loss_weights = (PyArray_SIZE(py_loss_weights) > 0) ?
        (real_t*) PyArray_DATA(py_loss_weights) : nullptr;

    /* graph structure */
    vertex_t E = PyArray_SIZE(py_edges) / 2;
    const vertex_t *edges = (vertex_t*) PyArray_DATA(py_edges);

    /* penalizations */
    const real_t *edge_weights = (PyArray_SIZE(py_edge_weights) > 1) ?
        (real_t*) PyArray_DATA(py_edge_weights) : nullptr;
    real_t* ptr_edge_weights = (real_t*) PyArray_DATA(py_edge_weights);
    real_t homo_edge_weight = (PyArray_SIZE(py_edge_weights) == 1) ?
        ptr_edge_weights[0] : 1;
    const real_t* d11_metric = (PyArray_SIZE(py_d11_metric) > 0) ?
        (real_t*) PyArray_DATA(py_d11_metric) : nullptr;

    /**  create output  **/
    /* NOTA: no check for successful allocations is performed */

    npy_intp size_py_X[] = {(npy_intp) D, V};
    PyArrayObject* py_X = (PyArrayObject*) PyArray_ZEROS(2, size_py_X, 
        NPY_REAL, 1);
    real_t* X = (real_t*) PyArray_DATA(py_X);

    real_t* Obj = nullptr;
    if (compute_Obj){ Obj = (real_t*) malloc(sizeof(real_t)*(it_max + 1)); }

    real_t* Dif = nullptr;
    if (compute_Dif){ Dif = (real_t*) malloc(sizeof(real_t)*it_max); }

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

    /* retrieve monitoring arrays */
    PyArrayObject* py_Obj = nullptr;
    if (compute_Obj){
        npy_intp size_py_Obj[] = {it + 1};
        py_Obj = (PyArrayObject*) PyArray_ZEROS(1, size_py_Obj, NPY_REAL, 1);
        real_t* Obj_ = (real_t*) PyArray_DATA(py_Obj);
        for (int i = 0; i < size_py_Obj[0]; i++){ Obj_[i] = Obj[i]; }
        free(Obj);
    }

    PyArrayObject* py_Dif = nullptr;
    if (compute_Dif){
        npy_intp size_py_Dif[] = {it};
        py_Dif = (PyArrayObject*) PyArray_ZEROS(1, size_py_Dif, NPY_REAL, 1);
        real_t* Dif_ = (real_t*) PyArray_DATA(py_Dif);
        for (int i = 0; i < size_py_Dif[0]; i++){ Dif_[i] = Dif[i]; }
        free(Dif);
    }

    pfdr->set_iterate(nullptr); // prevent X to be free()'d
    delete pfdr;

    /* build output according to optional output specified */
    if (compute_Obj && compute_Dif){
        return Py_BuildValue("NNN", py_X, py_Obj, py_Dif);
    }else if (compute_Obj){
        return Py_BuildValue("NN", py_X, py_Obj);
    }else if (compute_Dif){
        return Py_BuildValue("NN", py_X, py_Dif);
    }else{
        return Py_BuildValue("N", py_X);
    }
}

static PyObject* pfdr_d1_lsx_cpy(PyObject* self, PyObject* args)
{ 
    PyArrayObject *py_Y, *py_edges, *py_edge_weights, *py_loss_weights, 
        *py_d11_metric;
    double loss, rho, cond_min, dif_rcd, dif_tol;  
    int dif_it, it_max, verbose, real_t_double, compute_Obj, compute_Dif;

    if(!PyArg_ParseTuple(args, "dOOOOOddddiiiiii", &loss, &py_Y,
        &py_edges, &py_edge_weights, &py_loss_weights, &py_d11_metric,  
        &rho, &cond_min, &dif_rcd, &dif_tol, &dif_it, &it_max, &verbose,
        &real_t_double, &compute_Obj, &compute_Dif)){
        return NULL;
    }

    if (real_t_double){ /* real_t type is double */
        return pfdr_d1_lsx<double, NPY_FLOAT64>(loss, py_Y, py_edges,
            py_edge_weights, py_loss_weights, py_d11_metric, rho,
            cond_min, dif_rcd, dif_tol, dif_it, it_max, verbose, compute_Obj,
            compute_Dif);
    }else{ /* real_t type is float */
        return pfdr_d1_lsx<float, NPY_FLOAT32>(loss, py_Y, py_edges,
            py_edge_weights, py_loss_weights, py_d11_metric, rho,
            cond_min, dif_rcd, dif_tol, dif_it, it_max, verbose, compute_Obj,
            compute_Dif);
    }
}

static PyMethodDef pfdr_d1_lsx_methods[] = {
    {"pfdr_d1_lsx_cpy", pfdr_d1_lsx_cpy, METH_VARARGS,
        "wrapper for PFDR loss d1 simplex"},
    {NULL, NULL, 0, NULL}
}; 

static struct PyModuleDef pfdr_d1_lsx_module = {
    PyModuleDef_HEAD_INIT,
    "pfdr_d1_lsx_cpy", /* name of module */
    NULL, /* module documentation, may be null */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    pfdr_d1_lsx_methods, /* actual methods in the module */
    NULL, /* multi-phase initialization, may be null */
    NULL, /* traversal function, may be null */
    NULL, /* clearing function, may be null */
    NULL  /* freeing function, may be null */
};

PyMODINIT_FUNC
PyInit_pfdr_d1_lsx_cpy(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */
    return PyModule_Create(&pfdr_d1_lsx_module);
}
