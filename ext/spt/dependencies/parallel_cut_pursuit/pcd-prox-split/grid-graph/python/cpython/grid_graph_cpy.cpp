/*=============================================================================
 * Python extension module for creating and manipulating grid graph structures
 * in edge list or forward star representations
 * 
 * Hugo Raguet 2020
 *===========================================================================*/
#include <cstdint>
#include <limits>
#include <cstdio>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "grid_graph.hpp"

typedef uint8_t conn_t;
# define NPY_CONN NPY_UINT8
# define CONN_T_STRING "uint8"

/* static global character string for errors */
static char err_msg[1000];

/* template for handling several index types in edge_list_to_forward_star */
template <typename index_t, NPY_TYPES NPY_INDEX>
static PyObject* edge_list_to_forward_star(size_t V, size_t E,
    PyArrayObject *py_edges)
{
    if (V > std::numeric_limits<index_t>::max()){
        std::sprintf(err_msg, "Edge list to forward star: the number of "
            "vertices 'V' cannot be represented by the given integer type "
            "(%lu provided, no more than %lu allowed)", V,
            (size_t) std::numeric_limits<index_t>::max());
        PyErr_SetString(PyExc_ValueError, err_msg);
        return NULL;
    }
    if (E > std::numeric_limits<index_t>::max()){
        std::sprintf(err_msg, "Edge list to forward star: the number of "
            "edges cannot be represented by the given integer type "
            "(%lu provided, no more than %lu allowed)", E,
            (size_t) std::numeric_limits<index_t>::max());
        PyErr_SetString(PyExc_ValueError, err_msg);
        return NULL;
    }

    const index_t* edges = (index_t*) PyArray_DATA(py_edges);

    npy_intp size_py_first_edge[] = {(npy_intp) V + 1};
    PyArrayObject* py_first_edge = (PyArrayObject*) PyArray_Zeros(1,
        size_py_first_edge, PyArray_DescrFromType(NPY_INDEX), 0);
    index_t* first_edge = (index_t*) PyArray_DATA(py_first_edge);

    npy_intp size_py_reindex[] = {(npy_intp) E};
    PyArrayObject* py_reindex = (PyArrayObject*) PyArray_Zeros(1,
        size_py_reindex, PyArray_DescrFromType(NPY_INDEX), 0);
    index_t* reindex = (index_t*) PyArray_DATA(py_reindex);

    edge_list_to_forward_star<index_t, index_t>(V, E, edges, first_edge,
        reindex);

    npy_intp size_py_adj_vertices[] = {(npy_intp) E};
    PyArrayObject* py_adj_vertices = (PyArrayObject*) PyArray_Zeros(1,
        size_py_adj_vertices, PyArray_DescrFromType(NPY_INDEX), 0);
    index_t* adj_vertices = (index_t*) PyArray_DATA(py_adj_vertices);
    for (size_t e = 0; e < E; e++){
        adj_vertices[reindex[e]] = edges[2*e + 1];
    }

    return Py_BuildValue("OOO", py_first_edge, py_adj_vertices, py_reindex);
}

/* actual interface for edge_list_to_forward_star */
#if PY_VERSION_HEX >= 0x03040000 // Py_UNUSED suppress warning from 3.4
static PyObject* edge_list_to_forward_star_cpy(PyObject* Py_UNUSED(self),
    PyObject* args)
{ 
#else
static PyObject* edge_list_to_forward_star_cpy(PyObject* self, PyObject* args)
{   (void) self; // suppress unused parameter warning
#endif

    /***  get and check inputs  ***/
    Py_ssize_t V;
    PyArrayObject *py_edges;

    if (!PyArg_ParseTuple(args, "nO", &V, &py_edges)){ return NULL; }

    if (!PyArray_Check(py_edges)){
        PyErr_SetString(PyExc_TypeError, "Edge list to forward star: argument "
            "'edges' must be a numpy array.");
        return NULL;
    }

    const npy_intp* dims = PyArray_DIMS(py_edges);
    size_t E;

    if (PyArray_NDIM(py_edges) == 2){
        if (dims[0] == 2){ /* 2-by-E array, must be column-major */
            if (!PyArray_IS_F_CONTIGUOUS(py_edges)){
                PyErr_SetString(PyExc_TypeError, "Edge list to forward star: "
                    "internal memory of 'edges' must store each edge "
                    "contiguously; a 2-by-E array must be column-major "
                    "(F-contiguous).");
                return NULL;
            }
            E = dims[1];
        }else if (dims[1] == 2){ /* E-by-2 array, must be row-major */
            if (!PyArray_IS_C_CONTIGUOUS(py_edges)){
                PyErr_SetString(PyExc_TypeError, "Edge list to forward star: "
                    "internal memory of 'edges' must store each edge "
                    "contiguously; a E-by-2 array must be row-major "
                    "(C-contiguous).");
                return NULL;
            }
            E = dims[0];
        }else{
            std::sprintf(err_msg, "Edge list to forward star: when 'edges' "
                "is two-dimensional, one of the dimensions must be 2 "
                "(%li-by-%li given).", dims[0], dims[1]);
            PyErr_SetString(PyExc_TypeError, err_msg);
            return NULL;
        }
    }else{ /* treat as monodimensional, hoping for right memory layout */
        E = PyArray_SIZE(py_edges)/2;
    }

    if (!PyArray_ISINTEGER(py_edges)){
        PyErr_SetString(PyExc_TypeError, "Edge list to forward star: elements "
            "in 'edges' must be of integer type.");
        return NULL;
    }else if (PyArray_TYPE(py_edges) == NPY_INT16 ||
              PyArray_TYPE(py_edges) == NPY_UINT16){
        return edge_list_to_forward_star<uint16_t, NPY_UINT16>(V, E, py_edges);
    }else if (PyArray_TYPE(py_edges) == NPY_INT32 ||
              PyArray_TYPE(py_edges) == NPY_UINT32){
        return edge_list_to_forward_star<uint32_t, NPY_UINT32>(V, E, py_edges);
    }else if (PyArray_TYPE(py_edges) == NPY_INT64 ||
              PyArray_TYPE(py_edges) == NPY_UINT64){
        return edge_list_to_forward_star<uint64_t, NPY_UINT64>(V, E, py_edges);
    }else{
        PyErr_SetString(PyExc_TypeError, "Edge list to forward star: not "
            "implemented for the given integer type.");
        return NULL;
    }
}

static const char* edge_list_to_forward_star_doc = 
"first_edge, adj_vertices, reindex = edge_list_to_forward_star(V, edges)\n"
"\n"
"Convert the graph representation given by the edge list 'edges' into a\n"
"forward-star representation 'first_edge', 'adj_vertices'.\n"
"\n"
"A graph with V vertices and E edges is represented either as edge list\n"
"(array of E edges given as ordered pair of vertices), or as forward star,\n"
"where edges are numeroted so that all edges originating from a same vertex\n"
"are consecutive, and represented by the following parameters:\n"
"first_edge - array of length V + 1, indicating for each vertex, the first\n"
"     edge starting from the vertex (or, if there are none, starting from\n"
"     the next vertex); the first value is always zero and the last value is\n"
"     always the total number of edges\n"
"adj_vertices - array of length E, indicating for each edge, its ending\n"
"     vertex\n"
"\n"
"NOTA: edges and vertices identifiers are integers (start at 0), whose type\n"
"is either uint16, uint32 or uint64, depending on the type of 'edges' input.\n"
"\n"
"INPUTS:\n"
" V - the number of vertices in the graph; usually max(edges) + 1\n"
" edges - list of edges, numpy array of integers, each edges being given by\n"
"         two consecutive vertex identifiers; can thus be a E-by-2 row-major\n"
"         (C-contiguous) array or 2-by-E column-major (F-contiguous) array,\n"
"         or unidimensionnal array of length 2E with correct memory layout.\n"
"\n"
"OUTPUTS:\n"
" first_edge, adj_vertices - see above.\n"
" reindex - for all edges originating from a same vertex being consecutive,\n"
"           they must be reordered; reindex keep track of this permutation\n"
"           so that edge number 'e' in the original list becomes edge number\n"
"           reindex[e] in the forward-star structure; in practice, in order\n"
"           to have the same order in both structures, permute the original\n"
"           list with `edges[reindex,:] = numpy.copy(edges)`\n"
"           (note that row-major layout is assumed here).\n"
"\n"
"Hugo Raguet 2020\n";

/* template for handling several index types in grid_to_graph */
template <typename index_t, NPY_TYPES NPY_INDEX>
static PyObject* grid_to_graph(PyArrayObject* py_grid_shape,
    conn_t connectivity, bool compute_connectivities,
    bool graph_as_forward_star, bool row_major_index)
{
    /* retrieve inputs */
    size_t D = PyArray_SIZE(py_grid_shape);
    index_t* grid_shape = (index_t*) PyArray_DATA(py_grid_shape);

    index_t* shape;
    if (row_major_index){
        shape = (index_t*) malloc(sizeof(index_t)*D);
        for (size_t d = 0; d < D; d++){ shape[d] = grid_shape[D - 1 - d]; }
    }else{ /* C++ routine uses column-major order */
        shape = grid_shape;
    }
    
    /* retrieve sizes */
    size_t E = num_edges_grid_graph<index_t, conn_t>(D, shape, connectivity);
    size_t V = 1;
    for (size_t d = 0; d < D; d++){ V *= (size_t) shape[d]; }
    
    if (V > std::numeric_limits<index_t>::max()){
        std::sprintf(err_msg, "Grid to graph: the number of "
            "vertices cannot be represented by the given integer type "
            "(%lu vertices requested, no more than %lu allowed)", V,
            (size_t) std::numeric_limits<index_t>::max());
        PyErr_SetString(PyExc_ValueError, err_msg);
        return NULL;
    }
    if (graph_as_forward_star && E > std::numeric_limits<index_t>::max()){
        std::sprintf(err_msg, "Grid to graph: the number of "
            "edges cannot be represented by the given integer type "
            "(%lu edges requested, no more than %lu allowed)", E,
            (size_t) std::numeric_limits<index_t>::max());
        PyErr_SetString(PyExc_ValueError, err_msg);
        return NULL;
    }

    /* compute edges and connectivities */
    npy_intp size_py_edges[] = {(npy_intp) E, 2};
    PyArrayObject* py_edges = (PyArrayObject*) PyArray_Zeros(2,
        size_py_edges, PyArray_DescrFromType(NPY_INDEX), 0);
    index_t* edges = (index_t*) PyArray_DATA(py_edges);

    conn_t* connectivities = nullptr;
    PyArrayObject* py_connectivities = nullptr;
    if (compute_connectivities){
        npy_intp size_py_connectivities[] = {(npy_intp) E};
        py_connectivities = (PyArrayObject*) PyArray_Zeros(1,
            size_py_connectivities, PyArray_DescrFromType(NPY_CONN), 0);
        connectivities = (conn_t*) PyArray_DATA(py_connectivities);
    }

    edge_list_grid_graph<index_t, conn_t>(D, shape, connectivity, edges,
        connectivities);

    if (row_major_index){ free(shape); }

    if (!graph_as_forward_star){
        if (compute_connectivities){
            return Py_BuildValue("OO", py_edges, py_connectivities);
        }else{
            return Py_BuildValue("O", py_edges);
        }
    }

    /* convert to forward star representation */
    npy_intp size_py_first_edge[] = {(npy_intp) V + 1};
    PyArrayObject* py_first_edge = (PyArrayObject*) PyArray_Zeros(1,
        size_py_first_edge, PyArray_DescrFromType(NPY_INDEX), 0);
    index_t* first_edge = (index_t*) PyArray_DATA(py_first_edge);

    index_t* reindex = (index_t*) malloc(sizeof(index_t)*E);

    edge_list_to_forward_star<index_t, index_t>(V, E, edges, first_edge,
        reindex);

    /* override source vertices with permuted end vertices */
    for (size_t e = 0; e < E; e++){
        edges[(size_t) 2*reindex[e]] = edges[2*e + 1];
    }
    /* store resulting adjacent vertices in first half of edges storage */
    index_t* adj_vertices = edges;
    for (size_t e = 0; e < E; e++){ adj_vertices[e] = edges[2*e]; }
    
    /* permute connectivity correspondingly if requested */
    if (compute_connectivities){
        index_t* buf = edges + E; // reuse second half of edges storage
        for (size_t e = 0; e < E; e++){ buf[reindex[e]] = connectivities[e]; }
        for (size_t e = 0; e < E; e++){ connectivities[e] = buf[e]; }
    }

    free(reindex);

    /* pass py_edges reference to py_adj_vertices and resize data */
    PyArrayObject* py_adj_vertices = py_edges;
    npy_intp size_py_adj_vertices[] = {(npy_intp) E};
    PyArray_Dims dims_py_adj_vertices = {size_py_adj_vertices, 1};
    PyArray_Resize(py_adj_vertices, &dims_py_adj_vertices, 0, NPY_ANYORDER);

    if (compute_connectivities){
        return Py_BuildValue("OOO", py_first_edge, py_adj_vertices,
            py_connectivities);
    }else{
        return Py_BuildValue("OO", py_first_edge, py_adj_vertices);
    }
}

/* actual interface for grid_to_graph */
#if PY_VERSION_HEX >= 0x03040000 // Py_UNUSED suppress warning from 3.4
static PyObject* grid_to_graph_cpy(PyObject* Py_UNUSED(self), PyObject* args,
    PyObject* kwargs)
{ 
#else
static PyObject* grid_to_graph_cpy(PyObject* self, PyObject* args,
    PyObject* kwargs)
{   (void) self; // suppress unused parameter warning
#endif

    /***  get and check inputs  ***/
    PyArrayObject *py_grid_shape;
    int connectivity = 1;
    int compute_connectivities = false;
    int graph_as_forward_star = true;
    int row_major_index = true;

    const char* keywords[] = {"", "connectivity", "compute_connectivities",
        "graph_as_forward_star", "row_major_index", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|iiii", (char**) keywords,
        &py_grid_shape, &connectivity, &compute_connectivities,
        &graph_as_forward_star, &row_major_index)){
        return NULL;
    }

    if (!PyArray_Check(py_grid_shape)){
        PyErr_SetString(PyExc_TypeError, "Grid to graph: argument "
            "'grid_shape' must be a numpy array.");
        return NULL;
    }

    if (!PyArray_ISINTEGER(py_grid_shape)){
        PyErr_SetString(PyExc_TypeError, "Grid to graph: elements in "
            "'grid_shape' must be a of integer type.");
        return NULL;
    }else if (PyArray_TYPE(py_grid_shape) == NPY_INT16 ||
              PyArray_TYPE(py_grid_shape) == NPY_UINT16){
        return grid_to_graph<uint16_t, NPY_UINT16>(py_grid_shape, connectivity,
            compute_connectivities, graph_as_forward_star, row_major_index);
    }else if (PyArray_TYPE(py_grid_shape) == NPY_INT32 ||
              PyArray_TYPE(py_grid_shape) == NPY_UINT32){
        return grid_to_graph<uint32_t, NPY_UINT32>(py_grid_shape, connectivity,
            compute_connectivities, graph_as_forward_star, row_major_index);
    }else if (PyArray_TYPE(py_grid_shape) == NPY_INT64 ||
              PyArray_TYPE(py_grid_shape) == NPY_UINT64){
        return grid_to_graph<uint64_t, NPY_UINT64>(py_grid_shape, connectivity,
            compute_connectivities, graph_as_forward_star, row_major_index);
    }else{
        PyErr_SetString(PyExc_TypeError, "Grid to graph: not "
            "implemented for the given integer type.");
        return NULL;
    }
}

static const char* grid_to_graph_doc = 
" first_edge, adj_vertices[, connectivities] = grid_to_graph(shape,\n"
"       connectivity = 1, compute_connectivities = False,\n"
"       graph_as_forward_star = True, row_major_index = True)\n"
"\n"
"or\n"
"\n"
" edges[, connectivities] = grid_to_graph(shape, connectivity = 1,\n"
"       compute_connectivities = False, graph_as_forward_star = False,\n"
"       row_major_index = True)\n"
"\n"
"Compute the graph structure defined by local connectivity on a\n"
"multidimensional grid.\n"
"\n"
"A graph with V vertices and E edges is represented either as edge list\n"
"(array of E edges given as ordered pair of vertices), or as forward star,\n"
"where edges are numeroted so that all edges originating from a same vertex\n"
"are consecutive, and represented by the following parameters:\n"
"first_edge - array of length V + 1, indicating for each vertex, the first\n"
"     edge starting from the vertex (or, if there are none, starting from\n"
"     the next vertex); the first value is always zero and the last value is\n"
"     always the total number of edges\n"
"adj_vertices - array of length E, indicating for each edge, its ending\n"
"     vertex\n"
"\n"
"NOTA: edges and vertices identifiers are integers (start at 0), whose type\n"
"is either uint16, uint32 or uint64, depending on the type of 'shape' input.\n"
"\n"
"INPUTS:\n"
" shape - numpy array of integers of length D, giving the grid size in each\n"
"         dimension\n"
" connectivity - defines the neighboring relationship;\n"
"       corresponds to the _square_ of the maximum Euclidean distance\n"
"       between two neighbors;\n"
"       if less than 4, it defines the number of coordinates allowed\n"
"       to simultaneously vary (+1 or -1) to define a neighbor; in that\n"
"       case, each level l of connectivity in dimension D adds\n"
"       binom(D, l)*2^l neighbors;\n"
"       corresponding number of neighbors for D = 2 and 3:\n"
"\n"
"       connectivity |  1   2   3 \n"
"       --------------------------\n"
"                 2D |  4   8  (8)\n"
"                 3D |  6  18  26 \n"
"\n"
"       note that a connectivity of 4 or more includes neighbors whose\n"
"       coordinates might differ by 2 or more from the coordinates of the\n"
"       considered vertex. Interestingly, in dimension 4 or more, including\n"
"       all surrounding vertices would then also include vertices from a\n"
"       \"more distant\" surround: the neighbor v + (2, 0, 0, 0) is at the\n"
"       same distance as the neighbor v + (1, 1, 1, 1).\n"
" compute_connectivities - set to True to request the connectivities (square\n"
"       Euclidean lengths) of the edges; in that case, one more output must\n"
"       be expected\n"
" graph_as_forward_star - representation of the output graph; if set to True\n"
"       (the default), forward-star as described above in 2 output arrays;\n"
"       otherwise, edge list in 1 output array\n"
" row_major_index - order in which vertices of the grid are indexed; if set\n"
"       to True, row-major order is used: indices increase first along the\n"
"       last dimension specified by 'shape' (in the usual convention in 2D,\n"
"       this corresponds to rows), and then along the second to last\n"
"       dimension, and so on up to the first dimension; this is the default\n"
"       because numpy uses row-major (C-contiguous) arrays by default.\n"
"       If set to False, indexing uses column-major order (indices increase\n"
"       first along the first dimension and so on up to the last); note that\n"
"       this is equivalent to reverting the order of the grid dimensions in\n"
"       'shape' (in 2D, this amounts to transposition).\n"
"\n"
"OUTPUTS:\n"
" first_edge, adj_vertices - if graph_as_forward_star is True, forward-star\n"
"       representation of the resulting graph, as described above\n"
" edges - if graph_as_forward_star is False, edge list representation of the\n"
"       resulting graph, integers numpy array of size E-by-2\n"
" connectivities - if requested, the connectivities (square Euclidean\n"
"       lengths) of the edges, numpy array of uint8 of length E\n"
"\n"
"Hugo Raguet 2020\n";

static PyMethodDef grid_graph_methods[] = {
    {"edge_list_to_forward_star", edge_list_to_forward_star_cpy, METH_VARARGS,
        edge_list_to_forward_star_doc},
    {"grid_to_graph", (PyCFunction) grid_to_graph_cpy,
        METH_VARARGS | METH_KEYWORDS, grid_to_graph_doc},
    {NULL, NULL, 0, NULL}
};

/* module initialization */
#if PY_MAJOR_VERSION >= 3
/* Python version 3 */
static struct PyModuleDef grid_graph_module = {
    PyModuleDef_HEAD_INIT,
    "grid_graph", /* name of module */
    /* module documentation, may be null */
    "creating and manipulating grid graph structures in edge list or\n"
    "forward star representations.",
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    grid_graph_methods, /* actual methods in the module */
    NULL, /* multi-phase initialization, may be null */
    NULL, /* traversal function, may be null */
    NULL, /* clearing function, may be null */
    NULL  /* freeing function, may be null */
};

PyMODINIT_FUNC
PyInit_grid_graph(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */

    PyObject* m;

    /* create the module */
    m = PyModule_Create(&grid_graph_module);
    if (!m){ return NULL; }

    return m;
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC
initgrid_graph(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */

    PyObject* m;

    m = Py_InitModule("grid_graph", grid_graph_methods);
    if (!m){ return; }
}

#endif
