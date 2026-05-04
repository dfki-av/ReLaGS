/*=============================================================================
 * Hugo Raguet 2019
 *===========================================================================*/
#include <cstdint>
#include "grid_graph.hpp"
#include "omp_num_threads.hpp"

template <typename vertex_t, typename conn_t>
size_t num_edges_grid_graph(size_t D, vertex_t* shape, conn_t connectivity)
{
    if (D == 0 || connectivity == 0){ return 0; }
    size_t num_edges;

    /***  a grid in dim D is a stack of shape[D - 1] grids in dim D - 1  ***/

    /**  edges within each grid in dimension D - 1  **/
    num_edges = shape[D - 1]*num_edges_grid_graph(D - 1, shape, connectivity);

    /**  edges connecting two different grids of dimension D - 1;
     **  shift is the distance between the vertices along dimension D  **/
    conn_t shift;
    conn_t shift2;

    /* edges parallel to direction D */
    vertex_t num_vertices_D_1 = 1;
    for (size_t d = 0; d < D - 1; d++){ num_vertices_D_1 *= shape[d]; }
    shift = 1;
    shift2 = 1;
    while (shift2 <= connectivity && shift < shape[D - 1]){
        num_edges += (shape[D - 1] - shift)*num_vertices_D_1;
        shift++;
        shift2 = shift*shift;
    }

    /* diagonal edges, two possible directions for the shift */
    shift = 1;
    shift2 = 1;
    while (shift2 < connectivity && shift < shape[D - 1]){
        num_edges += 2*(shape[D - 1] - shift)*num_edges_grid_graph<vertex_t,
            conn_t>(D - 1, shape, connectivity - shift2);
        shift++;
        shift2 = shift*shift;
    }

    return num_edges;
}

template <typename vertex_t, typename conn_t>
void edge_list_grid_graph(size_t D, vertex_t* shape, conn_t connectivity,
    vertex_t* edges, conn_t* connectivities, vertex_t offset_u,
    vertex_t offset_v, conn_t recursive_connectivity, bool recursive_call)
{
    if (D == 0 || connectivity == 0){ return; }

    /***  a grid in dim D is a stack of shape[D - 1] grids in dim D - 1  ***/

    /**  edges within each grid in dimension D - 1  **/

    /* number of vertices and edges involved in each grid in dimension D - 1 */
    vertex_t num_vertices_D_1 = 1;
    for (size_t d = 0; d < D - 1; d++){ num_vertices_D_1 *= shape[d]; }
    size_t num_edges_D_1 = num_edges_grid_graph(D - 1, shape, connectivity);

    /* spawn parallel threads only at recursion level 0 */
    int num_thrds = recursive_call ?
        1 : compute_num_threads(num_edges_D_1*shape[D - 1], shape[D - 1]);

    #pragma omp parallel for schedule(static) num_threads(num_thrds)
    /* unsigned loop counter is allowed since OpenMP 3.0 (2008)
     * but MSVC compiler still does not support it as of 2020 */
    for (long i = 0; i < (long) shape[D - 1]; i++){
        vertex_t* edges_i = edges + num_edges_D_1*2*i;
        conn_t* conn_i = connectivities;
        if (conn_i){ conn_i += i*num_edges_D_1; }
        vertex_t offset_u_i = offset_u + i*num_vertices_D_1;
        vertex_t offset_v_i = offset_v + i*num_vertices_D_1;
        edge_list_grid_graph(D - 1, shape, connectivity, edges_i, conn_i,
            offset_u_i, offset_v_i, recursive_connectivity, true);
    }

    edges += num_edges_D_1*2*shape[D - 1];
    if (connectivities){ connectivities += shape[D - 1]*num_edges_D_1; }

    /**  edges connecting two different grids of dimension D - 1;
     **  shift is the distance between the vertices along dimension D  **/

    conn_t shift;
    conn_t shift2;

    /* edges parallel to direction D */
    shift = 1;
    shift2 = 1;
    while (shift2 <= connectivity && shift < shape[D - 1]){
        conn_t rec_conn = recursive_connectivity + shift2;

        /* spawn parallel threads only at recursion level 0 */
        num_thrds = recursive_call ?
            1 : compute_num_threads(num_vertices_D_1*(shape[D - 1] - shift),
                shape[D - 1] - shift);

        #pragma omp parallel for schedule(static) num_threads(num_thrds)
        /* unsigned loop counter is allowed since OpenMP 3.0 (2008)
         * but MSVC compiler still does not support it as of 2020 */
        for (long i = 0; i < (long) shape[D - 1] - shift; i++){
            vertex_t* edges_i = edges + num_vertices_D_1*2*i;
            conn_t* conn_i = connectivities;
            if (conn_i){ conn_i += i*num_vertices_D_1; }
            vertex_t offset_u_i = offset_u + i*num_vertices_D_1;
            vertex_t offset_v_i = offset_v + (i + shift)*num_vertices_D_1;
            for (vertex_t v = 0; v < num_vertices_D_1; v++){
                edges_i[(size_t) 2*v] = offset_u_i + v;
                edges_i[(size_t) 2*v + 1] = offset_v_i + v;
                if (conn_i){ conn_i[v] = rec_conn; }
            }
        }

        edges += num_vertices_D_1*2*(shape[D - 1] - shift);
        if (connectivities){
            connectivities += (shape[D - 1] - shift)*num_vertices_D_1;
        }
        shift++;
        shift2 = shift*shift;
    }

    /* diagonal edges, two possible directions for the shift */
    shift = 1;
    shift2 = 1;
    while (shift2 < connectivity && shift < shape[D - 1]){
        conn_t rem_conn = connectivity - shift2; // remaining
        conn_t rec_conn = recursive_connectivity + shift2; // current
        size_t num_edges_rem = num_edges_grid_graph(D - 1, shape, rem_conn);

        /* spawn parallel threads only at recursion level 0 */
        num_thrds = recursive_call ?
            1 : compute_num_threads(2*num_edges_rem*(shape[D - 1] - shift),
                shape[D - 1] - shift);

        #pragma omp parallel for schedule(static) num_threads(num_thrds)
        /* unsigned loop counter is allowed since OpenMP 3.0 (2008)
         * but MSVC compiler still does not support it as of 2020 */
        for (long i = 0; i < (long) shape[D - 1] - shift; i++){
            /* positive shift */
            vertex_t* edges_i = edges + num_edges_rem*4*i;
            conn_t* conn_i = connectivities;
            if (conn_i){ conn_i += i*num_edges_rem*2; }
            vertex_t offset_u_i = offset_u + i*num_vertices_D_1;
            vertex_t offset_v_i = offset_v + (i + shift)*num_vertices_D_1;
            edge_list_grid_graph(D - 1, shape, rem_conn, edges_i, conn_i,
                offset_u_i, offset_v_i, rec_conn, true);

            /* negative shift */
            edges_i += num_edges_rem*2;
            if (conn_i){ conn_i += num_edges_rem; }
            offset_u_i = offset_u + (i + shift)*num_vertices_D_1;
            offset_v_i = offset_v + i*num_vertices_D_1;
            edge_list_grid_graph(D - 1, shape, rem_conn, edges_i, conn_i,
                offset_u_i, offset_v_i, rec_conn, true);
        }

        edges += num_edges_rem*4*(shape[D - 1] - shift);
        if (connectivities){
            connectivities += num_edges_rem*2*(shape[D - 1] - shift);
        }
        shift++;
        shift2 = shift*shift;
    }
}

/**  instantiate for compilation  **/
#define INSTANCE(vertex_t, conn_t) \
    template size_t num_edges_grid_graph<vertex_t, conn_t> \
        (size_t, vertex_t*, conn_t); \
    template void edge_list_grid_graph<vertex_t, conn_t> \
        (size_t, vertex_t*, conn_t, vertex_t*, conn_t*, vertex_t, vertex_t, \
            conn_t, bool);

INSTANCE(uint16_t, uint8_t)
INSTANCE(uint32_t, uint8_t)
INSTANCE(uint64_t, uint8_t)
