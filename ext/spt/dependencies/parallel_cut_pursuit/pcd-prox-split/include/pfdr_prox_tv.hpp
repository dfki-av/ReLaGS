/*=============================================================================
 * Derived class for preconditioned forward-Douglas–Rachford algorithm for the
 * proximal operator of the graph total variation, that is square quadratic
 * difference with penalization norm over finite differences:
 *
 * minimize functional F defined over a graph G = (V, E)
 *
 * F: R^{D-by-V} -> R
 *        x      -> 1/2 ||y - x||_Ms^2 + ||x||_d1p
 *
 * where y in R^{D-by-V}, Ms is a diagonal metric in R^{VxD-by-VxD} so that
 *
 *      ||y - x||_Ms^2 = sum_{v in V} sum_d ms_v_d (y_v - x_v)^2 ,
 *
 * and
 * 
 *      ||x||_d1p = sum_{uv in E} w_uv ||x_u - x_v||_{Md,p} ,
 *
 * where Md is a diagonal metric and p can be 1 or 2,
 *  ||x_v||_{M,1} = sum_d md_d |x_v|  (weighted l1-norm) or
 *  ||x_v||_{M,2} = sqrt(sum_d md_d x_v^2)  (weighted l2-norm).
 *
 * Parallel implementation with OpenMP API.
 *
 * H. Raguet and L. Landrieu, Preconditioning of a Generalized Forward-Backward
 * Splitting and Application to Optimization on Graphs, SIAM Journal on Imaging
 * Sciences, 2015, 8, 2706-2739
 *
 * Hugo Raguet 2023
 *===========================================================================*/
#pragma once
#include "pfdr_graph_d1.hpp"

/* vertex_t is an integer type able to represent the number of vertices */
template <typename real_t, typename vertex_t>
class Pfdr_prox_tv : public Pfdr_d1<real_t, vertex_t>
{
public:

    /**  type resolution for base template class members
     * https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
     **/
    using typename Pfdr_d1<real_t, vertex_t>::index_t;
    using typename Pfdr_d1<real_t, vertex_t>::D1p;
    using Pfdr_d1<real_t, vertex_t>::D11;
    using Pfdr_d1<real_t, vertex_t>::D12;
    /* reuse the Conshape type (see pcd_fwd_doug_rach.hpp) for the shape of
     * the metric l22_metric in the quadratic part (see member declarations);
     * NOTA: SCALAR is actually used for no metric */
    using typename Pfdr<real_t, vertex_t>::Condshape;
    using Pfdr<real_t, vertex_t>::SCALAR;
    using Pfdr<real_t, vertex_t>::MONODIM;
    using Pfdr<real_t, vertex_t>::MULTIDIM;

    /**  constructor, destructor  **/

    /* quadratic part defined by Y, metric_shape, l22_metric;
     * total variation defined by edges, edge_weights, d1p, d1p_metric;
     * see member declarations here and base class Pfdr_d1 for details;
     * d1p weights w_uv can be set from base class Pfdr_d1 method
     * set_edge_weights(); however, d1p_metric md_d above must be known at
     * construction */
    Pfdr_prox_tv(vertex_t V, index_t E, const vertex_t* edges, const real_t* Y,
        index_t D, D1p d1p = D12, const real_t* d1p_metric = nullptr,
        Condshape metric_shape = SCALAR, const real_t* l22_metric = nullptr);

    /* delegation for monodimensional setting */
    Pfdr_prox_tv(vertex_t V, index_t E, const vertex_t* edges, const real_t* Y)
        : Pfdr_prox_tv(V, E, edges, Y, 1, D11){}

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (adjacency graph structure given at construction, 
     * monitoring arrays, matrix and observation arrays); it does free the rest
     * (iterate, auxiliary variables etc.), but this can be prevented by
     * copying the corresponding pointer member and set it to null before
     * deleting */

private:
    /**  quadratic problem  **/

    const real_t* Y; /* observations, array of length V */
    const Condshape l22_metric_shape;
    const real_t* l22_metric; /* diagonal metric on squared l2-norm;
        * null pointer if l22_metric_shape is SCALAR (no metric)
        * array or length V if l22_metric_shape is MONODIM
        * D-by-V array, column major format, if l22_metric_shape is MULTIDIM
        * all weights must be positive */

    /**  specialization of base virtual methods  **/

    /* approximate diagonal hessian of quadratic functional */
    void compute_hess_f() override;

    /* compute the gradient of the quadratic functional in Pfdr::Ga_grad_f */
    void compute_Ga_grad_f() override;

    /* quadratic functional */
    real_t compute_f() const override; 

    /**  type resolution for base template class members
     * https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
     **/
    using Pfdr_d1<real_t, vertex_t>::V;
    using Pfdr_d1<real_t, vertex_t>::E;
    using Pfdr<real_t, vertex_t>::set_lipschitz_param;
    using Pfdr<real_t, vertex_t>::Ga_grad_f;
    using Pfdr<real_t, vertex_t>::Ga;
    using Pfdr<real_t, vertex_t>::ga;
    using Pfdr<real_t, vertex_t>::gashape;
    using Pfdr<real_t, vertex_t>::D;
    using Pcd_prox<real_t>::X;
    using Pcd_prox<real_t>::last_X;
    using Pcd_prox<real_t>::cond_min;
    using Pcd_prox<real_t>::dif_tol;
    using Pcd_prox<real_t>::dif_rcd;
    using Pcd_prox<real_t>::iterate_evolution;
    using Pcd_prox<real_t>::eps;
    using Pcd_prox<real_t>::malloc_check;
};
