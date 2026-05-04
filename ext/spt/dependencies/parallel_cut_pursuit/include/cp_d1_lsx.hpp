/*===========================================================================
 * Derived class for cut-pursuit algorithm with d1 (total variation) 
 * penalization, with a separable loss term and simplex constraints:
 *
 * minimize functional over a graph G = (V, E)
 *
 *        F(x) = f(x) + ||x||_d1 + i_{simplex}(x)
 *
 * where for each vertex, x_v is a D-dimensional vector,
 *       f(x) = sum_{v in V} f_v(x_v) is a separable data-fidelity loss
 *       ||x||_d1 = sum_{uv in E} w_d1_uv (sum_d w_d1_d |x_ud - x_vd|),
 * and i_{simplex} is the standard D-simplex constraint over each vertex,
 *     i_{simplex} = 0 for all v, (for all d, x_vd >= 0) and sum_d x_vd = 1,
 *                 = infinity otherwise;
 *
 * using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
 * splitting algorithm.
 *
 * Parallel implementation with OpenMP API.
 *
 * H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing 
 * Nonsmooth Functionals with Graph Total Variation, International Conference
 * on Machine Learning, PMLR, 2018, 80, 4244-4253
 *
 * H. Raguet, A Note on the Forward-Douglas-Rachford Splitting for Monotone 
 * Inclusion and Convex Optimization, Optimization Letters, 2018, 1-24
 *
 * Hugo Raguet 2018, 2021, 2022, 2023
 *=========================================================================*/
#pragma once
#include "cut_pursuit_d1.hpp"

/* real_t is the real numeric type, used for the base field and for the
 * objective functional computation;
 * index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph, as well as the dimension D */
template <typename real_t, typename index_t, typename comp_t>
class Cp_d1_lsx : public Cp_d1<real_t, index_t, comp_t>
{
private:
    /**  type resolution for base template class members
     * https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
     **/
    using Cp<real_t, index_t, comp_t>::dif_tol;
    using typename Cp_d1<real_t, index_t, comp_t>::D1p;
    using Cp_d1<real_t, index_t, comp_t>::d1p;
    using Cp_d1<real_t, index_t, comp_t>::D11;

public:
    /**  constructor, destructor  **/

    /* only store graph structure and assign Y, D */
    Cp_d1_lsx(index_t V, index_t E, const index_t* first_edge,
        const index_t* adj_vertices, size_t D, const real_t* Y);

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (forward-star graph structure given at construction, 
     * monitoring arrays, observation arrays); IT DOES FREE THE REST 
     * (components assignment and reduced problem elements, etc.), but this can
     * be prevented by getting the corresponding pointer member and setting it
     * to null beforehand */

    /**  methods for manipulating parameters  **/

    /* overload the base class Cp_d1 method to enforce d1p value to D11 */
    using Cp_d1<real_t, index_t, comp_t>::set_d1_param;
    void set_d1_param(const real_t* edge_weights = nullptr,
        real_t homo_edge_weight = 1.0, const real_t* d11_metric = nullptr,
        D1p d1p = D11);

    /* specific losses */
    static real_t linear_loss() { return 0.0; }
    static real_t quadratic_loss() { return 1.0; }

    /* Y is changed only if the corresponding argument is not null */
    void set_loss(real_t loss, const real_t* Y = nullptr,
        const real_t* loss_weights = nullptr);

    /* overload for changing only loss weights */
    void set_loss(const real_t* loss_weights)
        { set_loss(loss, nullptr, loss_weights); }

    void set_pfdr_param(real_t rho, real_t cond_min, real_t dif_rcd,
        int it_max, real_t dif_tol);

    /* overload for default dif_tol parameter */
    void set_pfdr_param(real_t rho = 1.0, real_t cond_min = 1e-2,
        real_t dif_rcd = 0.0, int it_max = 1e4)
    { set_pfdr_param(rho, cond_min, dif_rcd, it_max, 1e-3*dif_tol); }

private:
    /**  separable loss term  **/

    /* observations, D-by-V array, column major format;
     * must lie on the simplex */
    const real_t* Y; 

    /* 0 for linear (function linear_loss())
     *     f(x) = - <x, y>_w ,
     * with <x, y>_w = sum_{v,d} w_v y_{v,d} x_{v,d} ;
     *
     * 1 for quadratic (function quadratic_loss())
     *     f(x) = 1/2 ||y - x||_{l2,w}^2 ,
     * with ||y - x||_{l2,w}^2 = sum_{v,d} w_v (y_{v,d} - x_{v,d})^2 ;
     *
     * 0 < loss < 1 for smoothed Kullback-Leibler divergence (cross-entropy)
     *     f(x) = sum_v w_v KLs(x_v, y_v),
     * with KLs(y_v, x_v) = KL(s u + (1 - s) y_v ,  s u + (1 - s) x_v), where
     *     KL is the regular Kullback-Leibler divergence,
     *     u is the uniform discrete distribution over {1,...,D}, and
     *     s = loss is the smoothing parameter ;
     * it yields
     *     KLs(y_v, x_v) = - H(s u + (1 - s) y_v)
     *         - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) x_{v,d}) ,
     * where H is the entropy, that is H(s u + (1 - s) y_v)
     *       = - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) y_{v,d}) ;
     * note that the choosen order of the arguments in the Kullback-Leibler
     * does not favor the entropy of x (H(s u + (1 - s) y_v) is a constant),
     * hence this loss is actually equivalent to cross-entropy */
    real_t loss;

    /* weights on vertices; array of length V, or null for no weights */
    const real_t *loss_weights;

    /**  reduced problem  **/
    real_t pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol;
    int pfdr_it, pfdr_it_max;

    /**  cut-pursuit steps  **/

    /* compute reduced values */
    void solve_reduced_problem() override;

    /**  split  **/

    void compute_grad() override;

    using typename Cp<real_t, index_t, comp_t>::Split_info;

    /* override for taking into account nondifferentiable contributions */
    real_t vert_split_cost(const Split_info& split_info, index_t v, comp_t k)
        const override;
    real_t vert_split_cost(const Split_info& split_info, index_t v, comp_t k,
        comp_t l) const override;

    /* override for direction within the simplex */
    void project_descent_direction(Split_info& split_info, comp_t k) const
        override;

    /**  merge and updage  **/

    /* override for checking evolution of saturated components in l1 norm;
     * precision can be increased by decreasing dif_tol if necessary */
    index_t merge() override;

    /* override for relative iterate evolution in l1 norm */
    real_t compute_evolution() const override;

    real_t compute_objective() const override;

    /**  type resolution for base template class members
     * https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
     **/
    using Cp_d1<real_t, index_t, comp_t>::G;
    using Cp_d1<real_t, index_t, comp_t>::compute_graph_d1;
    using Cp_d1<real_t, index_t, comp_t>::d1p_metric;
    using Cp<real_t, index_t, comp_t>::split_iter_num;
    using Cp<real_t, index_t, comp_t>::split_damp_ratio;
    using Cp<real_t, index_t, comp_t>::split_values_init_num;
    using Cp<real_t, index_t, comp_t>::split_values_iter_num;
    using Cp<real_t, index_t, comp_t>::K;
    using Cp<real_t, index_t, comp_t>::rX;
    using Cp<real_t, index_t, comp_t>::last_rX;
    using Cp<real_t, index_t, comp_t>::saturated_comp;
    using Cp<real_t, index_t, comp_t>::saturated_vert;
    using Cp<real_t, index_t, comp_t>::is_saturated;
    using Cp<real_t, index_t, comp_t>::last_comp_assign;
    using Cp<real_t, index_t, comp_t>::eps;
    using Cp<real_t, index_t, comp_t>::V;
    using Cp<real_t, index_t, comp_t>::E;
    using Cp<real_t, index_t, comp_t>::D;
    using Cp<real_t, index_t, comp_t>::first_edge;
    using Cp<real_t, index_t, comp_t>::adj_vertices; 
    using Cp<real_t, index_t, comp_t>::rV;
    using Cp<real_t, index_t, comp_t>::rE;
    using Cp<real_t, index_t, comp_t>::comp_assign;
    using Cp<real_t, index_t, comp_t>::comp_list;
    using Cp<real_t, index_t, comp_t>::first_vertex;
    using Cp<real_t, index_t, comp_t>::reduced_edges;
    using Cp<real_t, index_t, comp_t>::reduced_edge_weights;
    using Cp<real_t, index_t, comp_t>::verbose;
    using Cp<real_t, index_t, comp_t>::malloc_check;
    using Cp<real_t, index_t, comp_t>::real_inf;
};
