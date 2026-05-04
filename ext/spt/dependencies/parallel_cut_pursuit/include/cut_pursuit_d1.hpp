/*=============================================================================
 * Derived class for cut-pursuit algorithm with d1 (total variation) 
 * penalization on directionnaly differentiable problems:
 *
 * Minimize functional over a graph G = (V, E)
 *
 *        F : R^{D-by-V} -> R
 *                     x -> f(x) + ||x||_d1p
 *
 * involving the graph total variation penalisation
 *
 *      ||x||_d1p = sum_{uv in E} w_uv ||x_u - x_v||_{M,p} ,
 *
 * where M in R^{D-by-D} is a diagonal metric and p can be 1 or 2,
 *  ||x_v||_{M,1} = sum_d m_d |x_v|  (weighted l1-norm) or
 *  ||x_v||_{M,2} = sqrt(sum_d m_d x_v^2)  (weighted l2-norm)
 *
 * and where f is an extended directionaly differentiable functional, using
 * cut-pursuit approach.
 *
 * Parallel implementation with OpenMP API.
 *
 * H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing Nonsmooth
 * Functionals with Graph Total Variation, International Conference on Machine
 * Learning, PMLR, 2018, 80, 4244-4253
 *
 * Hugo Raguet 2018, 2023
 *===========================================================================*/
#pragma once
#include "cut_pursuit.hpp"

/* real_t is the real numeric type, used for the base field and for the
 * objective functional computation;
 * index_t must be able to represent the numbers of vertices and of
 * (undirected) edges in the main graph; comp_t must be able to represent one
 * plus the number of constant connected components in the reduced graph */
template <typename real_t, typename index_t, typename comp_t>
class Cp_d1 : public Cp<real_t, index_t, comp_t>
{
public:
    Cp_d1(index_t V, index_t E, const index_t* first_edge, 
        const index_t* adj_vertices, size_t D = 1);

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (forward-star graph structure given at construction, 
     * edge weights, etc.); IT DOES FREE THE REST (components assignment 
     * and reduced problem elements, etc.), but this can be prevented by
     * getting the corresponding pointer member and setting it to null
     * beforehand */

    /* for multidimensional data, type of graph total variation, which is
     * nothing but the sum of lp norms of finite differences over the edges:
     * d1,1 is the sum of l1 norms;
     * d1,2 is the sum of l2 norms */
    enum D1p {D11, D12};

    /* edge_weights are the w_uv above, if set to null, homogeneously equal to
     * homo_edge_weight; d1p_metric is the weighting matrice M above */
    using Cp<real_t, index_t, comp_t>::set_edge_weights;
    void set_d1_param(const real_t* edge_weights = nullptr,
        real_t homo_edge_weight = 1.0, const real_t* d1p_metric = nullptr,
        D1p d1p = D12);

    /* overload for forcing some values when D = 1 */
    void set_split_param(index_t max_split_size, comp_t K = 2,
        int split_iter_num = 1, real_t split_damp_ratio = 1.0,
        int split_values_init_num = 1, int split_values_iter_num = 1);

protected:
    /* for multidimensional data, weights the coordinates in the lp norms;
     * all weights must be strictly positive, and it is advised to normalize
     * the weights so that the first value is unity */
    const real_t* d1p_metric;

    /** split  **/

    real_t* G; // store gradient of smooth part

    using typename Cp<real_t, index_t, comp_t>::Split_info;

    /* override for product with the gradient */
    real_t vert_split_cost(const Split_info& split_info, index_t v, comp_t k)
        const override;
    /* factor product with difference */
    real_t vert_split_cost(const Split_info& split_info, index_t v, comp_t k,
        comp_t l) const override;

    /* override for computing relative l2 norm evolution */
    real_t compute_evolution() const override;

    /* compute graph total variation; use reduced edges and reduced weights */
    real_t compute_graph_d1() const;

    D1p d1p; // see public enum declaration

    /**  type resolution for base template class members
     * https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
     **/
    using Cp<real_t, index_t, comp_t>::last_rX;
    using Cp<real_t, index_t, comp_t>::last_comp_assign;
    using Cp<real_t, index_t, comp_t>::rX;
    using Cp<real_t, index_t, comp_t>::eps;
    using Cp<real_t, index_t, comp_t>::dif_tol;
    using Cp<real_t, index_t, comp_t>::split_iter_num;
    using Cp<real_t, index_t, comp_t>::K;
    using Cp<real_t, index_t, comp_t>::V;
    using Cp<real_t, index_t, comp_t>::E;
    using Cp<real_t, index_t, comp_t>::D;
    using Cp<real_t, index_t, comp_t>::first_edge;
    using Cp<real_t, index_t, comp_t>::adj_vertices; 
    using Cp<real_t, index_t, comp_t>::rV;
    using Cp<real_t, index_t, comp_t>::rE;
    using Cp<real_t, index_t, comp_t>::first_vertex;
    using Cp<real_t, index_t, comp_t>::comp_list;
    using Cp<real_t, index_t, comp_t>::comp_assign;
    using Cp<real_t, index_t, comp_t>::label_assign;
    using Cp<real_t, index_t, comp_t>::is_separation;
    using Cp<real_t, index_t, comp_t>::reduced_edge_weights;
    using Cp<real_t, index_t, comp_t>::malloc_check;
    using Cp<real_t, index_t, comp_t>::realloc_check;
    using Cp<real_t, index_t, comp_t>::is_saturated;
    using Cp<real_t, index_t, comp_t>::saturated_comp;
    using Cp<real_t, index_t, comp_t>::saturated_vert;
    using Cp<real_t, index_t, comp_t>::homo_edge_weight;
    using Cp<real_t, index_t, comp_t>::edge_weights;

private:
    /**  split  **/

    /* override for computing the gradient */
    index_t split() override;

protected:
    /* compute gradient of smooth part;
     * base version compute gradient of smooth part of the d1 term */
    virtual void compute_grad();

private:
    /* override for D = 1 : set {-1, +1} and omit competing value k = 0 for
     * binary cut; assumes differentiability */
    Split_info initialize_split_info(comp_t rv) override;
    /* override the setting and update of split values for using opposite of
     * the gradient; add a projection on the unit ball, with virtualization for
     * possible further specializations; in particular, if no descent direction
     * can be found, set to zero */
    void set_split_value(Split_info& split_info, comp_t k, index_t v) const
        override;
protected:
    virtual void project_descent_direction(Split_info& split_info, comp_t k)
        const;
private:
    void update_split_info(Split_info& split_info) const override;

    /* override for counting no k-means when D = 1 */
    uintmax_t split_values_complexity() const override;
    /* override for ommiting one graph cut when D = 1 */
    uintmax_t split_complexity() const override;

    /* override for computing D11 or D12 norm */
    real_t edge_split_cost(const Split_info& split_info, index_t e, comp_t lu,
        comp_t lv) const override;

    /* remove or activate separating edges used for balancing split workload,
     * see header `cut_pursuit.hpp`;
     * separation edges should be activated if and only if the descent
     * directions at its vertices are different; on directionnally
     * differentiable problems, descent directions depends in theory only on
     * the components values; since these are the same on both sides of a
     * parallel separation, it is sometimes possible to implement the method
     * split_component() so that the same label assignment means the same
     * descent direction;
     * we provide here a convenient implementation for this case, but if 
     * split_component() provided by a derived class cannot guarantee the above
     * it is advised to override this in derived class by calling the base
     * method CP::remove_balance_separations */
    index_t remove_balance_separations(comp_t rV_new) override;

    /**  merge  **/

    /* test if two components are sufficiently close to merge */
    bool is_almost_equal(comp_t ru, comp_t rv) const;

    /* compute the merge chains and return the number of effective merges */
    comp_t compute_merge_chains() override;

    /* override for checking value evolution of saturated components;
     * precision can be increased by decreasing dif_tol if necessary */
    bool monitor_evolution() const override { return true; }
    index_t merge() override;

    /**  type resolution for base template class members  **/
    using Cp<real_t, index_t, comp_t>::maxflow_complexity;
    using Cp<real_t, index_t, comp_t>::cut;
    using Cp<real_t, index_t, comp_t>::is_cut;
    using Cp<real_t, index_t, comp_t>::bind;
    using Cp<real_t, index_t, comp_t>::merge_components;
    using Cp<real_t, index_t, comp_t>::get_merge_chain_root;
    using Cp<real_t, index_t, comp_t>::reduced_edges_u;
    using Cp<real_t, index_t, comp_t>::reduced_edges_v;
};
