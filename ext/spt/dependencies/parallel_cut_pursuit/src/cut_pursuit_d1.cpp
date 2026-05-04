/*=============================================================================
 * Hugo Raguet 2018, 2020, 2021, 2023
 *===========================================================================*/
#include <cmath>
#include "cut_pursuit_d1.hpp"

#define EDGE_WEIGHTS_(e) (edge_weights ? edge_weights[(e)] : homo_edge_weight)
#define D1P_METRIC_(d) (d1p_metric ? d1p_metric[(d)] : (real_t) 1.0)

#define TPL template <typename real_t, typename index_t, typename comp_t>
#define CP_D1 Cp_d1<real_t, index_t, comp_t>

using namespace std;

TPL CP_D1::Cp_d1(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices, size_t D)
    : Cp<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices, D)
{
    d1p_metric = nullptr;
    G = nullptr;
    d1p = D > 1 ? D12 : D11;
}

TPL void CP_D1::set_d1_param(const real_t* edge_weights,
    real_t homo_edge_weight, const real_t* d1p_metric, D1p d1p)
{
    set_edge_weights(edge_weights, homo_edge_weight);
    this->d1p_metric = d1p_metric;
    this->d1p = D > 1 ? d1p : D11;
}

TPL void CP_D1::set_split_param(index_t max_split_size, comp_t K,
    int split_iter_num, real_t split_damp_ratio, int split_values_init_num,
    int split_values_iter_num)
{
    if (D == 1){
        if (K < 2 || K > 3 || split_iter_num > 1 || split_damp_ratio != 1.0
            || split_values_init_num > 1 || split_values_iter_num > 1){
            cerr << "Cut-pursuit d1: for unidimensional problems, the only "
                "split parameter which can be changed is the maximum split "
                "size." << endl;
            exit(EXIT_FAILURE);
        }
    }
    Cp<real_t, index_t, comp_t>::set_split_param(max_split_size, K,
        split_iter_num, split_damp_ratio, split_values_init_num,
        split_values_iter_num);
}

TPL void CP_D1::compute_grad()
{
    for (size_t vd = 0; vd < D*V; vd++){ G[vd] = 0.0; }

    /**  differentiable d1p contribution  **/ 
    /* cannot parallelize with graph structure available here */
    for (index_t v = 0; v < V; v++){
        const real_t* rXv = rX + D*comp_assign[v];
        real_t* Gv = G + D*v;
        for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
            if (is_cut(e)){
                index_t u = adj_vertices[e];
                const real_t* rXu = rX + D*comp_assign[u];
                real_t* Gu = G + D*u; 
                if (d1p == D11){
                /* strictly speaking, in the d11 case, equality of some
                 * coordinates constitutes a source of nondifferentiability;
                 * this is actually not taken into account, favoring split */
                    for (size_t d = 0; d < D; d++){
                        real_t grad_d11 = EDGE_WEIGHTS_(e)*D1P_METRIC_(d);
                        if (rXv[d] - rXu[d] > eps){
                            Gv[d] += grad_d11;
                            Gu[d] -= grad_d11;
                        }else if (rXu[d] - rXv[d] > eps){
                            Gu[d] += grad_d11;
                            Gv[d] -= grad_d11;
                        }
                    }
                }else{
                    real_t norm_weight = 0.0;
                    for (size_t d = 0; d < D; d++){
                        norm_weight += (rXu[d] - rXv[d])*(rXu[d] - rXv[d])
                            *D1P_METRIC_(d);
                    }
                    norm_weight = EDGE_WEIGHTS_(e)/sqrt(norm_weight);
                    for (size_t d = 0; d < D; d++){
                        real_t grad_d12 = norm_weight*(rXv[d] - rXu[d])
                            *D1P_METRIC_(d);
                        Gv[d] += grad_d12;
                        Gu[d] -= grad_d12;
                    }
                }
            }
        }
    }
}

TPL index_t CP_D1::split()
{
    G = (real_t*) malloc_check(sizeof(real_t)*D*V);
    compute_grad();
    index_t activation = Cp<real_t, index_t, comp_t>::split();
    free(G);
    return activation;
}

TPL typename CP_D1::Split_info CP_D1::initialize_split_info(comp_t rv)
{
    if (D == 1){ /* assume differentiability : -1 vs. +1 */
        Split_info split_info(rv);
        real_t* sX = split_info.sX = (real_t*) malloc_check(sizeof(real_t)*2);
        sX[0] = -1.0; sX[1] = 1.0;
        split_info.first_k = 1;
        split_info.K = 2;
        /* assign all vertex to -1 */
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
             label_assign[comp_list[i]] = 0; 
        }
        return split_info;
    }else{
        return Cp<real_t, index_t, comp_t>::initialize_split_info(rv);
    }
}


TPL real_t CP_D1::vert_split_cost(const Split_info& split_info, index_t v,
    comp_t k) const
{
    const real_t* Gv = G + D*v;
    const real_t* sXk = split_info.sX + D*k;
    real_t c = 0.0;
    for (size_t d = 0; d < D; d++){ c += sXk[d]*Gv[d]; }
    return c;
}

TPL real_t CP_D1::vert_split_cost(const Split_info& split_info, index_t v,
    comp_t k, comp_t l) const
{
    if (k == l){ return 0.0; }
    const real_t* Gv = G + D*v;
    const real_t* sXk = split_info.sX + D*k;
    const real_t* sXl = split_info.sX + D*l;
    real_t c = 0.0;
    for (size_t d = 0; d < D; d++){ c += (sXk[d] - sXl[d])*Gv[d]; }
    return c;
}

TPL real_t CP_D1::edge_split_cost(const Split_info& split_info, index_t e,
        comp_t lu, comp_t lv) const
{
    if (lu == lv){ return 0.0; }
    const real_t* sXu = split_info.sX + D*lu;
    const real_t* sXv = split_info.sX + D*lv;
    real_t c = 0.0;
    if (d1p == D11){
        /* strictly speaking, in the d11 case, active edges should not be
         * directly ignored, because some neighboring coordinates can still
         * be equal, yielding nondifferentiability and thus corresponding to
         * positive split cost; however, such capacities are somewhat
         * cumbersome to compute, and more importantly max flows cannot be
         * easily computed in parallel, since the components would not be
         * independent anymore; we thus stick with the current heuristic */
        for (size_t d = 0; d < D; d++){
            c += abs(sXu[d] - sXv[d])*D1P_METRIC_(d);
        }
    }else if (d1p == D12){
        for (size_t d = 0; d < D; d++){
            c += (sXu[d] - sXv[d])*(sXu[d] - sXv[d])*D1P_METRIC_(d);
        }
        c = sqrt(c);
    }
    return EDGE_WEIGHTS_(e)*c;
}

TPL void CP_D1::project_descent_direction(Split_info& split_info, comp_t k)
    const
{
    real_t* sXk = split_info.sX + D*k;
    real_t norm = 0.0;
    for (size_t d = 0; d < D; d++){ norm += sXk[d]*sXk[d]; }
    if (norm < eps){
        for (size_t d = 0; d < D; d++){ sXk[d] = 0.0; }
    }else{
        norm = sqrt(norm);
        for (size_t d = 0; d < D; d++){ sXk[d] = sXk[d]/norm; }
    }
}

TPL void CP_D1::set_split_value(Split_info& split_info, comp_t k, index_t v)
    const
{
    const real_t* Gv = G + D*v;
    real_t* sXk = split_info.sX + D*k;
    for (size_t d = 0; d < D; d++){ sXk[d] = -Gv[d]; }
    project_descent_direction(split_info, k);
}

TPL void CP_D1::update_split_info(Split_info& split_info) const
{
    comp_t rv = split_info.rv;
    real_t* sX = split_info.sX;
    index_t* total_weights = (index_t*)
        malloc_check(sizeof(index_t)*split_info.K);
    for (comp_t k = 0; k < split_info.K; k++){
        total_weights[k] = 0.0;
        real_t* sXk = sX + D*k;
        for (size_t d = 0; d < D; d++){ sXk[d] = 0.0; }
    }
    for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
        index_t v = comp_list[i];
        comp_t k = label_assign[v];
        total_weights[k]++;
        const real_t* Gv = G + D*v;
        real_t* sXk = sX + D*k;
        for (size_t d = 0; d < D; d++){ sXk[d] -= Gv[d]; }
    }
    comp_t kk = 0; // actual number of alternatives kept
    for (comp_t k = 0; k < split_info.K; k++){
        const real_t* sXk = sX + D*k;
        real_t* sXkk = sX + D*kk;
        if (total_weights[k]){
            for (size_t d = 0; d < D; d++){
                sXkk[d] = sXk[d]/total_weights[k];
            }
            project_descent_direction(split_info, kk);
            kk++;
        } // else no vertex assigned to k, discard this alternative
    }
    split_info.K = kk;
    free(total_weights);
}

TPL uintmax_t CP_D1::split_values_complexity() const
{
    if (D == 1){ return 0; }
    else{ return Cp<real_t, index_t, comp_t>::split_values_complexity(); }
}

TPL uintmax_t CP_D1::split_complexity() const
{
    /* graph cut */
    uintmax_t complexity = D*V; // account unary split cost and final labeling
    complexity += E; // account for binary split cost capacities
    complexity += maxflow_complexity(); // graph cut
    /* K alternative labels, one is not needed when D = 1 or K = 2 */
    if (D == 1){ complexity *= K - 1; }
    else if (K > 2){ complexity *= K; } 
    complexity *= split_iter_num; // repeated
    /* all split value computations (init and updates) */
    complexity += split_values_complexity();
    return complexity*(V - saturated_vert)/V; // account saturation linearly
}

TPL index_t CP_D1::remove_balance_separations(comp_t rV_new)
{
    index_t activation = 0;

    /* separation edges must be activated if and only if the descent
     * directions at its vertices are different; on directionnally
     * differentiable problems, descent directions depends in theory only on
     * the components values; since these are the same on both sides of a
     * parallel separation, it is sometimes possible to implement
     * split_component() so that the same label assignment means the same
     * descent direction */

    /* in practice, the above in not possible for multidimensional problems */
    if (D > 1){
        return Cp<real_t, index_t, comp_t>::remove_balance_separations(rV_new);
    }

    #pragma omp parallel for schedule(static) reduction(+:activation) \
        NUM_THREADS(E*first_vertex[rV_new]/V, rV_new)
    for (comp_t rv_new = 0; rv_new < rV_new; rv_new++){
        for (index_t i = first_vertex[rv_new];
             i < first_vertex[rv_new + 1]; i++){
            index_t v = comp_list[i];
            comp_t l = label_assign[v];
            for (index_t e = first_edge[v]; e < first_edge[v + 1]; e++){
                if (is_separation(e)){
                    if (l == label_assign[adj_vertices[e]]){
                        bind(e);
                    }else{
                        cut(e);
                        activation++;
                    }
                }
            }
        }
    }

    return activation;
}

TPL bool CP_D1::is_almost_equal(comp_t ru, comp_t rv) const
{
    real_t dif = 0.0, ampu = 0.0, ampv = 0.0;
    real_t *rXu = rX + ru*D;
    real_t *rXv = rX + rv*D;
    for (size_t d = 0; d < D; d++){
        if (d1p == D11){
            dif += abs(rXu[d] - rXv[d])*D1P_METRIC_(d);
            ampu += abs(rXu[d])*D1P_METRIC_(d);
            ampv += abs(rXv[d])*D1P_METRIC_(d);
        }else if (d1p == D12){
            dif += (rXu[d] - rXv[d])*(rXu[d] - rXv[d])*D1P_METRIC_(d);
            ampu += rXu[d]*rXu[d]*D1P_METRIC_(d);
            ampv += rXv[d]*rXv[d]*D1P_METRIC_(d);
        }
    }
    real_t amp = ampu > ampv ? ampu : ampv;
    if (d1p == D12){ dif = sqrt(dif); amp = sqrt(amp); }
    if (eps > amp){ amp = eps; }
    return dif <= dif_tol*amp;
}

TPL comp_t CP_D1::compute_merge_chains()
{
    comp_t merge_count = 0;
    for (index_t re = 0; re < rE; re++){
        comp_t ru = reduced_edges_u(re);
        comp_t rv = reduced_edges_v(re);
        /* get the root of each component's chain */
        ru = get_merge_chain_root(ru);
        rv = get_merge_chain_root(rv);
        if (ru != rv && is_almost_equal(ru, rv)){
            merge_components(ru, rv);
            merge_count++;
        }
    }
    return merge_count;
}

TPL index_t CP_D1::merge()
{
    index_t deactivation = Cp<real_t, index_t, comp_t>::merge();

    /* desaturate components with value evolving more than tolerance */
    index_t desaturated_vert = 0;
    comp_t desaturated_comp = 0;
    #pragma omp parallel for schedule(static) NUM_THREADS(D*saturated_comp) \
        reduction(+:desaturated_vert, desaturated_comp)
    for (comp_t rv = 0; rv < rV; rv++){
        if (is_saturated[rv]){
            const real_t* rXv = rX + D*rv;
            const real_t* lrXv = last_rX +
                D*last_comp_assign[comp_list[first_vertex[rv]]];
            real_t dif = 0.0, amp = 0.0;
            for (size_t d = 0; d < D; d++){ 
                dif += (rXv[d] - lrXv[d])*(rXv[d] - lrXv[d]);
                amp += rXv[d]*rXv[d];
            }
            if (dif > amp*dif_tol*dif_tol){
                is_saturated[rv] = false;
                desaturated_comp++;
                desaturated_vert += first_vertex[rv + 1] - first_vertex[rv];
            }
        }
    }
    saturated_comp -= desaturated_comp;
    saturated_vert -= desaturated_vert;

    return deactivation;
}

TPL real_t CP_D1::compute_evolution() const
{
    real_t dif = 0.0, amp = 0.0;
    #pragma omp parallel for schedule(dynamic) \
        NUM_THREADS(D*(V - saturated_vert), rV) reduction(+:dif, amp)
    for (comp_t rv = 0; rv < rV; rv++){
        real_t* rXv = rX + D*rv;
        real_t amp_rv = 0.0;
        for (size_t d = 0; d < D; d++){ amp_rv += rXv[d]*rXv[d]; }
        amp += amp_rv*(first_vertex[rv + 1] - first_vertex[rv]);
        if (is_saturated[rv]){
            real_t* lrXv = last_rX +
                 D*last_comp_assign[comp_list[first_vertex[rv]]];
            real_t dif_rv = 0.0;
            for (size_t d = 0; d < D; d++){
                dif_rv += (rXv[d] - lrXv[d])*(rXv[d] - lrXv[d]);
            }
            dif += dif_rv*(first_vertex[rv + 1] - first_vertex[rv]);
        }else{
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                real_t* lrXv = last_rX + D*last_comp_assign[comp_list[i]];
                for (size_t d = 0; d < D; d++){
                    dif += (rXv[d] - lrXv[d])*(rXv[d] - lrXv[d]);
                }
            }
        }
    }
    dif = sqrt(dif);
    amp = sqrt(amp);
    return amp > eps ? dif/amp : dif/eps;
}

TPL real_t CP_D1::compute_graph_d1() const
{
    real_t tv = 0.0;
    #pragma omp parallel for schedule(static) NUM_THREADS(2*rE*D, rE) \
        reduction(+:tv)
    for (index_t re = 0; re < rE; re++){
        real_t *rXu = rX + reduced_edges_u(re)*D;
        real_t *rXv = rX + reduced_edges_v(re)*D;
        real_t dif = 0.0;
        for (size_t d = 0; d < D; d++){
            if (d1p == D11){
                dif += abs(rXu[d] - rXv[d])*D1P_METRIC_(d);
            }else if (d1p == D12){
                dif += (rXu[d] - rXv[d])*(rXu[d] - rXv[d])*D1P_METRIC_(d);
            }
        }
        if (d1p == D12){ dif = sqrt(dif); }
        tv += reduced_edge_weights[re]*dif;
    }
    return tv;
}

/**  instantiate for compilation  **/
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Cp_d1<float, int32_t, int16_t>;
template class Cp_d1<double, int32_t, int16_t>;
template class Cp_d1<float, int32_t, int32_t>;
template class Cp_d1<double, int32_t, int32_t>;
#else
template class Cp_d1<float, uint32_t, uint16_t>;
template class Cp_d1<double, uint32_t, uint16_t>;
template class Cp_d1<float, uint32_t, uint32_t>;
template class Cp_d1<double, uint32_t, uint32_t>;
#endif
