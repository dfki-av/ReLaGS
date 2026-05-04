/*=============================================================================
 * Hugo Raguet 2018, 2020, 2021, 2023
 *===========================================================================*/
#include <cmath>
#include <algorithm>
#include "omp_num_threads.hpp"
#include "cp_d1_lsx.hpp"
#include "pfdr_d1_lsx.hpp"

#define LOSS_WEIGHTS_(v) (loss_weights ? loss_weights[(v)] : (real_t) 1.0)

#define TPL template <typename real_t, typename index_t, typename comp_t>
#define CP_D1_LSX Cp_d1_lsx<real_t, index_t, comp_t>

using namespace std;

TPL CP_D1_LSX::Cp_d1_lsx(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices, size_t D, const real_t* Y)
    : Cp_d1<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices, D),
      Y(Y)
{
    if (numeric_limits<comp_t>::max() < D){
        cerr << "Cut-pursuit d1 loss simplex: comp_t must be able to represent"
            "the dimension D (" << D << ")." << endl;
        exit(EXIT_FAILURE);
    }

    loss = linear_loss();
    loss_weights = nullptr;

    K = 2;
    split_iter_num = 1;
    split_damp_ratio = 1.0;
    split_values_init_num = 2;
    split_values_iter_num = 2;

    pfdr_rho = 1.0; pfdr_cond_min = 1e-2; pfdr_dif_rcd = 0.0;
    pfdr_dif_tol = 1e-2*dif_tol; pfdr_it = pfdr_it_max = 1e4;

    d1p = D11;
}

TPL void CP_D1_LSX::set_d1_param(const real_t* edge_weights,
    real_t homo_edge_weight, const real_t* d11_metric, D1p d1p)
{
    Cp_d1<real_t, index_t, comp_t>::set_d1_param(edge_weights,
        homo_edge_weight, d11_metric, D11);
}

TPL void CP_D1_LSX::set_loss(real_t loss, const real_t* Y,
    const real_t* loss_weights)
{
    if (loss < 0.0 || loss > 1.0){
        cerr << "Cut-pursuit d1 loss simplex: loss parameter should be between"
            " 0 and 1 (" << loss << " given)." << endl;
        exit(EXIT_FAILURE);
    }
    this->loss = loss;
    if (Y){ this->Y = Y; }
    this->loss_weights = loss_weights; 
}

TPL void CP_D1_LSX::set_pfdr_param(real_t rho, real_t cond_min, real_t dif_rcd,
    int it_max, real_t dif_tol)
{
    this->pfdr_rho = rho;
    this->pfdr_cond_min = cond_min;
    this->pfdr_dif_rcd = dif_rcd;
    this->pfdr_it_max = it_max;
    this->pfdr_dif_tol = dif_tol;
}

TPL void CP_D1_LSX::solve_reduced_problem()
{
    if (rV == 1){ /**  single connected component  **/

        #pragma omp parallel for schedule(static) NUM_THREADS(D*V, D)
        /* unsigned loop counter is allowed since OpenMP 3.0 (2008)
         * but MSVC compiler still does not support it as of 2020;
         * comp_t has been checked to be able to represent D anyway */
        for (comp_t d = 0; d < (comp_t) D; d++){
            rX[d] = 0.0;
            size_t vd = d;
            for (index_t v = 0; v < V; v++){
                rX[d] += LOSS_WEIGHTS_(v)*Y[vd];
                vd += D;
            }
        }

        if (loss == linear_loss()){ /* optimum at simplex corner */
            size_t idx = 0;
            real_t max = rX[idx];
            for (size_t d = 1; d < D; d++){
                if (rX[d] > max){ max = rX[idx = d]; }
            }
            for (size_t d = 0; d < D; d++){ rX[d] = d == idx ? 1.0 : 0.0; }
        }else{ /* optimum at barycenter */
            real_t total_weight = 0.0;
            #pragma omp parallel for schedule(static) NUM_THREADS(V) \
                reduction(+:total_weight)
            for (index_t v = 0; v < V; v++){
                total_weight += LOSS_WEIGHTS_(v);
            }
            for (size_t d = 0; d < D; d++){ rX[d] /= total_weight; }
        }

    }else{ /**  preconditioned forward-Douglas-Rachford  **/

        /* compute reduced observation and weights */
        real_t* rY = (real_t*) malloc_check(sizeof(real_t)*D*rV);
        real_t* reduced_loss_weights =
            (real_t*) malloc_check(sizeof(real_t)*rV);
        #pragma omp parallel for schedule(dynamic) NUM_THREADS(V, rV)
        for (comp_t rv = 0; rv < rV; rv++){
            real_t *rYv = rY + rv*D;
            for (size_t d = 0; d < D; d++){ rYv[d] = 0.0; }
            reduced_loss_weights[rv] = 0.0;
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                index_t v = comp_list[i];
                const real_t *Yv = Y + v*D;
                for (size_t d = 0; d < D; d++){
                    rYv[d] += LOSS_WEIGHTS_(v)*Yv[d];
                }
                reduced_loss_weights[rv] += LOSS_WEIGHTS_(v);
            }
            for (size_t d = 0; d < D; d++){
                rYv[d] /= reduced_loss_weights[rv];
            }
        }

        Pfdr_d1_lsx<real_t, comp_t> *pfdr =
            new Pfdr_d1_lsx<real_t, comp_t>
                (rV, rE, reduced_edges, loss, D, rY, d1p_metric);

        pfdr->set_edge_weights(reduced_edge_weights);
        pfdr->set_loss(reduced_loss_weights);
        pfdr->set_conditioning_param(pfdr_cond_min, pfdr_dif_rcd);
        pfdr->set_relaxation(pfdr_rho);
        pfdr->set_algo_param(pfdr_dif_tol, sqrt(pfdr_it_max),
            pfdr_it_max, verbose);
        pfdr->set_iterate(rX);
        pfdr->initialize_iterate();

        pfdr_it = pfdr->precond_proximal_splitting();

        pfdr->set_iterate(nullptr); // prevent rX to be free()'d at deletion
        delete pfdr;

        free(rY); free(reduced_loss_weights);
    }
}

TPL void CP_D1_LSX::compute_grad()
{
    /* gradient of smooth part of d11 penalization */
    Cp_d1<real_t, index_t, comp_t>::compute_grad();

    /* add gradient of differentiable loss term */
    const real_t c = (1.0 - loss), q = loss/D, r = q/c; // useful for KLs
    #ifdef _OPENMP
    uintmax_t num_ops = D*(V - saturated_vert)*
        (loss == linear_loss() || loss == quadratic_loss() ? 1 : 3);
    #endif
    #pragma omp parallel for schedule(static) NUM_THREADS(num_ops, V)
    for (index_t v = 0; v < V; v++){
        comp_t rv = comp_assign[v];
        if (is_saturated[rv]){ continue; }

        real_t* Gv = G + D*v;
        const real_t* rXv = rX + D*rv;
        const real_t* Yv = Y + D*v;

        if (loss == linear_loss()){ /* grad = - w Y */
            for (size_t d = 0; d < D; d++){ Gv[d] -= LOSS_WEIGHTS_(v)*Yv[d]; }
        }else if (loss == quadratic_loss()){ /* grad = w(X - Y) */
            for (size_t d = 0; d < D; d++){ 
                Gv[d] += LOSS_WEIGHTS_(v)*(rXv[d] - Yv[d]);
            }
        }else{ /* dKLs/dx_k = -(1-s)(s/D + (1-s)y_k)/(s/D + (1-s)x_k) */
            for (size_t d = 0; d < D; d++){ 
                Gv[d] -= LOSS_WEIGHTS_(v)*(q + c*Yv[d])/(r + rXv[d]);
            }
        }
    }
}

TPL real_t CP_D1_LSX::vert_split_cost(const Split_info& split_info, index_t v,
    comp_t k) const
{
    /* infinite cost for leaving the simplex */
    const real_t* rXv = rX + D*split_info.rv;
    const real_t* sXk = split_info.sX + D*k;
    for (size_t d = 0; d < D; d++){
        if ((rXv[d] <= eps && sXk[d] < -eps)
            || (rXv[d] >= (real_t) 1.0 - eps && sXk[d] > eps))
        { return real_inf(); }
    }

    return Cp_d1<real_t, index_t, comp_t>::vert_split_cost(split_info, v, k);
}

TPL real_t CP_D1_LSX::vert_split_cost(const Split_info& split_info, index_t v,
    comp_t k, comp_t l) const
{
    if (k == l){ return 0.0; }

    /* infinite cost for leaving the simplex */
    const real_t* rXv = rX + D*split_info.rv;
    const real_t* sXk = split_info.sX + D*k;
    const real_t* sXl = split_info.sX + D*l;
    for (size_t d = 0; d < D; d++){
        if (rXv[d] <= eps){
            if (sXk[d] < -eps){ return real_inf(); }
            else if (sXl[d] < -eps){ return -real_inf(); }
        }else if (rXv[d] >= (real_t) 1.0 - eps){
            if (sXk[d] > eps){ return real_inf(); }
            else if (sXl[d] > eps){ return -real_inf(); }
        }
    }

    return Cp_d1<real_t, index_t, comp_t>::vert_split_cost(split_info, v, k,
        l);
}

TPL void CP_D1_LSX::project_descent_direction(Split_info& split_info, comp_t k)
    const
{
    /* make suitable descent direction for staying in the standard simplex
     *      Δ_D = {x ∈ ℝ^D | ∀d 0 ≤ x_d ≤ 1,  and  ∑_d x_d = 1}
     * so "project" sXk
     *      argmin_x ⟨x, - sXk⟩
     * under the following constraints:
     *      ║x║ ≤ 1,
     *      ∑_d x_d = 0, 
     *      ∀d, rXv_d = 0 ⇒ x_d ≥ 0
     *      ∀d, rXv_d = 1 ⇒ x_d ≤ 0
     * optimality conditions read as
     *      - sXk + λx + μ 1 + ν = 0                    (1)
     * where
     *      λ ≥ 0 if x ≠ 0,
     *      μ ∈ ℝ, 1 stands for (1, ..., 1)
     *      ∀d, v_d ≤ 0 if rXv_d = 0 and x_d = 0
     *          v_d ≥ 0 if rXv_d = 1 and x_d = 0
     *          ν_d = 0 otherwise
     * in particular, with
     *      I0 = {d | rXv_d = 0 and x_d = 0}
     *      I1 = {d | rXv_d = 1 and x_d = 0}
     *      I- = {1,...,d}\(I0 υ I1)
     *          (nota: |I1| = 0 or 1 because rXv ∈ Δ_D)
     * μ must satisfy
     *      ∀d ∈ I0, ν_d = sXk_d - μ ≤ 0, so μ ≥ sXk_d,
     *      ∀d ∈ I1, ν_d = sXk_d - μ ≥ 0, so μ ≤ sXk_d, (2)
     *      if I- ≠ ∅, μ = ∑_{d ∈ I-} sXk_d / |I-|
     * so this problem amounts to finding I0 and I1 so that the above
     * is satisfied; λ is deduced by normalizing sXk - μ - ν obtained from (1);
     * sort {sXk_d | rXv_d = 0} and {sXk_d | rXv_d = 1} and remove iteratively
     * the index of the current maximum from I0 or minimum from I1 while
     * maintaining the mean m over I-, until (2) is satisfied with μ = m */
    const real_t* rXv = rX + D*split_info.rv;
    real_t* sXk = split_info.sX + D*k;
    size_t* I = (size_t*) malloc_check(sizeof(size_t)*D);
    size_t nI0 = 0, nI1 = 0, nI_ = 0;
    real_t m = 0.0;
    for (size_t d = 0; d < D; d++){
        if (rXv[d] <= eps){ I[nI0++] = d; }
        else if (rXv[d] >= (real_t) 1.0 - eps){ I[D - ++nI1] = d; }
        else { m += rXv[d]; nI_++; }
    }
    sort(I, I + nI0,
        [sXk] (comp_t d1, comp_t d2) -> bool { return sXk[d1] < sXk[d2]; });
    if (nI1){ 
        /* |I1| is actually 0 or 1 because rXv ∈ Δ_D; if |I1| = 1, then rXv is
         * all 0 except one 1 ; at this stage |I0| = D - 1 and |I-| = 0 so m is
         * ill-defined */
        if (sXk[I[nI0 - 1]] <= sXk[I[D - 1]]){
        /* specific but common case: sXk_d is greatest where rXv_d = 1;
         * (2) is satisfied with any μ ∈ [max{sXk_d|rXv_d=0}, sXk_{d:rXv_d=1}]
         * and x = 0, that is no admissible descent direction exists */
            for (size_t d = 0; d < D; d++){ sXk[d] = 0.0; }
            free(I); return;
        } /* else, indices of sXk_d = 1 and the largest sX_d = 0 are in I- */
        m = sXk[I[--nI0]] + sXk[I[D - 1]];
        nI_ = 2;
    }
    while (nI0-- && m/nI_ < sXk[I[nI0]]){ m += sXk[I[nI0]]; nI_++; }
    m /= nI_;
    /* now μ = m satisfies (2), deduce corresponding un-normalized x */
    for (size_t d = 0; d < D; d++){
        if (rXv[d] <= eps && sXk[d] <= m){ sXk[d] = 0.0; }
        else if (rXv[d] >= (real_t) 1.0 - eps && sXk[d] >= m){ sXk[d] = 0.0; }
        else { sXk[d] -= m; }
    }
    /* normalize */
    Cp_d1<real_t, index_t, comp_t>::project_descent_direction(split_info, k);
    free(I);
}

TPL index_t CP_D1_LSX::merge()
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
            real_t dif = 0.0;
            for (size_t d = 0; d < D; d++){ dif += abs(rXv[d] - lrXv[d]); }
            if (dif > dif_tol){
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

TPL real_t CP_D1_LSX::compute_evolution() const
{
    real_t dif = 0.0;
    #pragma omp parallel for schedule(dynamic) \
        NUM_THREADS(D*(V - saturated_vert), rV) reduction(+:dif)
    for (comp_t rv = 0; rv < rV; rv++){
        const real_t* rXv = rX + D*rv;
        if (is_saturated[rv]){
            const real_t* lrXv = last_rX +
                 D*last_comp_assign[comp_list[first_vertex[rv]]];
            real_t dif_rv = 0.0;
            for (size_t d = 0; d < D; d++){ dif_rv += abs(rXv[d] - lrXv[d]); }
            dif += dif_rv*(first_vertex[rv + 1] - first_vertex[rv]);
        }else{
            for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
                real_t* lrXv = last_rX + D*last_comp_assign[comp_list[i]];
                for (size_t d = 0; d < D; d++){ dif += abs(rXv[d] - lrXv[d]); }
            }
        }
    }
    return dif/V;
}


TPL real_t CP_D1_LSX::compute_objective() const
/* unfortunately, at this point one does not have access to the reduced objects
 * computed in the routine solve_reduced_problem() */
{
    real_t obj = 0.0;

    if (loss == linear_loss()){
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:obj)
        for (index_t v = 0; v < V; v++){
            real_t* rXv = rX + comp_assign[v]*D;
            const real_t* Yv = Y + v*D;
            real_t prod = 0.0;
            for (size_t d = 0; d < D; d++){ prod += rXv[d]*Yv[d]; }
            obj -= LOSS_WEIGHTS_(v)*prod;
        }
    }else if (loss == quadratic_loss()){
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:obj)
        for (index_t v = 0; v < V; v++){
            const real_t* rXv = rX + comp_assign[v]*D;
            const real_t* Yv = Y + v*D;
            real_t dif2 = 0.0;
            for (size_t d = 0; d < D; d++){
                dif2 += (rXv[d] - Yv[d])*(rXv[d] - Yv[d]);
            }
            obj += LOSS_WEIGHTS_(v)*dif2;
        }
        obj /= 2.0;
    }else{ /* smoothed Kullback-Leibler */
        const real_t c = (1.0 - loss);
        const real_t q = loss/D;
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:obj) 
        for (index_t v = 0; v < V; v++){
            const real_t* rXv = rX + comp_assign[v]*D;
            const real_t* Yv = Y + v*D;
            real_t KLs = 0.0;
            for (size_t d = 0; d < D; d++){
                real_t ys = q + c*Yv[d];
                KLs += ys*log(ys/(q + c*rXv[d]));
            }
            obj += LOSS_WEIGHTS_(v)*KLs;
        }
    }

    obj += compute_graph_d1(); // ||x||_d1

    return obj;
}

/**  instantiate for compilation  **/
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Cp_d1_lsx<float, int32_t, int16_t>;
template class Cp_d1_lsx<double, int32_t, int16_t>;
template class Cp_d1_lsx<float, int32_t, int32_t>;
template class Cp_d1_lsx<double, int32_t, int32_t>;
#else
template class Cp_d1_lsx<float, uint32_t, uint16_t>;
template class Cp_d1_lsx<double, uint32_t, uint16_t>;
template class Cp_d1_lsx<float, uint32_t, uint32_t>;
template class Cp_d1_lsx<double, uint32_t, uint32_t>;
#endif
