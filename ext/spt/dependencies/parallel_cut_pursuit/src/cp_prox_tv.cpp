/*=============================================================================
 * Hugo Raguet 2021, 2023
 *===========================================================================*/
#include <cmath>
#include "cp_prox_tv.hpp"
#include "pfdr_prox_tv.hpp"

#define L22_METRIC_(v, vd) (l22_metric_shape == IDENTITY ? (real_t) 1.0 : \
    l22_metric_shape == MONODIM ? l22_metric[(v)] : l22_metric[(vd)])

#define TPL template <typename real_t, typename index_t, typename comp_t>
#define CP_PROX_TV Cp_prox_tv<real_t, index_t, comp_t>
#define PFDR Pfdr_prox_tv<real_t, comp_t>

using namespace std;

TPL CP_PROX_TV::Cp_prox_tv(index_t V, index_t E, const index_t* first_edge,
    const index_t* adj_vertices, const real_t* Y, size_t D)
    : Cp_d1<real_t, index_t, comp_t>(V, E, first_edge, adj_vertices, D),
    Y(Y)
{
    /* TODO: subgradient retrieval */
    // Gd1 = nullptr;

    K = 2;
    split_iter_num = 1;
    split_damp_ratio = 1.0;
    if (D == 1){
        split_values_init_num = 1;
        split_values_iter_num = 1;
    }else{
        split_values_init_num = 2;
        split_values_iter_num = 2;
    }

    pfdr_rho = 1.0; pfdr_cond_min = 1e-2; pfdr_dif_rcd = 0.0;
    pfdr_dif_tol = 1e-2*dif_tol; pfdr_it = pfdr_it_max = 1e4;
}

/* TODO: subgradient retrieval */
/* TPL void CP_PROX_TV::set_d1_subgradients(real_t* Gd1)
{
    this->Gd1 = Gd1;
} */

TPL void CP_PROX_TV::set_quadratic(Metric_shape l22_metric_shape,
    const real_t* l22_metric)
{
    this->l22_metric_shape = l22_metric_shape;
    this->l22_metric = l22_metric;
}

TPL void CP_PROX_TV::set_quadratic(const real_t* Y,
    Metric_shape l22_metric_shape, const real_t* l22_metric)
{
    this->Y = Y;
    set_quadratic(l22_metric_shape, l22_metric);
}

TPL void CP_PROX_TV::set_pfdr_param(real_t rho, real_t cond_min,
    real_t dif_rcd, int it_max, real_t dif_tol)
{
    this->pfdr_rho = rho;
    this->pfdr_cond_min = cond_min;
    this->pfdr_dif_rcd = dif_rcd;
    this->pfdr_it_max = it_max;
    this->pfdr_dif_tol = dif_tol;
}

TPL void CP_PROX_TV::solve_reduced_problem()
{
    /**  compute reduced matrix  **/
    real_t *rY, *rl22M; // reduced observations and l22 metric
    rY = (real_t*) malloc_check(sizeof(real_t)*rV*D);
    rl22M = l22_metric_shape == MULTIDIM ?
        (real_t*) malloc_check(sizeof(real_t)*rV*D) :
        (real_t*) malloc_check(sizeof(real_t)*rV);

    #pragma omp parallel for schedule(dynamic) NUM_THREADS(V*D, rV)
    for (comp_t rv = 0; rv < rV; rv++){
        real_t* rYv = rY + D*rv;
        real_t* rl22Mv = rl22M + (l22_metric_shape == MULTIDIM ? D*rv : rv);
        for (size_t d = 0; d < D; d++){
            rYv[d] = 0.0;
            if (d == 0 || l22_metric_shape == MULTIDIM){ rl22Mv[d] = 0.0; }
        }
        /* run along the component rv */
        for (index_t i = first_vertex[rv]; i < first_vertex[rv + 1]; i++){
            index_t v = comp_list[i];
            size_t vd = D*v;
            for (size_t d = 0; d < D; d++){
                rYv[d] += L22_METRIC_(v, vd)*Y[vd];
                if (d == 0 || l22_metric_shape == MULTIDIM){
                    rl22Mv[d] += L22_METRIC_(v, vd);
                }
                vd++;
            }
        }
        for (size_t d = 0; d < D; d++){
            rYv[d] /= (l22_metric_shape == MULTIDIM ? rl22Mv[d] : rl22Mv[0]);
        }
    }
    
    if (rV == 1){ /**  single connected component  **/

        for (size_t d = 0; d < D; d++){ rX[d] = rY[d]; }

    }else{ /**  preconditioned forward-Douglas-Rachford  **/

        Pfdr_prox_tv<real_t, comp_t> *pfdr =
            new Pfdr_prox_tv<real_t, comp_t>(rV, rE, reduced_edges, rY, D,
                d1p == D11 ? PFDR::D11 : PFDR::D12, d1p_metric,
                l22_metric_shape == MULTIDIM ? PFDR::MULTIDIM : PFDR::MONODIM,
                rl22M);
        
        pfdr->set_edge_weights(reduced_edge_weights);
        pfdr->set_conditioning_param(pfdr_cond_min, pfdr_dif_rcd);
        pfdr->set_relaxation(pfdr_rho);
        pfdr->set_algo_param(pfdr_dif_tol, sqrt(pfdr_it_max), pfdr_it_max,
            verbose);
        pfdr->set_iterate(rX);
        pfdr->initialize_iterate();

        pfdr_it = pfdr->precond_proximal_splitting();

        pfdr->set_iterate(nullptr); // prevent rX to be free()'d
        delete pfdr;

    }

    free(rY); free(rl22M);
}

TPL void CP_PROX_TV::compute_grad()
{
    /**  gradient of smooth part of d12 penalization  **/
    Cp_d1<real_t, index_t, comp_t>::compute_grad();

    /**  gradient of quadratic term  **/ 
    #pragma omp parallel for schedule(static) NUM_THREADS(V - saturated_vert)
    for (index_t v = 0; v < V; v++){
        comp_t rv = comp_assign[v];
        if (is_saturated[rv]){ continue; }

        real_t* Gv = G + D*v;
        const real_t* rXv = rX + D*rv;
        const real_t* Yv = Y + D*v;

        size_t vd = D*v;
        for (size_t d = 0; d < D; d++){
            Gv[d] += L22_METRIC_(v, vd)*(rXv[d] - Yv[d]);
            vd++;
        }
    }
}

TPL real_t CP_PROX_TV::compute_objective() const
{
    real_t obj = 0.0;

    #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
        reduction(+:obj)
    for (index_t v = 0; v < V; v++){
        const real_t* rXv = rX + D*comp_assign[v];
        size_t vd = D*v;
        for (size_t d = 0; d < D; d++){
            obj += L22_METRIC_(v, vd)*(rXv[d] - Y[vd])*(rXv[d] - Y[vd]);
            vd++;
        }
    }
    obj /= 2.0;

    obj += compute_graph_d1(); // ||x||_d1

    return obj;
}

/**  instantiate for compilation  **/
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Cp_prox_tv<double, int32_t, int16_t>;
template class Cp_prox_tv<float, int32_t, int16_t>;
template class Cp_prox_tv<double, int32_t, int32_t>;
template class Cp_prox_tv<float, int32_t, int32_t>;
#else
template class Cp_prox_tv<double, uint32_t, uint16_t>;
template class Cp_prox_tv<float, uint32_t, uint16_t>;
template class Cp_prox_tv<double, uint32_t, uint32_t>;
template class Cp_prox_tv<float, uint32_t, uint32_t>;
#endif
