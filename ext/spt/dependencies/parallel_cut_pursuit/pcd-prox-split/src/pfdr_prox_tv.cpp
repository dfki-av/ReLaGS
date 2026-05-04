/*=============================================================================
 * Hugo Raguet 2023
 *===========================================================================*/
#include <cmath>
#include "omp_num_threads.hpp"
#include "pfdr_prox_tv.hpp"

#define Ga_(v, vd)  (gashape == SCALAR ? ga : \
                     gashape == MONODIM ? Ga[(v)] : Ga[(vd)])
#define L22_METRIC_(v, vd)   (l22_metric_shape == SCALAR ? (real_t) 1.0 : \
          l22_metric_shape == MONODIM ? l22_metric[(v)] : l22_metric[(vd)])

#define TPL template <typename real_t, typename vertex_t>
#define PFDR_PROX_TV Pfdr_prox_tv<real_t, vertex_t>

using namespace std;

TPL PFDR_PROX_TV::Pfdr_prox_tv(vertex_t V, index_t E, const vertex_t* edges,
    const real_t* Y, index_t D, D1p d1p, const real_t* d1p_metric,
    Condshape l22_metric_shape, const real_t* l22_metric)
    : Pfdr_d1<real_t, vertex_t>(V, E, edges, D, d1p, d1p_metric,
        l22_metric_shape),
      Y(Y), l22_metric_shape(l22_metric_shape), l22_metric(l22_metric)
{
    /* ensure handling of infinite values (negation, comparisons) is safe */
    static_assert(numeric_limits<real_t>::is_iec559,
        "PFDR prox TV: real_t must satisfy IEEE 754.");
    set_lipschitz_param(l22_metric, 1.0, l22_metric_shape);
}

TPL void PFDR_PROX_TV::compute_hess_f()
{
    if (l22_metric_shape == SCALAR){
        for (vertex_t v = 0; v < V; v++){ Ga[v] = 1.0; }
    }else if (l22_metric_shape == MONODIM){
        for (vertex_t v = 0; v < V; v++){ Ga[v] = l22_metric[v]; }
    }else{
        for (index_t vd = 0; vd < V*D; vd++){ Ga[vd] = l22_metric[vd]; }
    }
}

TPL void PFDR_PROX_TV::compute_Ga_grad_f()
{
    #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V)
    for (vertex_t v = 0; v < V; v++){
        index_t vd = D*v;
        for (index_t d = 0; d < D; d++){
            Ga_grad_f[vd] = Ga_(v, vd)*L22_METRIC_(v, vd)*(X[vd] - Y[vd]);
            vd++;
        }
    }
}

TPL real_t PFDR_PROX_TV::compute_f() const
{
    real_t obj = 0.0;
    #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
        reduction(+:obj)
    for (vertex_t v = 0; v < V; v++){
        index_t vd = D*v;
        for (index_t d = 0; d < D; d++){
            obj += L22_METRIC_(v, vd)*(X[vd] - Y[vd])*(X[vd] - Y[vd]);
            vd++;
        }
    }
    return obj/2.0;
}

/**  instantiate for compilation  **/
#include <cstdint>
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Pfdr_prox_tv<float, int16_t>;
template class Pfdr_prox_tv<float, int32_t>;
template class Pfdr_prox_tv<double, int16_t>;
template class Pfdr_prox_tv<double, int32_t>;
#else
template class Pfdr_prox_tv<float, uint16_t>;
template class Pfdr_prox_tv<float, uint32_t>;
template class Pfdr_prox_tv<double, uint16_t>;
template class Pfdr_prox_tv<double, uint32_t>;
#endif
