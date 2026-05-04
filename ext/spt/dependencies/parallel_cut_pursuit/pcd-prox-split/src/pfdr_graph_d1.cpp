/*=============================================================================
 * Hugo Raguet 2016, 2023
 *===========================================================================*/
#include <cmath>
#include "omp_num_threads.hpp"
#include "pfdr_graph_d1.hpp"

/* macros for indexing data arrays depending on their shape */
#define EDGE_WEIGHTS_(e) (edge_weights ? edge_weights[(e)] : homo_edge_weight)
#define D1P_METRIC_(d) (d1p_metric ? d1p_metric[(d)] : (real_t) 1.0)
#define W_d1_(i, id)  (wd1shape == SCALAR ? w_d1 : \
                       wd1shape == MONODIM ? W_d1[(i)] : W_d1[(id)])
#define Th_d1_(e, ed) (thd1shape == SCALAR ? th_d1 : \
                       thd1shape == MONODIM ? Th_d1[(e)] : Th_d1[(ed)])

#define TPL template <typename real_t, typename vertex_t>
#define PFDR_D1 Pfdr_d1<real_t, vertex_t>

using namespace std;

TPL PFDR_D1::Pfdr_d1(vertex_t V, index_t E, const vertex_t* edges, index_t D,
    D1p d1p, const real_t* d1p_metric, Condshape hess_f_h_shape) :
    Pfdr<real_t, vertex_t>(V, 2*E, edges, D,
        compute_ga_shape(d1p_metric, hess_f_h_shape),
        compute_w_shape(d1p, d1p_metric, hess_f_h_shape)),
    E(E), d1p(d1p), d1p_metric(d1p_metric),
    wd1shape(compute_wd1_shape(d1p, d1p_metric, hess_f_h_shape)),
    thd1shape(compute_thd1_shape(d1p, d1p_metric, hess_f_h_shape))
{
    edge_weights = nullptr;
    homo_edge_weight = 1.0;
    W_d1 = Th_d1 = nullptr;
}

TPL PFDR_D1::~Pfdr_d1(){ free(W_d1); free(Th_d1); }

TPL void PFDR_D1::set_edge_weights(const real_t* edge_weights,
    real_t homo_edge_weight, const real_t* d1p_metric)
{
    this->edge_weights = edge_weights;
    this->homo_edge_weight = homo_edge_weight;
    if (!this->d1p_metric != !d1p_metric){
        cerr << "PFDR graph d1: d1p_metric attribute cannot be changed from "
            "null to varying weights or vice versa; for changing these "
            "weights, create a new instance of Pfdr_d1." << endl;
        exit(EXIT_FAILURE);
    }
    this->d1p_metric = d1p_metric;
}

TPL void PFDR_D1::add_pseudo_hess_g()
/* d1 contribution and splitting weights
 * a local quadratic approximation of (x1,x2) -> ||x1 - x2|| at (y1,y2) is
 * x -> 1/2 (||x1 - x2||^2/||y1 - y2|| + ||y1 - y2||)
 * whose hessian is not diagonal, but keeping only the diagonal terms yields
 * the second order derivative 1/||y1 - y2|| */
{
    /* finite differences and amplitudes */
    #pragma omp parallel for schedule(static) NUM_THREADS(4*E, E)
    for (index_t e = 0; e < E; e++){
        real_t* Xu = X + edges[2*e]*D;
        real_t* Xv = X + edges[2*e + 1]*D;
        real_t dif = 0.0, ampu = 0.0, ampv = 0.0;
        for (index_t d = 0; d < D; d++){
            if (d1p == D11){
                dif += abs(Xu[d] - Xv[d])*D1P_METRIC_(d);
                ampu += abs(Xu[d])*D1P_METRIC_(d);
                ampv += abs(Xv[d])*D1P_METRIC_(d);
            }else{
                dif += (Xu[d] - Xv[d])*(Xu[d] - Xv[d])*D1P_METRIC_(d);
                ampu += Xu[d]*Xu[d]*D1P_METRIC_(d);
                ampv += Xv[d]*Xv[d]*D1P_METRIC_(d);
            }
        }
        real_t amp;
        if (d1p == D11){
            amp = ampu > ampv ? ampu : ampv;
        }else{
            dif = sqrt(dif);
            amp = ampu > ampv ? sqrt(ampu) : sqrt(ampv);
        }
        /* stability of the preconditioning */
        if (dif < amp*cond_min){ dif = amp*cond_min; }
        if (dif < eps){ dif = eps; }
        Th_d1[e] = EDGE_WEIGHTS_(e)/dif; /* use Th_d1 as temporary storage */
    }

    /* actual pseudo-hessian, can be parallelized along coordinates */
    const index_t Dga = gashape == MULTIDIM ? D : 1;
    const index_t Dw = wshape == MULTIDIM ? D : 1; /* Dw <= Dga */
    #pragma omp parallel for schedule(static) NUM_THREADS(4*E*Dga, Dga)
    for (index_t d = 0; d < Dga; d++){
        index_t id = d;
        index_t jd = d + Dw;
        for (index_t e = 0; e < E; e++){
            real_t coef = D1P_METRIC_(d)*Th_d1[e];
            Ga[d + edges[2*e]*Dga] += coef;
            Ga[d + edges[2*e + 1]*Dga] += coef;
            if (!Id_W && (wshape == MULTIDIM || d == 0)){
                /* zero weight might create NaNs when normalized */
                W[id] = W[jd] = coef > 0.0 ? coef :
                                             numeric_limits<real_t>::min();
                id += 2*Dw;
                jd += 2*Dw;
            }
        }
    }
}

TPL void PFDR_D1::make_sum_Wi_Id()
{
    /* compute splitting weights sum */
    real_t* sum_Wi;
    /* use temporary storage if available */
    const index_t Dwd1 = wd1shape == MULTIDIM ? D : wd1shape == MONODIM ? 1: 0;
    const index_t Dthd1 = thd1shape == MULTIDIM ? D : 1;

    if (2*E*Dwd1 >= V){ sum_Wi = W_d1; }
    else if (E*Dthd1 >= V){ sum_Wi = Th_d1; }
    else{ sum_Wi = (real_t*) malloc_check(sizeof(real_t)*V); }

    for (index_t v = 0; v < V; v++){ sum_Wi[v] = 0.0; }
    for (index_t e = 0; e < 2*E; e++){
        sum_Wi[edges[e]] += Id_W ? (real_t) 1.0 : W[e];
    }

    if (!Id_W){ /* weights can just be normalized */

        #pragma omp parallel for schedule(static) NUM_THREADS(2*E)
        for (index_t e = 0; e < 2*E; e++){ W[e] /= sum_Wi[edges[e]]; }

    }else{ /* weights are used in order to shape the metric */
        /* compute shape and maximum */
        #pragma omp parallel for schedule(static) NUM_THREADS(2*V*D, V)
        for (index_t v = 0; v < V; v++){
            index_t vd = v*D;
            real_t wmax = Id_W[vd] = Ga[vd]*D1P_METRIC_(0);
            vd++;
            for (index_t d = 1; d < D; d++){
                Id_W[vd] = Ga[vd]*D1P_METRIC_(d);
                if (Id_W[vd] > wmax){ wmax = Id_W[vd]; }
                vd++;
            }
            vd -= D;
            for (index_t d = 0; d < D; d++){
                Id_W[vd] = (real_t) 1.0 - Id_W[vd]/wmax;
                vd++;
            }
        }
        /* set weights */
        #pragma omp parallel for schedule(static) NUM_THREADS(2*E*D, 2*E)
        for (index_t e = 0; e < 2*E; e++){
            index_t v = edges[e];
            index_t vd = v*D;
            index_t ed = e*D;
            for (index_t d = 0; d < D; d++){
                W[ed++] = ((real_t) 1.0 - Id_W[vd++])/sum_Wi[v];
            }
        }
    }

    if (2*E*Dwd1 < V && E*Dthd1 < V){ free(sum_Wi); }
}

TPL void PFDR_D1::preconditioning(bool init)
{
    /* allocate weights and thresholds for d1 prox operator */
    if (!W_d1 && wd1shape != SCALAR){
        index_t wd1size = 2*E*(wd1shape == MULTIDIM ? D : 1);
        W_d1 = (real_t*) malloc_check(sizeof(real_t)*wd1size);
    }
    if (!Th_d1){
        index_t thd1size = E*(thd1shape == MULTIDIM ? D : 1);
        Th_d1 = (real_t*) malloc_check(sizeof(real_t)*thd1size);
    }

    /* allocate supplementary weights and auxiliary variables if necessary */
    if (!Id_W && wshape == MULTIDIM){
        Id_W = (real_t*) malloc_check(sizeof(real_t)*V*D);
        if (!Z_Id && rho != 1.0){
            Z_Id = (real_t*) malloc_check(sizeof(real_t)*V*D);
        }
    }

    Pfdr<real_t, vertex_t>::preconditioning(init);

    /* precompute weights and thresholds for d1 prox operator */
    if (wd1shape == SCALAR){ w_d1 = 0.5; }
    const index_t Dd1 = thd1shape == MULTIDIM ? D : 1;
    const index_t Dga = gashape == MULTIDIM ? D : 1;
    const index_t Dw = wshape == MULTIDIM ? D : 1;
    #pragma omp parallel for schedule(static) NUM_THREADS(8*E*Dd1, E)
    for (index_t e = 0; e < E; e++){
        index_t i = 2*e;
        index_t j = 2*e + 1;
        vertex_t u = edges[i];
        vertex_t v = edges[j];
        index_t ud = u*Dga;
        index_t vd = v*Dga;
        index_t ed = e*Dd1;
        index_t id, jd;
        id = jd = 0; // avoid uninitialization warning
        if (wd1shape != SCALAR){
            id = i*Dd1;
            jd = j*Dd1;
        }
        for (index_t d = 0; d < Dd1; d++){
            real_t w_ga_u = W[i*Dw]/Ga[ud++];
            real_t w_ga_v = W[j*Dw]/Ga[vd++];
            Th_d1[ed++] = EDGE_WEIGHTS_(e)*D1P_METRIC_(d)
                *((real_t) 1.0/w_ga_u + (real_t) 1.0/w_ga_v);
            if (wd1shape != SCALAR){
                if (w_ga_u == 0.0 && w_ga_v == 0.0){
                    W_d1[id++] = 0.5;
                    W_d1[jd++] = 0.5;
                }else{
                    W_d1[id++] = w_ga_u/(w_ga_u + w_ga_v);
                    W_d1[jd++] = w_ga_v/(w_ga_u + w_ga_v);
                }
            }
        }
    }
}

TPL void PFDR_D1::compute_prox_GaW_g()
{
    #pragma omp parallel for schedule(static) NUM_THREADS(8*E*D, E)
    for (index_t e = 0; e < E; e++){
        index_t i = 2*e;
        index_t j = 2*e + 1;
        index_t ud = edges[i]*D;
        index_t vd = edges[j]*D;
        index_t id = i*D;
        index_t jd = j*D;
        index_t ed = 0; // avoid uninitialization warning
        real_t dnorm = 0.0; // avoid uninitialization warning
        if (d1p == D12){ /* compute norm */
            for (index_t d = 0; d < D; d++){ 
                /* forward step */ 
                real_t fwd_zi = Ga_grad_f[ud++] - Z[id++];
                real_t fwd_zj = Ga_grad_f[vd++] - Z[jd++];
                dnorm += (fwd_zi - fwd_zj)*(fwd_zi - fwd_zj)*D1P_METRIC_(d);
            }
            dnorm = sqrt(dnorm);
            ud -= D; vd -= D; id -= D; jd -= D;
        }else if (thd1shape == MULTIDIM){
            ed = e*D;
        }
        /* soft thresholding, update and relaxation */
        for (index_t d = 0; d < D; d++){
            /* forward step */ 
            real_t fwd_zi = Ga_grad_f[ud] - Z[id];
            real_t fwd_zj = Ga_grad_f[vd] - Z[jd];
            /* backward step */
            real_t avg = W_d1_(i, id)*fwd_zi + W_d1_(j, jd)*fwd_zj;
            real_t dif = fwd_zi - fwd_zj;
            if (d1p == D11){
                if (dif > Th_d1_(e, ed)){ dif -= Th_d1_(e, ed); }
                else if (dif < -Th_d1_(e, ed)){ dif += Th_d1_(e, ed); }
                else{ dif = 0.0; }
                if (thd1shape == MULTIDIM){ ed++; }
            }else{
                dif *= dnorm > Th_d1[e] ? (real_t) 1.0 - Th_d1[e]/dnorm : 0.0;
            }
            Z[id] += rho*(avg + W_d1_(j, jd)*dif - X[ud]);
            Z[jd] += rho*(avg - W_d1_(i, id)*dif - X[vd]);
            ud++; vd++; id++; jd++;
        }
    }
}

TPL real_t PFDR_D1::compute_g() const
{
    real_t obj = 0.0;
    #pragma omp parallel for schedule(static) NUM_THREADS(2*E*D, E) \
        reduction(+:obj)
    for (index_t e = 0; e < E; e++){
        index_t ud = edges[2*e]*D;
        index_t vd = edges[2*e + 1]*D;
        real_t dif = 0.0;
        for (index_t d = 0; d < D; d++){
            if (d1p == D11){
                dif += abs(X[ud] - X[vd])*D1P_METRIC_(d);
            }else{
                dif += (X[ud] - X[vd])*(X[ud] - X[vd])*D1P_METRIC_(d);
            }
            ud++; vd++;
        }
        if (d1p == D12){ dif = sqrt(dif); }
        obj += EDGE_WEIGHTS_(e)*dif;
    }
    return obj;
}

/* instantiate for compilation */
#include <cstdint>
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Pfdr_d1<float, int16_t>;
template class Pfdr_d1<float, int32_t>;
template class Pfdr_d1<double, int16_t>;
template class Pfdr_d1<double, int32_t>;
#else
template class Pfdr_d1<float, uint16_t>;
template class Pfdr_d1<float, uint32_t>;
template class Pfdr_d1<double, uint16_t>;
template class Pfdr_d1<double, uint32_t>;
#endif
