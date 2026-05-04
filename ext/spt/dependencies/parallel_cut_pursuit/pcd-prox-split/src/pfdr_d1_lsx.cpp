/*=============================================================================
 * Hugo Raguet 2016, 2018, 2023
 *===========================================================================*/
#include <cmath>
#include "omp_num_threads.hpp"
#include "pfdr_d1_lsx.hpp"
#include "proj_simplex.hpp"

#define LOSS_WEIGHTS_(v) (loss_weights ? loss_weights[(v)] : (real_t) 1.0)
#define Ga_(v, vd) (gashape == MONODIM ? Ga[(v)] : Ga[(vd)])
#define W_Ga_Y_(v, vd) (gashape == MONODIM ? W_Ga_Y[(v)] : W_Ga_Y[(vd)])

#define TPL template <typename real_t, typename vertex_t>
#define PFDR_D1_LSX Pfdr_d1_lsx<real_t, vertex_t>

using namespace std;

TPL PFDR_D1_LSX::Pfdr_d1_lsx(vertex_t V, index_t E, const vertex_t* edges,
    real_t loss, index_t D, const real_t* Y, const real_t* d11_metric)
    : Pfdr_d1<real_t, vertex_t>(V, E, edges, D, D11, d11_metric, 
        loss == linear_loss() ? SCALAR :
        loss == quadratic_loss() ? MONODIM : MULTIDIM),
    loss(loss), Y(Y)
{
    W_Ga_Y = nullptr;
    loss_weights = nullptr;
    /* Lipschitz metric useful only during preconditioning, no point in wasting
     * memory for saving so few computations */
    lipschcomput = EACH; 
}

TPL PFDR_D1_LSX::~Pfdr_d1_lsx(){ if (W_Ga_Y != Ga){ free(W_Ga_Y); } }

TPL void PFDR_D1_LSX::set_loss(real_t loss, const real_t* Y,
    const real_t* loss_weights)
{
    if (loss < 0.0 || loss > 1.0){
        cerr << "PFDR graph d1 loss simplex: loss parameter should be between "
            "0 and 1 (" << loss << " given)." << endl;
        exit(EXIT_FAILURE);
    }
    if ((this->loss != loss) &&
        (this->loss == linear_loss() || this->loss == quadratic_loss() ||
         loss == linear_loss() || loss == quadratic_loss())){
        cerr << "PFDR graph d1 loss simplex: the type of loss cannot "
            "be changed; for changing from one loss type to another, create "
            "a new instance of Pfdr_d1_lsx." << endl;
        exit(EXIT_FAILURE);
    }
    this->loss = loss;
    if (Y){ this->Y = Y; }
    this->loss_weights = loss_weights;
}

TPL void PFDR_D1_LSX::compute_lipschitz_metric()
{
    if (loss == linear_loss()){
        l = 0.0; lshape = SCALAR;
    }else if (loss == quadratic_loss()){
        if (loss_weights){ L = loss_weights; lshape = MONODIM; }
        else{ l = 1.0; lshape = SCALAR; }
    }else{ /* KLs loss, Ld = max_{0 <= x_d <= 1} d^2KLs/dx_d^2
            *              = (1-s)^2/(s/D)^2 (s/D + (1-s)y_d) */
        const real_t c = (1.0 - loss);
        const real_t q = loss/D;
        const real_t r = c*c/(q*q);
        Lmut = (real_t*) malloc_check(sizeof(real_t)*V*D); 
        #pragma omp parallel for schedule(static) NUM_THREADS(2*V*D, V)
        for (vertex_t v = 0; v < V; v++){
            const real_t* Yv = Y + D*v;
            real_t* Lv = Lmut + D*v;
            for (index_t d = 0; d < D; d++){
                Lv[d] = LOSS_WEIGHTS_(v)*r*(q + c*Yv[d]);
            }
        }
        L = Lmut; lshape = MULTIDIM;
    }
}

TPL void PFDR_D1_LSX::compute_hess_f()
{
    const index_t Dga = gashape == MULTIDIM ? D : 1;
    if (loss == linear_loss()){
        for (index_t vd = 0; vd < V*Dga; vd++){ Ga[vd] = 0.0; }
    }else if (loss == quadratic_loss()){
        for (vertex_t v = 0; v < V; v++){
            index_t vd = v*Dga;
            for (index_t d = 0; d < Dga; d++){ Ga[vd++] = LOSS_WEIGHTS_(v); }
        }
    }else{ /* d^2KLs/dx_d^2 = (1-s)^2 (s/D + (1-s)y_d)/(s/D + (1-s)x_d)^2 */
        const real_t c = (1.0 - loss);
        const real_t q = loss/D;
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V)
        for (vertex_t v = 0; v < V; v++){
            real_t* Xv = X + D*v;
            const real_t* Yv = Y + D*v;
            real_t* Gav = Ga + D*v;
            for (index_t d = 0; d < D; d++){
                real_t r = c/(q + c*Xv[d]);
                Gav[d] = LOSS_WEIGHTS_(v)*(q + c*Yv[d])*r*r;
            }
        }
    }
}

TPL void PFDR_D1_LSX::compute_Ga_grad_f()
{
    /**  forward and backward steps on auxiliary variables  **/
    /* explicit step */
    if (loss == linear_loss()){ /* grad = - w Y */
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V)
        for (vertex_t v = 0; v < V; v++){
            index_t vd = D*v;
            for (index_t d = 0; d < D; d++){
                Ga_grad_f[vd] = -W_Ga_Y_(v, vd)*Y[vd];
                vd++;
            }
        }
    }else if (loss == quadratic_loss()){ /* grad = w (X - Y) */
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V)
        for (vertex_t v = 0; v < V; v++){
            index_t vd = D*v;
            for (index_t d = 0; d < D; d++){
                Ga_grad_f[vd] = W_Ga_Y_(v, vd)*(X[vd] - Y[vd]);
                vd++;
            }
        }
    }else{ /* dKLs/dx_k = -(1-s)(s/D + (1-s)y_k)/(s/D + (1-s)x_k) */
        real_t r = loss/D/(1.0 - loss);
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D)
        for (index_t vd = 0; vd < V*D; vd++){
            Ga_grad_f[vd] = W_Ga_Y[vd]/(r + X[vd]);
        }
    }
}

TPL void PFDR_D1_LSX::compute_prox_Ga_h()
{
    if (gashape == MULTIDIM){
        proj_simplex::proj_simplex<real_t>(X, D, V, nullptr, 1.0, Ga);
    }else{
        proj_simplex::proj_simplex<real_t>(X, D, V, nullptr, 1.0);
    }
}

TPL real_t PFDR_D1_LSX::compute_f() const
{
    real_t obj = 0.0;
    if (loss == linear_loss()){
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:obj)
        for (vertex_t v = 0; v < V; v++){
            real_t* Xv = X + D*v;
            const real_t* Yv = Y + D*v;
            real_t prod = 0.0;
            for (index_t d = 0; d < D; d++){ prod += Xv[d]*Yv[d]; }
            obj -= LOSS_WEIGHTS_(v)*prod;
        }
    }else if (loss == quadratic_loss()){
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:obj)
        for (vertex_t v = 0; v < V; v++){
            real_t* Xv = X + D*v;
            const real_t* Yv = Y + D*v;
            real_t dif2 = 0.0;
            for (index_t d = 0; d < D; d++){
                dif2 += (Xv[d] - Yv[d])*(Xv[d] - Yv[d]);
            }
            obj += LOSS_WEIGHTS_(v)*dif2;
        }
        obj *= 0.5;
    }else{ /* smoothed Kullback-Leibler */
        const real_t c = (1.0 - loss);
        const real_t q = loss/D;
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
            reduction(+:obj) 
        for (vertex_t v = 0; v < V; v++){
            real_t* Xv = X + D*v;
            const real_t* Yv = Y + D*v;
            real_t KLs = 0.0;
            for (index_t d = 0; d < D; d++){
                real_t ys = q + c*Yv[d];
                KLs += ys*log(ys/(q + c*Xv[d]));
            }
            obj += LOSS_WEIGHTS_(v)*KLs;
        }
    }
    return obj;
}

TPL void PFDR_D1_LSX::preconditioning(bool init)
{
    Pfdr_d1<real_t, vertex_t>::preconditioning(init);

    /* precompute first-order information for loss gradient */
    if (loss == linear_loss() || loss == quadratic_loss()){
        /* linear loss, grad = - w Y; quadratic loss, grad = w (X - Y) */
        if (loss_weights){
            const index_t Dga = gashape == MULTIDIM ? D : 1;
            if (!W_Ga_Y){
                W_Ga_Y = (real_t*) malloc_check(sizeof(real_t)*V*Dga);
            }
            #pragma omp parallel for schedule(static) NUM_THREADS(V*Dga, V)
            for (vertex_t v = 0; v < V; v++){
                real_t* W_Ga_Yv = W_Ga_Y + Dga*v;
                real_t* Gav = Ga + Dga*v;
                for (index_t d = 0; d < Dga; d++){
                    W_Ga_Yv[d] = loss_weights[v]*Gav[d];
                }
            }
        }else{
            W_Ga_Y = Ga;
        }
    }else{ /* dKLs/dx_d = -(1-s)(s/D + (1-s)y_d)/(s/K + (1-s)x_d) */
        if (!W_Ga_Y){ W_Ga_Y = (real_t*) malloc_check(sizeof(real_t)*V*D); }
        const real_t c = (1.0 - loss);
        const real_t q = loss/D;
        #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V)
        for (vertex_t v = 0; v < V; v++){
            real_t* W_Ga_Yv = W_Ga_Y + D*v;
            real_t* Gav = Ga + D*v;
            const real_t* Yv = Y + D*v;
            for (index_t d = 0; d < D; d++){
                W_Ga_Yv[d] = -LOSS_WEIGHTS_(v)*Gav[d]*(q + c*Yv[d]);
            }
        }
    }
}

TPL void PFDR_D1_LSX::initialize_iterate()
{
    /*if (loss == linear_loss()){ */
        /* Yv might not lie on the simplex;
         * create a point on the simplex by removing the minimum value
         * (resulting problem loss + d1 + simplex problem strictly equivalent)
         * and dividing by the sum */
    /*
        #pragma omp parallel for schedule(static) NUM_THREADS(2*V*D, V)
        for (vertex_t v = 0; v < V; v++){
            const real_t* Yv = Y + D*v;
            real_t* Xv = X + D*v;
            real_t min = Yv[0], max = Yv[0], sum = Yv[0];
            for (index_t d = 1; d < D; d++){
                sum += Yv[d];
                if (Yv[d] < min){ min = Yv[d]; }
                else if (Yv[d] > max){ max = Yv[d]; }
            }
            if (min == max){ // avoid trouble if all equal
                for (index_t d = 0; d < D; d++){ Xv[d] = 1.0/D; }
            }else{
                sum -= D*min;
                for (index_t d = 0; d < D; d++){
                    Xv[d] = (Yv[d] - min)/sum;
                }
            }
        }
    }else{ */
        /* Yv lies on the simplex */

        /* currently all assumed to lie on the simplex */
        for (index_t vd = 0; vd < V*D; vd++){ X[vd] = Y[vd]; }

    /* } */
}

/* relative iterate evolution in l1 norm */
TPL real_t PFDR_D1_LSX::compute_evolution() const
{
    real_t dif = 0.0;
    real_t amp = 0.0;
    #pragma omp parallel for schedule(static) NUM_THREADS(V*D, V) \
        reduction(+:dif, amp)
    for (vertex_t v = 0; v < V; v++){
        const real_t* Xv = X + D*v;
        const real_t* last_Xv = last_X + D*v;
        real_t dif_v = 0.0; 
        for (index_t d = 0; d < D; d++){ dif_v += abs(last_Xv[d] - Xv[d]); }
        dif += LOSS_WEIGHTS_(v)*dif_v;
        amp += LOSS_WEIGHTS_(v);
    }
    return dif/amp;
}

/**  instantiate for compilation  **/
#include <cstdint>
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Pfdr_d1_lsx<float, int16_t>;
template class Pfdr_d1_lsx<float, int32_t>;
template class Pfdr_d1_lsx<double, int16_t>;
template class Pfdr_d1_lsx<double, int32_t>;
#else
template class Pfdr_d1_lsx<float, uint16_t>;
template class Pfdr_d1_lsx<float, uint32_t>;
template class Pfdr_d1_lsx<double, uint16_t>;
template class Pfdr_d1_lsx<double, uint32_t>;
#endif
