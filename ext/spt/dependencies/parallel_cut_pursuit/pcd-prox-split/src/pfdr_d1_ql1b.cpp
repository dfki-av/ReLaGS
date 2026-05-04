/*=============================================================================
 * Hugo Raguet 2016, 2023
 *===========================================================================*/
#include <cmath>
#include "omp_num_threads.hpp"
#include "pfdr_d1_ql1b.hpp"
#include "matrix_tools.hpp"

#define Y_(n) (Y ? Y[(n)] : (real_t) 0.0)
#define Yl1_(v) (Yl1 ? Yl1[(v)] : (real_t) 0.0)
#define L1_WEIGHTS_(v) (l1_weights ? l1_weights[(v)] : homo_l1_weight)

#define TPL template <typename real_t, typename vertex_t>
#define PFDR_D1_QL1B Pfdr_d1_ql1b<real_t, vertex_t>

using namespace std;

TPL PFDR_D1_QL1B::Pfdr_d1_ql1b(vertex_t V, index_t E, const vertex_t* edges)
    : Pfdr_d1<real_t, vertex_t>(V, E, edges)
{
    /* ensure handling of infinite values (negation, comparisons) is safe */
    static_assert(numeric_limits<real_t>::is_iec559,
        "PFDR d1 quadratic l1 bounds: real_t must satisfy IEEE 754.");
    Y = Yl1 = A = R = nullptr;
    N = Gram_diag();
    a = 1.0;
    l1_weights = nullptr; homo_l1_weight = 0.0;
    low_bnd = nullptr; homo_low_bnd = -real_inf();
    upp_bnd = nullptr; homo_upp_bnd = real_inf();

    lipsch_equi = JACOBI;
    lipsch_norm_tol = 1e-3;
    lipsch_norm_it_max = 100;
    lipsch_norm_nb_init = 10;
}

TPL PFDR_D1_QL1B::~Pfdr_d1_ql1b(){ free(R); }

TPL void PFDR_D1_QL1B::set_lipsch_norm_param(Equilibration lipsch_equi,
    real_t lipsch_norm_tol, int lipsch_norm_it_max, int lipsch_norm_nb_init)
{
    this->lipsch_equi = lipsch_equi;
    this->lipsch_norm_tol = lipsch_norm_tol;
    this->lipsch_norm_it_max = lipsch_norm_it_max;
    this->lipsch_norm_nb_init = lipsch_norm_nb_init;
}

TPL void PFDR_D1_QL1B::set_quadratic(const real_t* Y, index_t N,
    const real_t* A, real_t a)
{
    if (!A){ N = Gram_diag(); }
    free(R);
    R = is_Gram(N) ? nullptr : (real_t*) malloc_check(sizeof(real_t)*N);
    this->Y = Y; this->N = N; this->A = A; this->a = a;
}

TPL void PFDR_D1_QL1B::set_l1(const real_t* l1_weights, real_t homo_l1_weight,
    const real_t* Yl1)
{
    if (!l1_weights && homo_l1_weight < 0.0){
        cerr << "PFDR graph d1 quadratic l1 bounds: negative homogeneous l1 "
            "penalization (" << homo_l1_weight << ")." << endl;
        exit(EXIT_FAILURE);
    }
    this->l1_weights = l1_weights; this->homo_l1_weight = homo_l1_weight;
    this->Yl1 = Yl1;
}

TPL void PFDR_D1_QL1B::set_bounds(const real_t* low_bnd, real_t homo_low_bnd,
    const real_t* upp_bnd, real_t homo_upp_bnd)
{
    if (!low_bnd && !upp_bnd && homo_low_bnd > homo_upp_bnd){
        cerr << "PFDR graph d1 quadratic l1 bounds: homogeneous lower bound ("
            << homo_low_bnd << ") greater than homogeneous upper bound ("
            << homo_upp_bnd << ")." << endl;
        exit(EXIT_FAILURE);
    }
    this->low_bnd = low_bnd; this->homo_low_bnd = homo_low_bnd;
    this->upp_bnd = upp_bnd; this->homo_upp_bnd = homo_upp_bnd;
}

TPL void PFDR_D1_QL1B::apply_A()
{
    if (!is_Gram(N)){ /* direct matricial case, compute residual R = Y - A X */
        #pragma omp parallel for schedule(static) NUM_THREADS(N*V, N)
        for (index_t n = 0; n < N; n++){
            R[n] = Y_(n);
            index_t i = n;
            for (vertex_t v = 0; v < V; v++){
                R[n] -= A[i]*X[v];
                i += N;
            }
        }
    }else if (N == Gram_full()){ /* premultiplied by A^t, compute (A^t A) X */
        #pragma omp parallel for schedule(static) NUM_THREADS(V*V, V)
        for (vertex_t v = 0; v < V; v++){
            const real_t *Av = A + V*v;
            AX[v] = 0.0;
            for (vertex_t u = 0; u < V; u++){ AX[v] += Av[u]*X[u]; }
        }
    }else if (A){ /* diagonal case, compute (A^t A) X */
        #pragma omp parallel for schedule(static) NUM_THREADS(V)
        for (vertex_t v = 0; v < V; v++){ AX[v] = A[v]*X[v]; }
    }else if (a){ /* identity matrix */
        for (vertex_t v = 0; v < V; v++){ AX[v] = X[v]; }
    }
}

TPL void PFDR_D1_QL1B::compute_lipschitz_metric()
{
    if (N == Gram_diag()){ /* diagonal case */
        if (A){
            L = A;
            lshape = MONODIM;
        }else if (a){ /* identity matrix */
            l = 1.0;
            lshape = SCALAR;
        }else{ /* no quadratic penalty */
            l = 0.0;
            lshape = SCALAR;
        }
    }else if (lipsch_equi == NOEQUI){
        l = matrix_tools::operator_norm_matrix(N, V, A,
                (const real_t*) nullptr, lipsch_norm_tol, lipsch_norm_it_max,
                lipsch_norm_nb_init);
        lshape = SCALAR;
    }else{
        Lmut = (real_t*) malloc_check(V*sizeof(real_t));
        switch (lipsch_equi){
        case NOEQUI: break; // not possible
        case JACOBI:
            matrix_tools::symmetric_equilibration_jacobi<real_t>(N, V, A,
                Lmut); break;
        case BUNCH:
            matrix_tools::symmetric_equilibration_bunch<real_t>(N, V, A, Lmut);
            break;
        }

        /* stability: ratio between two elements no more than cond_min;
         * outliers are expected to be in the high range, hence the minimum
         * is used for the base reference */
        real_t lmin = Lmut[0];
        #if defined _OPENMP && _OPENMP >= 201107
        /* MSVC still does not support max reduction (OpenMP 3.1) in 2020 */
        #pragma omp parallel for schedule(static) NUM_THREADS(V) \
            reduction(min:lmin)
        #endif
        for (vertex_t v = 0; v < V; v++){
            if (Lmut[v] < lmin){ lmin = Lmut[v]; }
        }
        real_t lmax = lmin/cond_min;
        #pragma omp parallel for schedule(static) NUM_THREADS(V)
        for (vertex_t v = 0; v < V; v++){
            if (Lmut[v] > lmax){ Lmut[v] = lmax; }
        }

        /* norm of the equilibrated matrix and final Lipschitz norm */
        l = matrix_tools::operator_norm_matrix(N, V, A, Lmut, lipsch_norm_tol, 
                lipsch_norm_it_max, lipsch_norm_nb_init);
        #pragma omp parallel for schedule(static) NUM_THREADS(2*V, V)
        for (vertex_t v = 0; v < V; v++){ Lmut[v] = l/(Lmut[v]*Lmut[v]); }
        L = Lmut;
        lshape = MONODIM;
    }
}

TPL void PFDR_D1_QL1B::compute_hess_f()
{ for (vertex_t v = 0; v < V; v++){ Ga[v] = L ? L[v] : l; } }

TPL void PFDR_D1_QL1B::add_pseudo_hess_h()
/* l1 contribution
 * a local quadratic approximation of x -> ||x|| at z is
 * x -> 1/2 (||x||^2/||z|| + ||z||)
 * whose second order derivative is 1/||z|| */
{
    if (l1_weights || homo_l1_weight){
        #pragma omp parallel for schedule(static) NUM_THREADS(3*V, V)
        for (vertex_t v = 0; v < V; v++){
            real_t amp = abs(X[v] - Yl1_(v));
            if (amp < eps){ amp = eps; }
            Ga[v] += L1_WEIGHTS_(v)/amp;
        }
    }
}

TPL void PFDR_D1_QL1B::compute_Ga_grad_f()
/* supposed to be called after apply_A() */
{
    if (!is_Gram(N)){ /* direct matricial case, grad = -(A^t) R */
        #pragma omp parallel for schedule(static) NUM_THREADS(V*N, V)
        for (vertex_t v = 0; v < V; v++){
            const real_t *Av = A + N*v;
            Ga_grad_f[v] = 0.0;
            for (index_t n = 0; n < N; n++){ Ga_grad_f[v] -= Av[n]*R[n]; }
            Ga_grad_f[v] *= Ga[v];
        }
    }else if (A || a){ /* premultiplied by A^t, grad = (A^t A) X - A^t Y */
        #pragma omp parallel for schedule(static) NUM_THREADS(V)
        for (vertex_t v = 0; v < V; v++){
            Ga_grad_f[v] = Ga[v]*(AX[v] - Y_(v));
        }
    }else{ /* no quadratic part */
        for (vertex_t v = 0; v < V; v++){ Ga_grad_f[v] = 0.0; }
    }
}

TPL void PFDR_D1_QL1B::compute_prox_Ga_h()
{
    #pragma omp parallel for schedule(static) NUM_THREADS(V)
    for (vertex_t v = 0; v < V; v++){
        if (l1_weights || homo_l1_weight){
            real_t th_l1 = L1_WEIGHTS_(v)*Ga[v];
            real_t dif = X[v] - Yl1_(v);
            if (dif > th_l1){ dif -= th_l1; }
            else if (dif < -th_l1){ dif += th_l1; }
            else{ dif = 0.0; }
            X[v] = Yl1_(v) + dif;
        }
        if (low_bnd){
            if (X[v] < low_bnd[v]){ X[v] = low_bnd[v]; }
        }else if (homo_low_bnd > -real_inf()){
            if (X[v] < homo_low_bnd){ X[v] = homo_low_bnd; }
        }
        if (upp_bnd){
            if (X[v] > upp_bnd[v]){ X[v] = upp_bnd[v]; }
        }else if (homo_upp_bnd < real_inf()){
            if (X[v] > homo_upp_bnd){ X[v] = homo_upp_bnd; }
        }
    }
}

TPL real_t PFDR_D1_QL1B::compute_f() const
{
    real_t obj = 0.0;
    if (!is_Gram(N)){ /* direct matricial case, 1/2 ||Y - A X||^2 */
        #pragma omp parallel for schedule(static) NUM_THREADS(N) \
            reduction(+:obj)
        for (index_t n = 0; n < N; n++){ obj += R[n]*R[n]; }
        obj *= 0.5;
    }else if (A || a){ /* premultiplied by A^t, 1/2<X, A^t AX> - <X, A^t Y> */
        #pragma omp parallel for schedule(static) NUM_THREADS(V) \
            reduction(+:obj)
        for (vertex_t v = 0; v < V; v++){
            obj += X[v]*((real_t) 0.5*AX[v] - Y_(v));
        }
    }
    return obj;
}

TPL real_t PFDR_D1_QL1B::compute_h() const
{
    real_t obj = 0.0;
    if (l1_weights || homo_l1_weight){ /* ||x||_l1 */
        #pragma omp parallel for schedule(static) NUM_THREADS(V) \
             reduction(+:obj)
        for (vertex_t v = 0; v < V; v++){
            obj += L1_WEIGHTS_(v)*abs(X[v] - Yl1_(v));
        }
    }
    return obj;
}

TPL void PFDR_D1_QL1B::initialize_iterate()
/* initialize with coordinatewise pseudo-inverse pinv = <Av, Y>/||Av||^2,
 * or on l1 target if there is no quadratic part */
{
    if (!X){ X = (real_t*) malloc_check(sizeof(real_t)*V); }

    /* prevent useless computations */
    if (A && !Y){
        for (vertex_t v = 0; v < V; v++){ X[v] = 0.0; }
        return;
    }

    if (is_Gram(N)){ /* left-premultiplied by A^t case */
        if (A){
            index_t Vdiag = N == Gram_full() ? (index_t) V + 1 : 1;
            #pragma omp parallel for schedule(static) NUM_THREADS(V)
            for (vertex_t v = 0; v < V; v++){
                X[v] = A[Vdiag*v] > 0.0 ? Y[v]/A[Vdiag*v] : 0.0;
            }
        }else if (a){ /* identity */
            for (vertex_t v = 0; v < V; v++){ X[v] = Y_(v); }
        }else{ /* no quadratic part, initialize on l1 */
            for (vertex_t v = 0; v < V; v++){ X[v] = Yl1_(v); }
        }
    }else{ /* direct matricial case */
        #pragma omp parallel for schedule(static) NUM_THREADS(2*N*V, V)
        for (vertex_t v = 0; v < V; v++){
            const real_t* Av = A + N*v;
            real_t AvY = 0.0;
            real_t Av2 = 0.0;
            for (index_t n = 0; n < N; n++){
                AvY += Av[n]*Y[n];
                Av2 += Av[n]*Av[n];
            }
            X[v] = Av2 > 0.0 ? AvY/Av2 : 0.0;
        }
    }
}

TPL void PFDR_D1_QL1B::preconditioning(bool init)
{
    Pfdr_d1<real_t, vertex_t>::preconditioning(init);

    if (init){ /* reinitialize according to penalizations */
        vertex_t num_ops = (low_bnd || homo_low_bnd > -real_inf() ||
            upp_bnd || homo_upp_bnd < real_inf()) ? V : 1;
        #pragma omp parallel for schedule(static) NUM_THREADS(num_ops)
        for (vertex_t v = 0; v < V; v++){
            if (l1_weights || homo_l1_weight){ X[v] = Yl1_(v); } /* sparsity */
            if (low_bnd){
                if (X[v] < low_bnd[v]){ X[v] = low_bnd[v]; }
            }else if (homo_low_bnd > -real_inf()){
                if (X[v] < homo_low_bnd){ X[v] = homo_low_bnd; }
            }
            if (upp_bnd){
                if (X[v] > upp_bnd[v]){ X[v] = upp_bnd[v]; }
            }else if (homo_upp_bnd < real_inf()){
                if (X[v] > homo_upp_bnd){ X[v] = homo_upp_bnd; }
            }
        }
        initialize_auxiliary();
    }

    apply_A();
}

TPL void PFDR_D1_QL1B::main_iteration()
{
    Pfdr<real_t, vertex_t>::main_iteration();
    
    apply_A();
}

/**  instantiate for compilation  **/
#include <cstdint>
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template class Pfdr_d1_ql1b<float, int16_t>;
template class Pfdr_d1_ql1b<float, int32_t>;
template class Pfdr_d1_ql1b<double, int16_t>;
template class Pfdr_d1_ql1b<double, int32_t>;
#else
template class Pfdr_d1_ql1b<float, uint16_t>;
template class Pfdr_d1_ql1b<float, uint32_t>;
template class Pfdr_d1_ql1b<double, uint16_t>;
template class Pfdr_d1_ql1b<double, uint32_t>;
#endif
