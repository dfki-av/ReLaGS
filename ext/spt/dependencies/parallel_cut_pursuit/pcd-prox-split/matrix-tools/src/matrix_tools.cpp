/*=============================================================================
 * tools for manipulating real matrices
 *
 * Hugo Raguet 2016, 2018
 *===========================================================================*/
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "omp_num_threads.hpp"
#include "matrix_tools.hpp"

/* constants of the correct type */
#define ZERO ((real_t) 0.0)
#define ONE ((real_t) 1.0)

#define MT matrix_tools // shorthand for the namespace

using namespace std;
typedef MT::index_t index_t;

static const int HALF_RAND_MAX = (RAND_MAX/2 + 1);
static const double HALF_RAND_MAX_D = (double) HALF_RAND_MAX;

template <typename real_t>
static real_t compute_norm(const real_t *X, const index_t N)
{
    real_t norm = ZERO;
    for (index_t n = 0; n < N; n++){ norm += X[n]*X[n]; }
    return sqrt(norm);
}

template <typename real_t>
static void normalize_and_apply_matrix(const real_t* A, real_t* X, real_t* AX,
    const real_t* D, real_t norm, bool sym, index_t M, index_t N)
{
    if (sym){
        if (D){ for (index_t n = 0; n < N; n++){ AX[n] = X[n]*D[n]/norm; } }
        else{ for (index_t n = 0; n < N; n++){ AX[n] = X[n]/norm; } }
    }else{
        if (D){ for (index_t n = 0; n < N; n++){ X[n] *= D[n]/norm; } }
        else{ for (index_t n = 0; n < N; n++){ X[n] /= norm; } }
        /* apply A */
        for (index_t m = 0; m < M; m++){
            AX[m] = ZERO;
            index_t p = m;
            for (index_t n = 0; n < N; n++){
                AX[m] += A[p]*X[n];
                p += M;
            }
        }
    }
    /* apply A^t or AA */
    const real_t *An = A;
    for (index_t n = 0; n < N; n++){
        X[n] = ZERO;
        for (index_t m = 0; m < M; m++){ X[n] += An[m]*AX[m]; }
        An += M;
    }
    if (D){ for (index_t n = 0; n < N; n++){ X[n] *= D[n]; } }
}

template <typename real_t>
real_t MT::operator_norm_matrix(index_t M, index_t N, const real_t* A,
    const real_t* D, real_t tol, int it_max, int nb_init, bool verbose)
{
    real_t* AA = nullptr;
    bool sym = false;

    /**  preprocessing  **/
    const index_t P = (M < N) ? M : N;
    const int i_tot = nb_init*it_max;
    if (P == Gram()){
        sym = true;
        M = (M > N) ? M : N;
        N = M;
    }else if (2*M*N*i_tot > (M*N*P + P*P*i_tot)){
        sym = true;
        /* compute symmetrization */
        AA = (real_t*) malloc(sizeof(real_t)*P*P);
        if (!AA){
            cerr << "Operator norm matrix: not enough memory." << endl;
            exit(EXIT_FAILURE);
        }
        for (index_t p = 0; p < P*P; p++){ AA[p] = ZERO; }
        if (M < N){ /* A A^t is smaller */
            /* fill upper triangular part (from lower triangular products) */
            #pragma omp parallel for schedule(static) NUM_THREADS(M*N*P/2, P)
            for (index_t p = 0; p < P; p++){
                const real_t *Ap = A + p; // run along p-th row of A
                const real_t *An = A; // n-th row of A^t
                real_t *AAp = AA + P*p; // p-th column of AA
                real_t ApnDn2;
                for (index_t n = 0; n < N; n++){
                    if (D){
                        ApnDn2 = (*Ap)*D[n]*D[n];
                        for (index_t m = 0; m <= p; m++){
                            AAp[m] += ApnDn2*An[m];
                        }
                    }else{
                        for (index_t m = 0; m <= p; m++){
                            AAp[m] += (*Ap)*An[m];
                        }
                    }
                    An += M;
                    Ap += M;
                }
            }
        }else{ /* A^t A is smaller */
            /* fill upper triangular part */
            #pragma omp parallel for schedule(static) NUM_THREADS(M*N*P/2, P)
            for (index_t p = 0; p < P; p++){
                const real_t *Ap = A + M*p; // p-th column of A 
                const real_t *An = A; // run along n-th column of A
                real_t *AAp = AA + P*p; // p-th column of AA
                for (index_t n = 0; n <= p; n++){
                    AAp[n] = ZERO;
                    for (index_t m = 0; m < M; m++){
                        AAp[n] += (*(An++))*Ap[m];
                    }
                    if (D){ AAp[n] *= D[p]*D[n]; }
                }
            }
        }
        /* fill lower triangular part */
        #pragma omp parallel for schedule(static) NUM_THREADS(P, P - 1)
        for (index_t p = 0; p < P - 1; p++){
            real_t *AAp = AA + P*p;
            index_t m = (P + 1)*p + P;
            for (index_t n = p + 1; n < P; n++){
                AAp[n] = AA[m];
                m += P;
            }
        }
        M = P;
        N = P;
        A = AA;
        /* D has been taken into account in A D^2 A^t or D A^t A D */
        if (D){ D = nullptr; }
    }

    /**  power method  **/
    const int num_procs = omp_get_num_procs();
    nb_init = (1 + (nb_init - 1)/num_procs)*num_procs;
    if (verbose){
        cout << "compute matrix operator norm on " << nb_init << " random "
            << "initializations, over " << num_procs << " parallel threads... "
            << flush;
    }

    real_t matrix_norm2 = ZERO;
    #if defined _OPENMP && _OPENMP >= 201107
    /* dumb MSVC still does not support max reduction (OpenMP 3.1) in 2020 */
    #pragma omp parallel reduction(max:matrix_norm2) num_threads(num_procs)
    #endif
    {
    unsigned int rand_seed = time(nullptr) + omp_get_thread_num();
    real_t* X = (real_t*) malloc(sizeof(real_t)*N);
    real_t* AX = (real_t*) malloc(sizeof(real_t)*M);
    #pragma omp for schedule(static)
    for (int init = 0; init < nb_init; init++){
        /* random initialization */
        for (index_t n = 0; n < N; n++){
            /* very crude uniform distribution on [-1,1] */
            #ifdef _MSC_VER /* dumb MSVC does not have rand_r equivalent */
            X[n] = (rand() - HALF_RAND_MAX)/HALF_RAND_MAX_D;
            #else
            X[n] = (rand_r(&rand_seed) - HALF_RAND_MAX)/HALF_RAND_MAX_D;
            #endif
        }
        real_t norm = compute_norm(X, N);
        normalize_and_apply_matrix(A, X, AX, D, norm, sym, M, N);
        norm = compute_norm(X, N);
        /* iterate */
        if (norm > ZERO){
            for (int it = 0; it < it_max; it++){
                normalize_and_apply_matrix(A, X, AX, D, norm, sym, M, N);
                real_t norm_ = compute_norm(X, N);
                if ((norm_ - norm)/norm < tol){ break; }
                norm = norm_;
            }
        }
        if (norm > matrix_norm2){ matrix_norm2 = norm; }
    }
    free(X);
    free(AX);
    } // end pragma omp parallel
    if (verbose){ cout << "done." << endl; }
    free(AA);
    return matrix_norm2;
}

template <typename real_t>
void MT::symmetric_equilibration_jacobi(index_t M, index_t N, const real_t* A,
    real_t* D)
{
    if (M == Gram()){ /* premultiplied by A^t */
        #pragma omp parallel for schedule(static) NUM_THREADS(N)
        for (index_t n = 0; n < N; n++){ D[n] = ONE/sqrt(A[n*(N + 1)]); }
    }else{
        #pragma omp parallel for schedule(static) NUM_THREADS(M*N, N)
        for (index_t n = 0; n < N; n++){
            const real_t *An = A + M*n;
            D[n] = ZERO;
            for (index_t m = 0; m < M; m++){ D[n] += An[m]*An[m]; }
            D[n] = ONE/sqrt(D[n]);
        }
    }
}

template <typename real_t>
void MT::symmetric_equilibration_bunch(index_t M, index_t N, const real_t* A,
    real_t* D)
{
    if (M == Gram()){ /* premultiplied by A^t */
        D[0] = ONE/sqrt(A[0]);
    }else{
        real_t A1A1 = ZERO;
        #pragma omp parallel for NUM_THREADS(M) reduction(+:A1A1)
        for (index_t m = 0; m < M; m++){ A1A1 += A[m]*A[m]; }
        D[0] = ONE/sqrt(A1A1);
    }

    for (index_t i = 1; i < N; i++){ 
        real_t invDi = ZERO;
        if (M == Gram()){
            #if defined _OPENMP && _OPENMP >= 201107
            /* dumb MSVC still does not support max reduction in 2020 */
            #pragma omp parallel for NUM_THREADS(i + 1) reduction(max:invDi)
            #endif
            for (index_t j = 0; j <= i; j++){
                real_t DjAiAj = A[i + N*j];
                DjAiAj = (j < i) ? abs(DjAiAj)*D[j] : sqrt(DjAiAj);
                if (DjAiAj > invDi){ invDi = DjAiAj; }
            }
        }else{
            const real_t* Ai = A + M*i;
            #if defined _OPENMP && _OPENMP >= 201107
            /* dumb MSVC still does not support max reduction in 2020 */
            #pragma omp parallel for NUM_THREADS((i + 1)*M, i + 1) \
                reduction(max:invDi)
            #endif
            for (index_t j = 0; j <= i; j++){
                real_t DjAiAj = ZERO;
                const real_t* Aj = A + M*j;
                for (index_t m = 0; m < M; m++){
                    DjAiAj += Ai[m]*Aj[m];
                }
                DjAiAj = (j < i) ? abs(DjAiAj)*D[j] : sqrt(DjAiAj);
                if (DjAiAj > invDi){ invDi = DjAiAj; }
            }
        }
        D[i] = ONE/invDi;
    }
}

/* instantiate for compilation */
template float MT::operator_norm_matrix<float>(index_t, index_t,
    const float*, const float*, float, int, int, bool);

template double MT::operator_norm_matrix<double>(index_t, index_t,
    const double*, const double*, double, int, int, bool);

template void MT::symmetric_equilibration_jacobi<float>(index_t M,
    index_t N, const float* A, float* L);

template void MT::symmetric_equilibration_jacobi<double>(index_t M,
    index_t N, const double* A, double* L);

template void MT::symmetric_equilibration_bunch<float>(index_t M,
    index_t N, const float* A, float* L);

template void MT::symmetric_equilibration_bunch<double>(index_t M,
    index_t N, const double* A, double* L);
