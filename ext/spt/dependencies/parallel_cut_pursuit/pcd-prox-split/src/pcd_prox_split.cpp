/*=============================================================================
 * Hugo Raguet 2016, 2018
 *===========================================================================*/
#include <cmath>
#include "omp_num_threads.hpp"
#include "pcd_prox_split.hpp"

#define TPL template <typename real_t>
#define PCD_PROX Pcd_prox<real_t>

using namespace std;

TPL PCD_PROX::Pcd_prox(index_t size) : size(size)
{
    name = "Preconditioned proximal splitting algorithm";
    objective_values = iterate_evolution = nullptr;
    cond_min = 1e-2;
    dif_rcd = 0;
    dif_tol = 1e-4;
    dif_it = 32;
    it_max = 1e3;
    verbose = 1e2;
    eps = numeric_limits<real_t>::epsilon();
    X = nullptr;
}

TPL PCD_PROX::~Pcd_prox(){ free(X); }

TPL void PCD_PROX::set_name(const char* name)
{ this->name = name; }

TPL void PCD_PROX::set_monitoring_arrays(real_t* objective_values,
    real_t* iterate_evolution)
{
    this->objective_values = objective_values;
    this->iterate_evolution = iterate_evolution;
}

TPL void PCD_PROX::set_conditioning_param(real_t cond_min, real_t dif_rcd)
{
    this->cond_min = cond_min;
    this->dif_rcd = dif_rcd;
}

TPL void PCD_PROX::set_algo_param(real_t dif_tol, int dif_it, int it_max,
    int verbose, real_t eps)
{
    if (dif_it == 0 && dif_tol > 0.0){ /* roughly sqrt(it_max) */
        dif_it = 1;
        while (dif_it*dif_it < it_max){ dif_it *= 2; }
    }
    this->dif_tol = dif_tol;
    this->dif_it = dif_it;
    this->it_max = it_max;
    this->verbose = verbose;
    this->eps = eps;
}

TPL void PCD_PROX::set_iterate(real_t* X){ this->X = X; }

TPL real_t* PCD_PROX::get_iterate(){ return this->X; }

TPL void PCD_PROX::initialize_iterate()
{
    if (!X){ X = (real_t*) malloc_check(sizeof(real_t)*size); }
    for (index_t i = 0; i < size; i++){ X[i] = 0.0; }
}

TPL void PCD_PROX::preconditioning(bool init)
{ if (init && !X){ initialize_iterate(); } }

TPL int PCD_PROX::precond_proximal_splitting(bool init)
{
    int it = 0;
    real_t dif = (dif_tol > 1.0) ? dif_tol : 1.0;
    if (dif_rcd > dif){ dif = dif_rcd; }
    int it_verb, it_dif;

    if (verbose){
        cout << name << ":" << endl;
        it_verb = 0;
    }

    if (verbose){ cout << "Preconditioning... " << flush; }
    preconditioning(init);
    if (verbose){ cout << "done." << endl; }

    if (init && objective_values){ objective_values[0] = compute_objective(); }

    if (dif_tol > 0.0 || dif_rcd > 0.0 || iterate_evolution){
        last_X = (real_t*) malloc_check(sizeof(real_t)*size);
        for (index_t i = 0; i < size; i++){ last_X[i] = X[i]; }
        it_dif = 0;
    }

    while (it < it_max && dif >= dif_tol){

        if (verbose && it_verb == verbose){
            print_progress(it, dif);
            it_verb = 0;
        }

        if (dif < dif_rcd){
            if (verbose){
                print_progress(it, dif);
                cout << "\nReconditioning... " << flush;
            }
            preconditioning();
            dif_rcd /= 10.0;
            if (verbose){ cout << "done." << endl; }
        }

        main_iteration();

        it++; it_verb++; it_dif++;

        if (iterate_evolution ||
            ((dif_tol > 0.0 || dif_rcd > 0.0) && it_dif == dif_it)){
            dif = compute_evolution();
            for (index_t i = 0; i < size; i++){ last_X[i] = X[i]; }
            if (iterate_evolution){ iterate_evolution[it] = dif; }
            it_dif = 0;
        }

        if (objective_values){ objective_values[it] = compute_objective(); }

    }
    
    if (verbose){ print_progress(it, dif); cout << endl; }
    
    if (dif_tol > 0.0 || dif_rcd > 0.0 || iterate_evolution){ free(last_X); }

    return it;
}

TPL void PCD_PROX::print_progress(int it, real_t dif)
{
    cout << "\r" << "iteration " << it << " (max. " << it_max << "); ";
    if (dif_tol > 0.0 || dif_rcd > 0.0){
        cout.precision(2);
        cout << scientific << "iterate evolution " << dif <<  " (recond. "
            << dif_rcd << ", tol. " << dif_tol << ")";
    }
    cout << flush;
}

TPL real_t PCD_PROX::compute_evolution() const
/* by default, relative evolution in Euclidean norm */
{
    real_t dif = 0.0;
    real_t amp = 0.0;
    #pragma omp parallel for schedule(static) NUM_THREADS(size) \
        reduction(+:dif, amp)
    for (index_t i = 0; i < size; i++){
        real_t d = last_X[i] - X[i];
        dif += d*d;
        amp += X[i]*X[i];
    }
    return sqrt(amp) > eps ? sqrt(dif/amp) : sqrt(dif)/eps;
}

/**  instantiate for compilation  **/
template class Pcd_prox<double>;
template class Pcd_prox<float>;
