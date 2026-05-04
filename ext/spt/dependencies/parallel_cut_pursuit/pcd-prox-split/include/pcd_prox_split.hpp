/*=============================================================================
 * Base class for preconditioned proximal splitting algorithm
 *
 * Parallel implementation with OpenMP API.
 * 
 * H. Raguet and L. Landrieu, Preconditioning of a Generalized Forward-Backward
 * Splitting and Application to Optimization on Graphs, SIAM Journal on Imaging
 * Sciences, 2015, 8, 2706-2739
 *
 * Hugo Raguet 2016, 2018
 *===========================================================================*/
#pragma once
#include <cstdlib>
#include <iostream>
#include <limits>

template <typename real_t> class Pcd_prox
{
public:

    #if defined _OPENMP && _OPENMP < 200805
    /* use of unsigned counter in parallel loops requires OpenMP 3.0;
     * although published in 2008, MSVC still does not support it as of 2020 */
    typedef long int index_t;
    #else
    typedef size_t index_t;
    #endif

    /**  constructor, destructor  **/

    Pcd_prox(index_t size);

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (monitoring arrays, etc.); it does free the rest (iterate, 
     * auxiliary variables etc.), but this can be prevented by copying the 
     * corresponding pointer member and set it to null before deleting */
	virtual ~Pcd_prox();

    /**  manipulate private members pointers and values  **/

    void set_name(const char * name =
        "Preconditioned proximal splitting algorithm");

    void set_monitoring_arrays(real_t* objective_values = nullptr,
        real_t* iterate_evolution = nullptr);

    void set_conditioning_param(real_t cond_min = 1e-2, real_t dif_rcd = 1e-3);

    void set_algo_param(real_t dif_tol, int dif_it, int it_max, int verbose,
        real_t eps);
    /* overload for allowing a function call for default parameter 'eps' */
    void set_algo_param(real_t dif_tol = 1e-4, int dif_it = 0,
        int it_max = 1e3, int verbose = 1e2)
    {
        set_algo_param(dif_tol, dif_it, it_max, verbose,
            std::numeric_limits<real_t>::epsilon());
    }

    /* NOTA:
     * 1) if not explicitely set by the user, memory pointed by these members
     * is allocated using malloc(), and thus should be deleted with free()
     * 2) they are free()'d by destructor, unless set to null beforehand
     * 3) if given, X must be initialized or initialize_iterate() must be
     * called */
    void set_iterate(real_t* X);

    virtual void initialize_iterate(); // initialize X to a meaningful value

    real_t* get_iterate();

    /**  solve the main problem  **/

    /* return number of iterations */
    int precond_proximal_splitting(bool init = true);

protected:

    /**  parameters  **/
    index_t size; // dimension of the problem

    /* stability of preconditioning; 0 < cond_min < 1;
     * corresponds roughly to the minimum ratio between different
     * directions of the descent metric; 1e-2 is a typical value;
     * smaller values might enhance preconditioning but might make it unstable;
     * increase this value if iteration steps seem to get too small */
    real_t cond_min;
    /* reconditioning criterion on iterate evolution;
     * a reconditioning is performed if relative changes of the iterate
     * warning: reconditioning might temporarily draw minimizer away from 
     * solution; monitor objective value when using reconditioning */ 
    real_t dif_rcd;

    /* stopping criterion on iterate evolution;
     * algorithm stops if relative change of the iterate is less */
    real_t dif_tol;
    int dif_it; // number of iterations between iterates for evolution measure
    int it_max; // maximum number of iterations

    /* information on the progress;
     * if nonzero, printed every 'verbose' iterations */
    int verbose = 1e2;

    real_t eps; // characteristic precision 

    /**  arrays  **/
    real_t *X, *last_X; // iterate, previous iterate

    /**  monitoring  **/

    /* records the values of the objective functional;
     * if not NULL, array of length it_max + 1 */
    real_t* objective_values;
    /* records the iterate relative evolution;
     * if not NULL, array of length it_max */
    real_t* iterate_evolution;

    /**  methods  **/

    /* compute preconditioning or reconditioning;
     * used also to allocate and initialize arrays */
    virtual void preconditioning(bool init = false);

    /* iteration of the proximal splitting algorithm */
    virtual void main_iteration() = 0;

    /* compute relative iterate evolution and store current iterate in last_X;
     * by default, relative evolution in Euclidean norm */
    virtual real_t compute_evolution() const;

    /* compute objective functional */
    virtual real_t compute_objective() const = 0;

    /* allocate memory and fail with error message if not successful */
    static void* malloc_check(index_t size)
    {
        void *ptr = malloc(size);
        if (ptr == nullptr){
            std::cerr << "Preconditioned proximal splitting: "
                "not enough memory." << std::endl;
            exit(EXIT_FAILURE); /* should we free all pointers manually? */
        }
        return ptr;
    }

private:

    const char* name;

     /**  methods  **/

    void print_progress(int it, real_t dif);
};
