/*=============================================================================
 * Base class for preconditioned forward Douglas-Rachford splitting algorithm
 *
 * minimize F: x -> f(x) + sum_i g_i(x) + h(x)
 *
 * where f has Lipschitz continuous gradient, and the proximity operator of
 * each g_i and of h are easy to compute
 *
 * can make use of diagonal preconditioning taking into account second-order
 * information and convenient splitting weights
 *
 * Parallel implementation with OpenMP API.
 * 
 * H. Raguet and L. Landrieu, Preconditioning of a Generalized Forward-Backward
 * Splitting and Application to Optimization on Graphs, SIAM Journal on Imaging
 * Sciences, 2015, 8, 2706-2739
 *
 * H. Raguet, A Note on the Forward-Douglas-Rachford Splitting for Monotone 
 * Inclusion and Convex Optimization, Optimization Letters, 2018, 1-24
 *
 * Hugo Raguet 2016, 2018
 *===========================================================================*/
#pragma once
#include "pcd_prox_split.hpp"

/* var_index_t is an integer type able to represent the size of the problem
 * without taking into account the dimension D */
template <typename real_t, typename var_index_t>
class Pfdr : public Pcd_prox<real_t>
{
public:

    using typename Pcd_prox<real_t>::index_t;

    /* shape of conditioning operators;
     * SCALAR for a scalar operator (upright shape);
     * MONODIM for a diagonal operator, constant along coordinates in case of
     * multidimensional data points (D > 1);
     * MULTIDIM for a diagonal operator, with possibly different values at
     * each coordinate of each data point */
    enum Condshape {SCALAR, MONODIM, MULTIDIM};

    /**  constructor, destructor  **/

    Pfdr(var_index_t size, index_t aux_size, const var_index_t* aux_idx,
        index_t D, Condshape gashape = MULTIDIM, Condshape wshape = MULTIDIM);

    /* delegation for monodimensional setting */
    Pfdr(var_index_t size, index_t aux_size, const var_index_t* aux_idx,
        Condshape gashape = MONODIM, Condshape wshape = MONODIM) : 
        Pfdr(size, aux_size, aux_idx, 1, gashape, wshape){};

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (monitoring arrays, etc.); it does free the rest (iterate, 
     * auxiliary variables etc.), but this can be prevented by copying the 
     * corresponding pointer member and set it to null before deleting */
	virtual ~Pfdr();

    using Pcd_prox<real_t>::set_name;

    /**  methods for manipulating parameters  **/

    void set_relaxation(real_t rho = 1.0);

    /* information on Lipschitzianity provided by user */
    void set_lipschitz_param(const real_t* L, real_t l, Condshape lshape);

    /* when Lipschitzianity of the gradient of f should be estimated:
     * USER is automatically set when Lipschitz info is set by the user, 
     * ONCE keeps info between reconditioning (saves computations and enables
     *  weighting of the iterate evolution by the Lipschitz metric, default),
     * EACH recomputes at each reconditioning (saves memory but do cannot
     *  weights iterate evolution by the Lipschitz metric) */
    enum Lipschcomput {USER, ONCE, EACH};

    /* overload for specifying computation of Lipschitz coefficients */
    void set_lipschitz_param(Lipschcomput lipschcomput = ONCE);

    /* retrieving and setting auxiliary variables might be usefull for warm
     * restart; NOTA:
     * 1) if not explicitely set by the user, memory pointed by these members
     * is allocated using malloc(), and thus should be deleted with free()
     * 2) they are free()'d by destructor, unless set to null beforehand
     * 3) if given, Z must be initialized or initialize_auxiliary() must be
     * called */
    void set_auxiliary(real_t* Z);

    /* initialize auxiliary to meaningful values */
    virtual void initialize_auxiliary(); 

    real_t* get_auxiliary();

protected:
    /**  structure  **/

    const var_index_t size; // number of data points; total dimension is D*size
    const index_t aux_size; // size of auxiliary variables
    const index_t D; // dimension of each data point
    /* auxiliary variable i corresponds to main coordinate aux_idx[i];
     * set to null for aux_idx[i] = i % size, in which case usually aux_size is
     * a multiple of size, and wshape is SCALAR */
    const var_index_t* const aux_idx; 

    /**  smooth functional f  **/

    /* information on Lipschitzianity of the gradient;
     * if not null, 'L' is an array of the size of the problem (same shape as
     * Ga, see gashape), representing a diagonal matrix such that
     * 0 < L and L^(-1/2) grad L^(-1/2) nonexpansive;
     * otherwise 'l' is a Lipschitz constant of the gradient */
    const real_t* L;
    real_t l;
    Lipschcomput lipschcomput; // see public declarations
    /* nonconst pointer to L in case Lipschitz metric must be computed */
    real_t* Lmut;

    /**  algorithmic parameters  **/

    real_t rho; /* relaxation parameter, 0 < rho < 2
     * 1 is a conservative value; 1.5 often speeds up convergence */    

    /**  metric and auxiliary variables  **/

    real_t *Ga; // preconditioning diagonal matrix
    real_t ga; // in case of a scalar matrix, only a step size
    /* auxiliary variables and splitting weights */
    real_t *Z, *W;
    real_t *Ga_grad_f; // store forward step
    /* additional auxiliary variables and weights of the size of the problem
     * this is useful to make W + Id_W = Id when the weights W cannot sum to 
     * identity because they are constrained by a certain metric;
     * the derived classes are responsible for allocating them */
    real_t *Z_Id, *Id_W;

    const Condshape gashape; // see public declarations
    const Condshape wshape; // see public declarations
    Condshape lshape; // see public declarations

    /**  preconditioning steps  **/

    /* compute Lipschitz metric of the gradient of f */
    virtual void compute_lipschitz_metric();

    /* initialize the metric Ga with (pseudo-)hessian of f; default to zero */
    virtual void compute_hess_f();

    /* pseudo-hessian of sum gi and splitting weights Wi;
     * usually based on local quadratic approximations */
    virtual void add_pseudo_hess_g() = 0; // add the result in Ga

    virtual void add_pseudo_hess_h(); // add the result in Ga, default to zero

    /* ensure sum Wi = Id */
    virtual void make_sum_Wi_Id();

    /**  forward-Douglas-Rachford steps  **/

    virtual void compute_Ga_grad_f(); // compute forward step in Ga_grad_f

    /* generalized forward-backward step over auxiliary Z; when this is called,
     * the forward step Ga_grad_f is supposed to contain 2*X - Ga*grad_f */
    virtual void compute_prox_GaW_g() = 0;

    virtual void compute_weighted_average();

    virtual void compute_prox_Ga_h(); // backward step over iterate X

    /**  for objective computation  **/

    virtual real_t compute_f() const; // smooth functional; default to zero

    /* sum of possibly nonsmooth functions sum_i g_i */
    virtual real_t compute_g() const = 0;

    /* can particularize one of the possibly nonsmooth functionals */
    virtual real_t compute_h() const; 

    /**  specialization of base virtual methods  **/

    /* compute preconditioning or reconditioning;
     * used also to allocate and initialize arrays */
    void preconditioning(bool init = false) override;

    void main_iteration() override;

    real_t compute_objective() const override;

    /* weight l2 norm by Lipschitz metric if available */
    real_t compute_evolution() const override;

    /**  type resolution for base template class members
     * https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
     **/
    using Pcd_prox<real_t>::eps;
    using Pcd_prox<real_t>::last_X;
    using Pcd_prox<real_t>::X;
    using Pcd_prox<real_t>::cond_min;
    using Pcd_prox<real_t>::malloc_check;
};
