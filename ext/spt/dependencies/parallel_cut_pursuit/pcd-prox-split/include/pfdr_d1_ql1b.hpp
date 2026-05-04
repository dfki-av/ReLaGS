/*=============================================================================
 * Minimize functional over a graph G = (V, E)
 *
 *        F(x) = 1/2 ||y - A x||^2 + ||x||_d1 + ||yl1 - x||_l1 + i_[m,M](x)
 *
 * where y in R^N, x in R^V, A in R^{N-by-|V|}, yl1 in R^V
 *      ||x||_d1 = sum_{uv in E} w_d1_uv |x_u - x_v|,
 *      ||x||_l1 = sum_{v  in V} w_l1_v |x_v|,
 * and the convex indicator
 *      i_[m,M](x) = infinity any x_v < m_v or x_v > M_v
 *                 = 0 otherwise;
 *
 * using preconditioned forward-Douglas-Rachford splitting algorithm.
 *
 * It is easy to introduce a SDP metric weighting the squared l2-norm
 * between y and A x. Indeed, if M is the matrix of such a SDP metric,
 *   ||y - A x||_M^2 = ||Dy - D A x||^2, with D = M^(1/2).
 * Thus, it is sufficient to call the method with Y <- Dy, and A <- D A.
 * Moreover, when A is the identity and M is diagonal (weighted square l2
 * distance between x and y), one should call on the precomposed version 
 * (N set to Gram_diag(), see below) with Y <- DDy = My and A <- D2 = M.
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
#include "pfdr_graph_d1.hpp"

/* vertex_t is an integer type able to represent the number of vertices */
template <typename real_t, typename vertex_t>
class Pfdr_d1_ql1b : public Pfdr_d1<real_t, vertex_t>
{
public:

    using typename Pfdr_d1<real_t, vertex_t>::index_t;

    /**  constructor, destructor  **/

    Pfdr_d1_ql1b(vertex_t V, index_t E, const vertex_t* edges);

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (adjacency graph structure given at construction, 
     * monitoring arrays, matrix and observation arrays); it does free the rest 
     * (iterate, auxiliary variables etc.), but this can be prevented by
     * copying the corresponding pointer member and set it to null before
     * deleting */
	~Pfdr_d1_ql1b();

    /**  methods for manipulating parameters  **/

    /* d1 weights (w_d1_uv above) can be set from base class Pfdr_d1 method
     * set_edge_weights() */

    /* for computation of information on Lipschitzianity;
     * NOEQUI only estimates the norm of ||A^t A||
     * the others attempts first at equilibrating (A^t A) with given method */
    enum Equilibration {NOEQUI, JACOBI, BUNCH};

    /* numeric constant for computing matrix norms with power methods, and 
     * equilibration method */
    void set_lipsch_norm_param(Equilibration lipsch_equi = JACOBI,
        real_t lipsch_norm_tol = 1e-3, int lipsch_norm_it_max = 100,
        int lipsch_norm_nb_init = 10);

    /* flag Gram matrices */
    static index_t Gram_full() { return 0; }
    static index_t Gram_diag() { return -1; }
    static bool is_Gram(index_t N)
        { return N == Gram_full() || N == Gram_diag(); }

    /* set the quadratic part, see members Y, N, A for details;
     * set Y to a null pointer for all zeros;
     * set A to a null pointer for identity matrix (set a to nonzero), or for
     * no quadratic part (set a to zero);
     * for a general scalar matrix, use the identity (A null, a zero) and scale
     * observations and penalizations accordingly */
    void set_quadratic(const real_t* Y, index_t N, const real_t* A = nullptr,
        real_t a = 1.0);

    /* overload for identity matrix */
    void set_quadratic(const real_t* Y){ set_quadratic(Y, Gram_diag()); }

    /* set l1_weights null for homogeneously equal to homo_l1_weight */
    void set_l1(const real_t* l1_weights = nullptr,
        real_t homo_l1_weight = 0.0, const real_t* Yl1 = nullptr);

    /* representing infinite values (has_infinity checked by constructor) */
    static real_t real_inf(){ return std::numeric_limits<real_t>::infinity(); }
    
    /* set bounds *_bnd to null for homogeneously equal to homo_*_bnd */
    void set_bounds(
        const real_t* low_bnd = nullptr, real_t homo_low_bnd = -real_inf(),
        const real_t* upp_bnd = nullptr, real_t homo_upp_bnd = real_inf());

    /* specialization, initialize with coordinatewise pseudo-inverse
     * pinv = <Av, Y>/||Av||^2, or on l1 target if there is no quadratic part
     * note that this is useful for preconditioning, but the iterate will be
     * reinitialized according to penalizations (zero if l1 enforces sparsity,
     * projection on bounds if any) before running */
    void initialize_iterate();

    using Pfdr<real_t, vertex_t>::initialize_auxiliary;

private:
    /**  quadratic problem  **/

    index_t N; /* number of observations;
     * if zero (function Gram_full()), matricial information is precomputed, 
     * that is, argument A is actually (A^t A), and argument Y is (A^t Y);
     * if negative one (or maximum value representable by matrix_index_t if
     * unsigned type, macro Gram_diag()), A is a diagonal matrix and only the
     * diagonal of (A^t A) = A^2 is given */

    const real_t* A; /* linear operator;
     * if N is positive, N-by-V array, column major format;
     * if N is zero (function Gram_full()), matrix (A^t A), V-by-V array,
     * column major format;
     * if N is negative one (function Gram_diag()), diagonal of (A^t A) = A^2,
     * array of length V, or null pointer for identity matrix (a = 1) or no
     * quadratic part (a = 0) */
    real_t a; 

    const real_t* Y; /* if N is positive, observations, array of length N;
     * otherwise, correlation of A with the observations (A^t Y), array of
     * length V; set to null for all zero */

    /* equilibration and norm of (A^t A) for Lipschitz metric */
    Equilibration lipsch_equi; // see public declaration
    /* parameters to compute operator norm with power method */
    real_t lipsch_norm_tol;
    int lipsch_norm_it_max;
    int lipsch_norm_nb_init;

    /* apply matrix A (or A^t A) to iterate;
     * if N is positive, compute residual (Y - A X) in R
     * otherwise, compute directly (A^t A X) in AX */
    void apply_A();

    /**  regularizations  **/

    /* observations for l1 fidelity, array of length V, set to null for zero */
    const real_t* Yl1;

    /* l1 penalization coefficients;
     * if 'l1_weights' is not null, array of length E;
     * otherwise homogeneously equal to 'homo_d1_weight' */
    const real_t* l1_weights;
    real_t homo_l1_weight;
    /* lower bounds of box constraints;
     * if 'low_bnd' is not null, array of length E;
     * otherwise homogeneously equal to 'homo_low_bnd' */
    const real_t* low_bnd;
    real_t homo_low_bnd;
    /* upper bounds of box constraints;
     * if 'upp_bnd' is not null, array of length E;
     * otherwise homogeneously equal to 'homo_upp_bnd' */
    const real_t* upp_bnd;
    real_t homo_upp_bnd;

    /**  preconditioning and auxiliary variables  **/

    real_t *R; // residual, array of length N, used only if N is positive
    real_t* &AX = Pfdr<real_t, vertex_t>::Ga_grad_f; // store product AX

    /**  specialization of base virtual methods  **/

    /* approximate diagonal hessian of quadratic functional */
    void compute_hess_f() override;

    /* l1 contribution */
    void add_pseudo_hess_h() override;

    /* compute Lipschitz metric of quadratic functional */
    void compute_lipschitz_metric() override;

    /* compute the gradient of the quadratic functional in Pfdr::Ga_grad_f */
    void compute_Ga_grad_f() override; // assume apply_A() have been called

    void compute_prox_Ga_h() override; // backward step over iterate X

    /* quadratic functional; in the precomputed A^t A version, 
     * a constant 1/2||Y||^2 is omited */
    real_t compute_f() const override; 

    real_t compute_h() const override; // l1 norm

    void preconditioning(bool init) override; // add some precomputations

    void main_iteration() override; // add application of matrix A

    /**  type resolution for base template class members
     * https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
     **/
    using Pfdr_d1<real_t, vertex_t>::V;
    using Pfdr_d1<real_t, vertex_t>::E;
    using Pfdr<real_t, vertex_t>::Ga_grad_f;
    using Pfdr<real_t, vertex_t>::Ga;
    using Pfdr<real_t, vertex_t>::L;
    using Pfdr<real_t, vertex_t>::l;
    using Pfdr<real_t, vertex_t>::lshape;
    using Pfdr<real_t, vertex_t>::SCALAR;
    using Pfdr<real_t, vertex_t>::MONODIM;
    using Pfdr<real_t, vertex_t>::lipschcomput;
    using Pfdr<real_t, vertex_t>::Lmut;
    using Pfdr<real_t, vertex_t>::ONCE;
    using Pfdr<real_t, vertex_t>::EACH;
    using Pcd_prox<real_t>::X;
    using Pcd_prox<real_t>::last_X;
    using Pcd_prox<real_t>::cond_min;
    using Pcd_prox<real_t>::dif_tol;
    using Pcd_prox<real_t>::dif_rcd;
    using Pcd_prox<real_t>::iterate_evolution;
    using Pcd_prox<real_t>::eps;
    using Pcd_prox<real_t>::malloc_check;
};
