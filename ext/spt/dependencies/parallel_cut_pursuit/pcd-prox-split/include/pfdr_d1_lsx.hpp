/*=============================================================================
 * Minimize functional over a graph G = (V, E)
 *
 *        F(x) = f(x) + ||x||_d1 + i_{simplex}(x)
 *
 * where for each vertex, x_v is a D-dimensional vector,
 *       f(x) = sum_{v in V} f_v(x_v) is a separable data-fidelity loss
 *       ||x||_d1 = sum_{uv in E} w_d1_uv (sum_d w_d1_d |x_ud - x_vd|),
 * and i_{simplex} is the standard D-simplex constraint over each vertex,
 *     i_{simplex} = 0 for all v, (for all d, x_vd >= 0) and sum_d x_vd = 1,
 *                 = infinity otherwise;
 *
 * using preconditioned forward-Douglas-Rachford splitting algorithm.
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
class Pfdr_d1_lsx : public Pfdr_d1<real_t, vertex_t>
{
public:

    using typename Pfdr_d1<real_t, vertex_t>::index_t;

    /**  constructor, destructor  **/

    Pfdr_d1_lsx(vertex_t V, index_t E, const vertex_t* edges, real_t loss,
        index_t D, const real_t* Y, const real_t* d11_metric = nullptr);

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (adjacency graph structure given at construction, 
     * monitoring arrays, matrix and observation arrays); it does free the rest 
     * (iterate, auxiliary variables etc.), but this can be prevented by
     * copying the corresponding pointer member and set it to null before
     * deleting */
	~Pfdr_d1_lsx();

    /**  methods for manipulating parameters  **/

    /* d11 weights w_d1_uv can be set from base class Pfdr_d1 method
     * set_edge_weights(); however, d11_metric w_d1_d above must be known
     * at construction */
 
    void initialize_iterate() override; // initialize on simplex based on Y

    /* specific losses */
    static real_t linear_loss() { return 0.0; }
    static real_t quadratic_loss() { return 1.0; }

    /* warning: the first parameter loss can only be used to change the 
     * smoothing parameter of a Kullback-Leibler loss; for changing from one
     * loss type to another, create a new instance; Y is changed only if the
     * argument is not null */
    void set_loss(real_t loss, const real_t* Y = nullptr,
        const real_t* loss_weights = nullptr);

    /* overload for changing only loss_weights */
    void set_loss(const real_t* loss_weights)
        { set_loss(loss, nullptr, loss_weights); }

private:
    /**  separable loss term  **/

    /* 0 for linear (function linear_loss())
     *     f(x) = - <x, y>_w ,
     * with <x, y>_w = sum_{v,d} w_v y_{v,d} x_{v,d} ;
     *
     * 1 for quadratic (function quadratic_loss())
     *     f(x) = 1/2 ||y - x||_{l2,w}^2 ,
     * with  ||y - x||_{l2,w}^2 = sum_{v,d} w_v (y_{v,d} - x_{v,d})^2 ;
     *
     * 0 < loss < 1 for smoothed (weighted) Kullback-Leibler divergence
     * (cross-entropy)
     *     f(x) = sum_v w_v KLs(x_v, y_v),
     * with KLs(y_v, x_v) = KL(s u + (1 - s) y_v ,  s u + (1 - s) x_v), where
     *     KL is the regular Kullback-Leibler divergence,
     *     u is the uniform discrete distribution over {1,...,D}, and
     *     s = loss is the smoothing parameter ;
     * it yields
     *
     *     KLs(y_v, x_v) = H(s u + (1 - s) y_v)
     *         - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) x_{v,d}) ,
     *
     * where H is the entropy, that is H(s u + (1 - s) y_v)
     *       = - sum_d (s/D + (1 - s) y_{v,d}) log(s/D + (1 - s) y_{v,d}) ;
     * note that the choosen order of the arguments in the Kullback-Leibler
     * does not favor the entropy of x (H(s u + (1 - s) y_v) is a constant),
     * hence this loss is actually equivalent to cross-entropy */
    real_t loss;

    /* observations, D-by-V array, column major format;
     * must lie on the simplex */
    const real_t* Y; 

    /* weights on vertices; array of length V, or null for no weights */
    const real_t *loss_weights;

    /**  preconditioning and auxiliary variables  **/

    real_t* W_Ga_Y; // precompute some first-order information

    /**  specialization of base virtual methods  **/

    /* hessian of separable loss */
    void compute_hess_f() override;

    /* compute Lipschitz metric of the loss */
    void compute_lipschitz_metric() override;

    /* compute the gradient of the separable functional in Pfdr::Ga_grad_f */
    void compute_Ga_grad_f() override;

    void compute_prox_Ga_h() override; // backward step over iterate X

    real_t compute_f() const override; // separable loss

    void preconditioning(bool init) override; // add some precomputations

    /* relative iterate evolution in l1 norm */
    real_t compute_evolution() const override;

    /**  type resolution for base template class members
     * https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
     **/
    using Pfdr_d1<real_t, vertex_t>::V;
    using Pfdr_d1<real_t, vertex_t>::E;
    using Pfdr_d1<real_t, vertex_t>::D11;
    using Pfdr<real_t, vertex_t>::Ga_grad_f;
    using Pfdr<real_t, vertex_t>::Ga;
    using Pfdr<real_t, vertex_t>::L;
    using Pfdr<real_t, vertex_t>::l;
    using Pfdr<real_t, vertex_t>::lshape;
    using Pfdr<real_t, vertex_t>::gashape;
    using Pfdr<real_t, vertex_t>::SCALAR;
    using Pfdr<real_t, vertex_t>::MONODIM;
    using Pfdr<real_t, vertex_t>::MULTIDIM;
    using Pfdr<real_t, vertex_t>::lipschcomput;
    using Pfdr<real_t, vertex_t>::Lmut;
    using Pfdr<real_t, vertex_t>::ONCE;
    using Pfdr<real_t, vertex_t>::EACH;
    using Pfdr<real_t, vertex_t>::D;
    using Pcd_prox<real_t>::X;
    using Pcd_prox<real_t>::last_X;
    using Pcd_prox<real_t>::cond_min;
    using Pcd_prox<real_t>::eps;
    using Pcd_prox<real_t>::malloc_check;
};
