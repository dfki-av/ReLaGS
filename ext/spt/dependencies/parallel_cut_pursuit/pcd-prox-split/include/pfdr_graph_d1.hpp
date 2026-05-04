/*=============================================================================
 * Derived class for preconditioned forward-Douglas–Rachford algorithm
 * minimizing functionals involving the graph total variation:
 *
 * minimize functional F defined over a graph G = (V, E)
 *
 * F: R^{D-by-V} -> R
 *        x      -> f(x) + ||x||_d1p + h(x)
 *
 * where
 * 
 *      ||x||_d1p = sum_{uv in E} w_uv ||x_u - x_v||_{M,p} ,
 *
 * where M in R^{D-by-D} can be a diagonal metric and p can be 1 or 2,
 *  ||x_v||_{M,1} = sum_d m_d |x_v|  (weighted l1-norm) or
 *  ||x_v||_{M,2} = sqrt(sum_d m_d x_v^2)  (weighted l2-norm)
 *
 * and where f has Lipschitz continuous gradient, and the proximal operator of
 * h is easy to compute, using preconditioned forward-Douglas-Rachford
 * splitting algorithm.
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
 * Hugo Raguet 2016, 2018, 2023
 *===========================================================================*/
#pragma once
#include "pcd_fwd_doug_rach.hpp"

/* vertex_t is an integer type able to represent the number of vertices */
template <typename real_t, typename vertex_t>
class Pfdr_d1 : public Pfdr<real_t, vertex_t>
{
public:

    using typename Pfdr<real_t, vertex_t>::index_t;

    /* for multidimensional data, type of graph total variation, which is
     * nothing but the sum of norms of finite differences over the edges:
     * d1,1 is the sum of l1 norms;
     * d1,2 is the sum of l2 norms */
    enum D1p {D11, D12};

    /* use SCALAR for null hessian */

    /* reuse the Conshape type (see pcd_fwd_doug_rach.hpp) for the shape of
     * various metrics and preconditioners;
     * for the Hessian matrix, SCALAR means null hessian */
    using typename Pfdr<real_t, vertex_t>::Condshape;
    using Pfdr<real_t, vertex_t>::SCALAR;
    using Pfdr<real_t, vertex_t>::MONODIM;
    using Pfdr<real_t, vertex_t>::MULTIDIM;

    /**  constructor, destructor  **/

    /* concerning the parameters pfdr::gashape, pfdr::wshape, wd1shape,
     * thd1shape, extensive explanations are given at the end of the file.
     * Resulting combinations are as follows.
     *
     * d1p H(f+h) coor_w | Ga (1) W (2,5) Wd1 (3)  Thd1 (4) | observations
     * -------------------------------------------------------------------
     * D11 ZERO   NO     | MONO   MONO    SCAL=1/2 MONO     | (1)(2)(3)
     * D11 ZERO   YES    | MULTI  MONO    SCAL=1/2 MONO     |    (2)(3)(4)
     * D11 MONO   NO     | MONO   MONO    MONO     MONO     | (1)(2)
     * D11 MONO   YES    | MULTI  MONO    MULTI    MULTI    |    (2)
     * D11 MULTI  NO     | MULTI  MONO    MULTI    MULTI    |    (2)
     * D11 MULTI  YES    | MULTI  MONO    MULTI    MULTI    |    (2)
     * D12 ZERO   NO     | MONO   MONO    SCAL=1/2 MONO     | (1) (5.1)
     * D12 ZERO   YES    | MULTI  MONO    SCAL=1/2 MONO     |     (5.1)
     * D12 MONO   NO     | MONO   MONO    MONO     MONO     | (1) (5.2)
     * D12 MONO   YES    | MULTI  MULTI   MONO     MONO     |   (5.3)
     * D12 MULTI  NO     | MULTI  MULTI   MONO     MONO     |   (5.3)
     * D12 MULTI  YES    | MULTI  MULTI   MONO     MONO     |   (5.3)
     *
     * note that the last three configurations resort to additional
     * auxiliary variables and weights Z_Id and Id_W (only for D > 1) */
    Pfdr_d1(vertex_t V, index_t E, const vertex_t* edges, index_t D,
        D1p d1p = D12, const real_t* d1p_metric = nullptr,
        Condshape hess_f_h_shape = MULTIDIM);

    /* delegation for monodimensional setting */
    Pfdr_d1(vertex_t V, index_t E, const vertex_t* edges,
        Condshape hess_f_h_shape = MONODIM) :
        Pfdr_d1(V, E, edges, 1, D11, nullptr, hess_f_h_shape){}

    /* the destructor does not free pointers which are supposed to be provided 
     * by the user (adjacency graph structure, monitoring arrays, etc.); it 
     * does free the rest (iterate, auxiliary variables etc.), but this can be
     * prevented by copying the corresponding pointer member and set it to null
     * before deleting */
	virtual ~Pfdr_d1();

    /**  methods for manipulating parameters  **/

    /* warning: d1p_metric cannot be changed from null to varying weights or
     * vice versa; a new instance should be created instead */
    void set_edge_weights(const real_t* edge_weights, real_t homo_edge_weight,
        const real_t* d1p_metric);

    /* overload making arguments optional */
    void set_edge_weights(const real_t* edge_weights = nullptr,
        real_t homo_edge_weight = 1.0)
    { set_edge_weights(edge_weights, homo_edge_weight, this->d1p_metric); }

protected:
    /**  graph  **/

    /* number of vertices and of (undirected) edges */
    const vertex_t& V = Pfdr<real_t, vertex_t>::size;
    const index_t E;

    /**  specialization of base virtual methods  **/

    /* specialization adding precomputations */
    void preconditioning(bool init) override; 

    /* generalized forward-backward step over auxiliary Z */
    void compute_prox_GaW_g() override;

    /* add pseudo-hessian and splitting weights of graph total variation */
    void add_pseudo_hess_g() override;

    /* ensure sum Wi = Id */
    virtual void make_sum_Wi_Id() override;

    real_t compute_g() const override; // sum_i g_i

    /**  type resolution for base template class members
     * https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
     **/
    using Pfdr<real_t, vertex_t>::gashape;
    using Pfdr<real_t, vertex_t>::wshape;
    using Pfdr<real_t, vertex_t>::rho;
    using Pfdr<real_t, vertex_t>::Ga_grad_f;
    using Pfdr<real_t, vertex_t>::Ga;
    using Pfdr<real_t, vertex_t>::Z;
    using Pfdr<real_t, vertex_t>::Z_Id;
    using Pfdr<real_t, vertex_t>::W;
    using Pfdr<real_t, vertex_t>::Id_W;
    using Pfdr<real_t, vertex_t>::D;
    using Pcd_prox<real_t>::X;
    using Pcd_prox<real_t>::cond_min;
    using Pcd_prox<real_t>::eps;
    using Pcd_prox<real_t>::malloc_check;

private:
    /** graph  **/

    /* list of edges, array of length 2E;
     * edge number e connects vertices indexed at edges[2*e] and edges[2*e+1];
     * every vertex should belong to at least one edge with a nonzero 
     * penalization coefficient. If it is not the case, a workaround is to add 
     * an edge from the vertex to itself with a small nonzero weight */
    const vertex_t* const &edges = Pfdr<real_t, vertex_t>::aux_idx;

    /**  graph total variation  **/

    /* see public enum declaration; must be initialized before d1shape */
    const D1p d1p;

    /* if 'edge_weights' is not null, array of length E;
     * otherwise homogeneously equal to 'homo_edge_weight' */
    const real_t* edge_weights;
    real_t homo_edge_weight;

    /* for multidimensional data, weights the coordinates in the lp norms;
     * array of length D; all weights must be strictly positive, and it is
     * advised to normalize the weights so that the first value is unity
     * for computation stability */
    const real_t* d1p_metric;

    real_t *W_d1, *Th_d1; // weights and thresholds for d1 proximal operator
    real_t w_d1, th_d1;
    const Condshape wd1shape; 
    const Condshape thd1shape; 

    /* functions for initializing constant members */
    Condshape compute_ga_shape(const real_t* d1p_metric,
        Condshape hess_f_h_shape) const
    {
        if (hess_f_h_shape == MULTIDIM || d1p_metric){
            return MULTIDIM;
        }else{
            return MONODIM;
        }
    }

    Condshape compute_w_shape(D1p d1p, const real_t* d1p_metric,
        Condshape hess_f_h_shape) const
    {
        if (d1p == D12 && (hess_f_h_shape == MULTIDIM || d1p_metric)){
            return MULTIDIM;
        }else{
            return MONODIM;
        }
    }

    Condshape compute_wd1_shape(D1p d1p, const real_t* d1p_metric,
        Condshape hess_f_h_shape) const
    {
        if (hess_f_h_shape == SCALAR /* null hessian */){
            return SCALAR;
        }else if (d1p == D12 || (hess_f_h_shape == MONODIM && !d1p_metric)){
            return MONODIM;
        }else{
            return MULTIDIM;
        }
    }

    Condshape compute_thd1_shape(D1p d1p, const real_t* d1p_metric,
        Condshape hess_f_h_shape) const
    {
        if (d1p == D11 && (hess_f_h_shape == MULTIDIM || d1p_metric)){
            return MULTIDIM;
        }else{
            return MONODIM;
        }
    }
};

/***  Concerning pfdr::gashape, pfdr::wshape, wd1shape, and thd1shape
 * ----------------------------------------------------------------------------
 * optimizing the required sizes and values of the preconditioner Γ, the
 * weights W, the weights W_d1 and the thresholds Th_d1 depends on:
 * - the ambiant dimension D (difference between MONODIM and MULTIDIM)
 * - the type of graph total variation used (member d1p : D11 or D12)
 * - the type of functionals f and h : linear (H(f+h) = 0), or hessian constant
 *   along coordinates (H(f+h) MONODIM), or hessian varying along coordinates
 *   (H(f+h) MULTIDIM)
 * - the presence of weights on the coordinates in the norm used in the graph
 *   total variation (d1p_metric)
 *
 * (1) The preconditionner is determined by the (pseudo-) hessians, roughly:
 *      Γ⁻¹_{v,d} = H(f + h)_{v,d} + Hg_{v,d}
 * where
 *      H(f + h)_{v,d} = ∂²(f + h)/∂x_{v,d}²
 * and
 *      Hg_{v,d} = sum_{u : (u, v) in E} edge_w(u, v)/dif(u, v)*coor_w(d)
 * (where dif(u, v) depends on ||x_u - x_v||p)
 *
 * It can be noted that when ∂²(f + h)/∂x_{v,d}² does not depend on d 
 * (i.e. H(f+h) MONODIM) and that there is no weights on the coordinates,
 * then Γ does not depend on d either, that is to say pfdr::gashape is MONODIM
 *
 * (2) With the D11 graph total variation (otherwise see (5)), the
 * splitting weights are directly determined for each edge by its contributions
 * to Hg above, and then normalized so that sum Wi = Id
 *
 *      W_{(u,v), v, d} = (edge_w(u, v)/dif(u, v)*coor_w(d))
 *          / (sum_{u' : (u', v) in E} edge_w(u', v)/dif(u', v)*coor_w(d))
 *
 * It can be noted that the coor_w(d) cancel, and thus W is constant along 
 * coordinates, that is to say pfdr::wshape is MONODIM
 *
 * Weights W_d1 (Wd1 below) and thresholds Th_d1 (Thd1 below), involved in the
 * proximal operator of the graph total variation, are computed according to
 * the following formulae, where, (u, v) and d being given,
 *
 *      w_γ_s stands for Γ⁻¹_{s, d} W_{(u, v), s, d}
 *
 * (3) Wd1_{(u, v), v, d} = w_γ_v /(w_γ_u + w_γ_v)
 *
 * It can be noted that when H(f + h) = 0, with weights (2) and in view of (1)
 *      w_γ_s = edge_w(u, v)/dif(u, v)*coor_w(d)
 * does not depend on s, and thus Wd1_{(u, v), v, d} is constant equal to 1/2
 * (for all (u, v), v, and d)
 *
 * (4) With the D11 graph total variation (otherwise see (5)), the proximal
 * operator is separable along coordinates, and 
 *      Thd1{(u,v),d} = edge_w(u, v)*coor_w(d)*(w_γ_u + w_γ_v)/(w_γ_u*w_γ_v)
 *
 * It can be noted that when H(f + h) = 0 and with weights (2),
 *      (w_γ_u + w_γ_v)/(w_γ_u*w_γ_v) = 2*dif(u, v)/(edge_w(u,v)*coor_w(d)) ;
 * in particular, weights on coordinates cancel and thd1shape can be MONODIM
 *
 * (5) In the case of D12 graph total variation, the weights W are used to
 * control the shape of the metric of the proximal operator : the metric must
 * have the shape determined by the weights on the coordinates (Euclidean
 * metric when there is no weights). The constraint is that for all (u, v), v,
 * there exists λ_{(u, v), v} such that for all d,
 *      Γ⁻¹_{v, d} W_{(u, v), v, d} = λ_{(u, v), v} coor_w(d) ;
 * the resulting weights and thresholds for the proximal operator of the D12
 * graph TV are then given by (3) and (4), substituing w_γ_v by
 * λ_{(u, v), * v} coor_w(d) 
 * It can be observed that the influence of coor_w(d) cancels, and thus (3) and
 * (4) does not depend on d; this is expected since the proximal operator is
 * not anymore separable along coordinates, thresholding the whole vector of
 * finite differences at once
 *
 * Once Γ is set by (1), we define ~W by
 *      ~W_{(u, v), v, d} = Γ_{v, d} coor_w(d)
 * and then modify ~W so that sum W = Id
 *
 * (5.1) supposing first that H(f + h) = 0, according to (1) the coor_w(d)
 * in (Γ_{v, d} coor_w(d)) cancel and we get
 *      ~W_{(u, v), v, d} = 1/(sum_{u' : (u',v) in E} edge_w(u',v)/dif(u',v))
 * In that case, it suffices to set
 *      W_{(u, v), v, d} = edge_w(u,v)/dif(u,v) * ~W_{(u, v), v, d}
 * and we get the desired properties with λ_{(u, v), v} = edge_w(u,v)/dif(u,v)
 * this amounts to the _very same weights_ as in case (2), and just as noted in
 * (3), Wd1 is constant equal to 1/2, Thd1{(u,v)} = 2*dif(u, v) can be
 * computed as in (4) where coor_w(d) cancel
 *
 * (5.2) supposing now that H(f + g) does not depend on d (i.e. is MONODIM) and
 * that there is no weights on the coordinates, according to (1) Γ is also 
 * MONODIM, and one can again use the weights given by (2) and compute Wd1 and
 * Thd1 according to (3) and (4), respectively.
 *
 * (5.3) in the general case, sum W = Id cannot be achieved while keeping the
 * metric shape imposed by the weights on the coordinates, and it is necessary
 * to resort to additional auxiliary variable Z_Id and weights Id_W so that
 * sum W + Id_W = Id. Similarly to the technique proposed by
 * Raguet and Landrieu (2015), set
 *      s_{v,d} = sum_{u':(u',v) in E} ~W_{(u',v),v,d},
 * and then
 *      W_{(u,v),v,d} = ~W_{(u,v),v,d} / (max_d' s_{v,d'})
 * Since here ~W_{(u',v),v,d} only depends on (v, d), we get  
 *      max_d' s_{v,d'} = card {u':(u',v) in E} * max_d' {Γ_{v, d'} coor_w(d')}
 * and finally
 *      W_{(u,v),v,d} = Γ_{v, d} coor_w(d) / (max_d' {Γ_{v, d'} coor_w(d')})
 *                                         / card {u':(u',v) in E}
 * (which in turn happens to depend only on (v, d)). We get the desired
 * properties with λ_{(u, v), v} = 1/ (max_d' {Γ_{v, d'} coor_w(d')})
 *                                  / card {u':(u',v) in E},
 * determining in turn Wd1 and Thd1 following respectively (3) and (4)
 * One can make clear the fact that
 *      sum_{u':(u',v) in E} W_{(u',v),v,d} = Γ_{v, d} coor_w(d)
 *                                            / (max_d' {Γ_{v, d'} coor_w(d')})
 * and deduce Id_W accordingly.                                            ***/
