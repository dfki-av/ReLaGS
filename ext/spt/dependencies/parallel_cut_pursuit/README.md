# Cut-Pursuit Algorithms, Parallelized Along Components

Generic C++ classes for implementing cut-pursuit algorithms.  
Specialization to convex problems involving **graph total variation**, and nonconvex problems involving **contour length**, as explained in our articles [(Landrieu and Obozinski, 2016; Raguet and Landrieu, 2018)](#references).   
Parallel implementation with OpenMP.  
MEX interfaces for GNU Octave or Matlab.  
Extension modules for Python.  

This Git repository uses submodules.  
Clone with  

    git clone --recurse-submodules https://gitlab.com/1a7r0ch3/parallel-cut-pursuit  

Pull changes with  

    git pull --recurse-submodules  


## Table of Contents  

1. [**General problem statement**](#general-problem-statement)  
2. [**C++ classes and Specializations**](#c-classes-pecializations)  
2.1. [Proximity operator of the graph total variation](#cp_prox_tv-proximity-operator-of-the-graph-total-variation)  
2.2. [Quadratic functional and graph total variation](#cp_d1_ql1b-quadratic-functional-ℓ1-norm-bounds-and-graph-total-variation)  
2.3. [Separable multidimensional loss and graph total variation](#cp_d1_lsx-separable-loss-simplex-constraints-and-graph-total-variation)  
2.4. [Separable distance and contour length](#cp_d0_dist-separable-distance-and-weighted-contour-length)  
3. [**Documentation**](#documentation)  
3.1. [Directory tree](#directory-tree)  
3.2. [Graph structure](#graph-structure)  
3.3. [C++ documentation](#c-documentation)  
3.4. [GNU Octave or Matlab](#gnu-octave-or-matlab)  
3.5. [Python](#python)  
4. [**References**](#references)  
5. [**License**](#license)  

## General problem statement
The cut-pursuit algorithms minimize functionals structured, over a weighted graph <i>G</i> = (<i>V</i>, <i>E</i>, <i>w</i>), as 

    <i>F</i>: <i>x</i> ∈ Ω<sup><i>V</i></sup> ↦ <i>f</i>(<i>x</i>) + 
 ∑<sub>(<i>u</i>,<i>v</i>) ∈ <i>E</i></sub> <i>w</i><sub>(<i>u</i>,<i>v</i>)</sub>  _ψ_(<i>x</i><sub><i>u</i></sub>, <i>x</i><sub><i>v</i></sub>) ,    

where Ω is some base set, and the functional _ψ_: Ω² → ℝ penalizes dissimilarity between its arguments, in order to enforce solutions which are *piecewise constant along the graph <i>G</i>*.

The cut-pursuit approach is to seek partitions __*V*__ of the set of vertices <i>V</i>, constituting the constant connected components of the solution, by successively solving the corresponding problem, structured over the reduced graph __*G*__ = (__*V*__, __*E*__), that is

  arg min<sub><i>ξ</i> ∈ Ω<sup><b><i>V</i></b></sup></sub>
  <i>F</i>(<i>x</i>) ,    such that ∀ <i>U</i> ∈ <b><i>V</i></b>, ∀ <i>u</i> ∈ <i>U</i>, <i>x</i><sub><i>u</i></sub> = <i>ξ</i><sub><i>U</i></sub> ,

and then refining the partition.  
A key requirement is thus the ability to solve the reduced problem, which often have the exact same structure as the original one, but with much less vertices |<b><i>V</i></b>| ≪ |<i>V</i>|. If the solution of the original problem has only few constant connected components in comparison to the number of vertices, the cut-pursuit strategy can speed-up minimization by several orders of magnitude.  

Cut-pursuit algorithms come in two main flavors, namely “directionally differentiable” and “noncontinuous”.

* In the **directionally differentiable** case, the base set Ω is typically a vector space, and it is required that <i>f</i> is differentiable, or at least that its nondifferentiable part is _separable along the graph_ and admits (potentially infinite) _directional derivatives_. This comprises notably many convex problems, where 
_ψ_(<i>x</i><sub><i>u</i></sub>, <i>x</i><sub><i>v</i></sub>) = ║<i>x</i><sub><i>u</i></sub> − <i>x</i><sub><i>v</i></sub>║, that is to say involving a _**graph total variation**_. The refinement of the partition is based on the search for a steep directional derivative, and the reduced problem is solved using convex or continuous optimization; optimality guarantees can be provided.  

* In the **noncontinuous** case, the dissimilarity penalization typically uses _ψ_(<i>x</i><sub><i>u</i></sub>, <i>x</i><sub><i>v</i></sub>) = 0 if <i>x</i><sub><i>u</i></sub> =<i>x</i><sub><i>v</i></sub>, 1 otherwise, resulting in a measure of the _**contour length**_ of the constant connected components. The functional <i>f</i> is typically required to be separable along the graph, and to have computational properties favorable enough for solving reduced problems. The refinement of the partition relies on greedy heuristics.

Both flavors admit multidimensional extensions, that is to say Ω is not required to be only a set of scalars.

## C++ classes and Specializations

The module `maxflow` implements the class `Maxflow`, a modification of the `Graph` class of Y. Boykov and V. Kolmogorov, for making use of their [maximum flow algorithm](#references).  

The module `cut_pursuit` implements the base class `Cp`, defining all steps of the cut-pursuit approach in virtual methods.  

The module `cut_pursuit_d1` implements the class `Cp_d1` derived from `Cp`, specializing cut-pursuit for directionally differentiable cases involving the graph total variation.  

The module `cut_pursuit_d0` implements the class `Cp_d0` derived from `Cp`, specializing cut-pursuit for noncontinuous cases involving the contour length penalization.  

### `Cp_prox_tv`: proximity operator of the graph total variation
Also coined “graph total variation denoising” or “general fused LASSO signal approximation”. The objective functional is 

    <i>F</i>: <i>x</i> ∈ ℝ<sup><i>D</i>⨯<i>V</i></sup> ↦ 
    1/2 ║<i>y</i> − <i>x</i>║<sub><i>M</i><sub>ℓ</sub></sub><sup>2</sup> +
    ∑<sub>(<i>u</i>,<i>v</i>) ∈ <i>E</i></sub> <i>w</i><sub>(<i>u</i>,<i>v</i>)</sub>
         ║<i>x</i><sub><i>u</i></sub> − <i>x</i><sub><i>v</i></sub>║<sub><i>p</i>, <i>M</i><sub>δ</sub></sub> ,   

where
<i>D</i> is the dimension of the signal on each vertex,
<i>y</i> ∈ ℝ<sup><i>D</i>⨯<i>V</i></sup>,
<i>M</i><sub>ℓ</sub> is a diagonal metric weighting the square ℓ<sub>2</sub> norm,
<i>w</i> ∈ ℝ<sup><i>E</i></sup> are regularization weights,
and the norm on the finite differences is defined by <i>p</i> being 1 or 2 and a weighting diagonal metric <i>M</i><sub>δ</sub>.  

The reduced problem is solved using the [preconditioned forward-Douglas–Rachford splitting algorithm](https://1a7r0ch3.github.io/fdr/), included as a git submodule `pcd-prox-split`.  

### `Cp_d1_ql1b`: quadratic functional, ℓ<sub>1</sub> norm, bounds, and graph total variation
The base set is Ω = ℝ, and the general form is  

    <i>F</i>: <i>x</i> ∈ ℝ<sup><i>V</i></sup> ↦ 
    1/2 ║<i>y</i><sup>(ℓ<sub>2</sub>)</sup> − <i>A</i><i>x</i>║<sup>2</sup> +
    ∑<sub><i>v</i> ∈ <i>V</i></sub> <i>λ</i><sub><i>v</i></sub>
        |<i>y</i><sup>(ℓ<sub>1</sub>)</sup> − <i>x</i><sub><i>v</i></sub>| +  
    ∑<sub><i>v</i> ∈ <i>V</i></sub>
        <i>ι</i><sub>[<i>m</i><sub><i>v</i></sub>, <i>M</i><sub><i>v</i></sub>]</sub>(<i>x</i><sub><i>v</i></sub>) +
    ∑<sub>(<i>u</i>,<i>v</i>) ∈ <i>E</i></sub> <i>w</i><sub>(<i>u</i>,<i>v</i>)</sub>
         |<i>x</i><sub><i>u</i></sub> − <i>x</i><sub><i>v</i></sub>| ,   

where
<i>y</i><sup>(ℓ<sub>2</sub>)</sup> ∈ ℝ<sup><i>n</i></sup>, 
<i>A</i>: ℝ<sup><i>V</i></sup> → ℝ<sup><i>n</i></sup> is a linear operator, 
<i>y</i><sup>(ℓ<sub>1</sub>)</sup> ∈ ℝ<sup><i>V</i></sup> and 
<i>λ</i> ∈ ℝ<sup><i>V</i></sup> and <i>w</i> ∈ ℝ<sup><i>E</i></sup> are regularization weights, 
<i>m</i>, <i>M</i> ∈ ℝ<sup><i>V</i></sup> are parameters and 
<i>ι</i><sub>[<i>a</i>,<i>b</i>]</sub> is the convex indicator of [<i>a</i>, <i>b</i>] : x ↦ 0 if <i>x</i> ∈ [<i>a</i>, <i>b</i>], +∞ otherwise.  

When <i>y</i><sup>(ℓ<sub>1</sub>)</sup> is zero, the combination of ℓ<sub>1</sub> norm and total variation is sometimes coined _fused LASSO_.  

When <i>A</i> is the identity, <i>λ</i> is zero and there are no box constraints, the problem boils down to [the proximity operator of the graph total variation](#cp_prox_tv-proximity-operator-of-the-graph-total-variation).  

Currently, <i>A</i> must be provided as a matrix. See the documentation for special cases.  

The reduced problem is solved using the [preconditioned forward-Douglas–Rachford splitting algorithm](https://1a7r0ch3.github.io/fdr/), included as a git submodule `pcd-prox-split`.  

An example with [GNU Octave or Matlab](#gnu-octave-or-matlab) and [Python](#python) interfaces, where <i>A</i> is a full ill-conditioned matrix, with positivity and fused LASSO constraints, on a task of _brain source identification from electroencephalography_.  


<table><tr>
<td width="10%"></td>
<td width="20%"> ground truth </td>
<td width="10%"></td>
<td width="20%"> raw retrieved activity </td>
<td width="10%"></td>
<td width="20%"> identified sources </td>
<td width="10%"></td>
</tr><tr>
<td width="10%"></td>
<td width="20%"><img src="https://gitlab.com/1a7r0ch3/pcd-prox-split/-/raw/master/data/EEG_ground_truth.png" width="100%"/></td>
<td width="10%"></td>
<td width="20%"><img src="https://gitlab.com/1a7r0ch3/pcd-prox-split/-/raw/master/data/EEG_brain_activity.png" width="100%"/></td>
<td width="10%"></td>
<td width="20%"><img src="https://gitlab.com/1a7r0ch3/pcd-prox-split/-/raw/master/data/EEG_brain_sources.png" width="100%"/></td>
<td width="10%"></td>
</tr></table>

### `Cp_d1_lsx`: separable loss, simplex constraints, and graph total variation
The base set is Ω = ℝ<sup><i>D</i></sup>, where <i>D</i> can be seen as a set of labels, and the general form is  

    <i>F</i>: <i>x</i> ∈ ℝ<sup><i>D</i>⨯<i>V</i></sup> ↦  <i>f</i>(<i>y</i>, <i>x</i>) +
 ∑<sub><i>v</i> ∈ <i>V</i></sub> <i>ι</i><sub>Δ<sub><i>D</i></sub></sub>(<i>x</i><sub><i>v</i></sub>) +  
 ∑<sub>(<i>u</i>,<i>v</i>) ∈ <i>E</i></sub> <i>w</i><sup>(d<sub>1</sub>)</sup><sub>(<i>u</i>,<i>v</i>)</sub>
 ∑<sub><i>d</i> ∈ <i>D</i></sub> <i>λ</i><sub><i>d</i></sub> |<i>x</i><sub><i>u</i>,<i>d</i></sub> − <i>x</i><sub><i>v</i>,<i>d</i></sub>| ,  

where <i>y</i> ∈ ℝ<sup><i>D</i>⨯<i>V</i></sup>, <i>f</i> is a loss functional (see below), <i>w</i><sup>(d<sub>1</sub>)</sup> ∈ ℝ<sup><i>E</i></sup> and <i>λ</i> ∈ ℝ<sup><i>D</i></sup> are regularization weights, and <i>ι</i><sub>Δ<sub><i>D</i></sub></sub> is the convex indicator of the simplex
Δ<sub><i>D</i></sub> = {<i>x</i> ∈ ℝ<sup><i>D</i></sup> | ∑<sub><i>d</i></sub> <i>x</i><sub><i>d</i></sub> = 1 and ∀ <i>d</i>, <i>x</i><sub><i>d</i></sub> ≥ 0}: <i>x</i> ↦ 0 if <i>x</i> ∈ Δ<sub><i>D</i></sub>, +∞ otherwise. 

The following loss functionals are available, where <i>w</i><sup>(<i>f</i>)</sup> ∈ ℝ<sup><i>V</i></sup> are weights on vertices.  
Linear: <i>f</i>(<i>y</i>, <i>x</i>) = − ∑<sub><i>v</i> ∈ <i>V</i></sub> <i>w</i><sup>(<i>f</i>)</sup><sub><i>v</i></sub> ∑<sub><i>d</i> ∈ <i>D</i></sub> <i>x</i><sub><i>v</i>,<i>d</i></sub> <i>y</i><sub><i>v</i>,<i>d</i></sub>  
Quadratic: <i>f</i>(<i>y</i>, <i>x</i>) = ∑<sub><i>v</i> ∈ <i>V</i></sub> <i>w</i><sup>(<i>f</i>)</sup><sub><i>v</i></sub> ∑<sub><i>d</i> ∈ <i>D</i></sub> (<i>x</i><sub><i>v</i>,<i>d</i></sub> − <i>y</i><sub><i>v</i>,<i>d</i></sub>)<sup>2</sup>  
Smoothed Kullback–Leibler divergence (equivalent to cross-entropy):  
<i>f</i>(<i>y</i>, <i>x</i>) = ∑<sub><i>v</i> ∈ <i>V</i></sub> <i>w</i><sup>(<i>f</i>)</sup><sub><i>v</i></sub>
KL(<i>α</i> <i>u</i> + (1 − <i>α</i>) <i>y</i><sub><i>v</i></sub>, <i>α</i> <i>u</i> + (1 − <i>α</i>) <i>x</i><sub><i>v</i></sub>),  
where <i>α</i> ∈ \]0,1\[,
<i>u</i> ∈ Δ<sub><i>D</i></sub> is the uniform discrete distribution over <i>D</i>,
and
KL: (<i>p</i>, <i>q</i>) ↦ ∑<sub><i>d</i> ∈ <i>D</i></sub> <i>p</i><sub><i>d</i></sub> log(<i>p</i><sub><i>d</i></sub>/<i>q</i><sub><i>d</i></sub>).  

The reduced problem is solved using the [preconditioned forward-Douglas–Rachford splitting algorithm](https://1a7r0ch3.github.io/fdr/), included as a git submodule `pcd-prox-split`.  

An example with the smoothed Kullback–Leibler is provided with [GNU Octave or Matlab](#gnu-octave-or-matlab) and [Python](#python) interfaces, on a task of _spatial regularization of semantic classification of a 3D point cloud_.  

<table><tr>
<td width="5%"></td>
<td width="25%"> ground truth </td>
<td width="5%"></td>
<td width="25%"> random forest classifier </td>
<td width="5%"></td>
<td width="25%"> regularized classification </td>
<td width="5%"></td>
</tr><tr>
<td width="5%"></td>
<td width="25%"><img src="https://gitlab.com/1a7r0ch3/pcd-prox-split/-/raw/master/data/labeling_3D_ground_truth.png" width="100%"/></td>
<td width="5%"></td>
<td width="25%"><img src="https://gitlab.com/1a7r0ch3/pcd-prox-split/-/raw/master/data/labeling_3D_random_forest.png" width="100%"/></td>
<td width="5%"></td>
<td width="25%"><img src="https://gitlab.com/1a7r0ch3/pcd-prox-split/-/raw/master/data/labeling_3D_regularized.png" width="100%"/></td>
<td width="5%"></td>
</tr></table>

### `Cp_d0_dist`: separable distance and weighted contour length
The base set is Ω = ℝ<sup><i>D</i></sup> or Δ<sub><i>D</i></sub> and the general form is  

    <i>F</i>: <i>x</i> ∈ ℝ<sup><i>D</i>⨯<i>V</i></sup> ↦  <i>f</i>(<i>y</i>, <i>x</i>) +
 ∑<sub>(<i>u</i>,<i>v</i>) ∈ <i>E</i></sub> <i>w</i><sup>(d<sub>0</sub>)</sup><sub>(<i>u</i>,<i>v</i>)</sub>
    ║<i>x</i><sub><i>u</i></sub> − <i>x</i><sub><i>v</i></sub>║<sub>0</sub> ,  

where <i>y</i> ∈ Ω<sup><i>V</i></sup>, <i>f</i> is a loss functional akin to a distance (see below), and 
║&middot;║<sub>0</sub> is the ℓ<sub>0</sub> pseudo-norm <i>x</i> ↦ 0 if <i>x</i> = 0, 1 otherwise.  

The following loss functionals are available, where <i>w</i><sup>(<i>f</i>)</sup> ∈ ℝ<sup><i>V</i></sup> are weights on vertices and <i>m</i><sup>(<i>f</i>)</sup> ∈ ℝ<sup><i>D</i></sup> are weights on coordinates.  
Weighted quadratic: Ω = ℝ<sup><i>D</i></sup> and 
<i>f</i>(<i>y</i>, <i>x</i>) = ∑<sub><i>v</i> ∈ <i>V</i></sub> <i>w</i><sup>(<i>f</i>)</sup><sub><i>v</i></sub> ∑<sub><i>d</i> ∈ <i>D</i></sub> <i>m</i><sup>(<i>f</i>)</sup><sub><i>d</i></sub> (<i>x</i><sub><i>v</i>,<i>d</i></sub> − <i>y</i><sub><i>v</i>,<i>d</i></sub>)<sup>2</sup>  
Weighted smoothed Kullback–Leibler divergence (equivalent to cross-entropy):
Ω = Δ<sub><i>D</i></sub> and  
<i>f</i>(<i>y</i>, <i>x</i>) = ∑<sub><i>v</i> ∈ <i>V</i></sub> <i>w</i><sup>(<i>f</i>)</sup><sub><i>v</i></sub>
KL<sub><i>m</i><sup>(<i>f</i>)</sup></sub>(<i>α</i> <i>u</i> + (1 − <i>α</i>) <i>y</i><sub><i>v</i></sub>, <i>α</i> <i>u</i> + (1 − <i>α</i>) <i>x</i><sub><i>v</i></sub>),  
where <i>α</i> ∈ \]0,1\[,
<i>u</i> ∈ Δ<sub><i>D</i></sub> is the uniform discrete distribution over <i>D</i>,
and  
KL<sub><i>m</i><sup>(<i>f</i>)</sup></sub>: (<i>p</i>, <i>q</i>) ↦ ∑<sub><i>d</i> ∈ <i>D</i></sub> <i>m</i><sup>(<i>f</i>)</sup><sub><i>d</i></sub> <i>p</i><sub><i>d</i></sub> log(<i>p</i><sub><i>d</i></sub>/<i>q</i><sub><i>d</i></sub>).   

The reduced problem amounts to averaging, and the split step uses <i>k</i>-means++ algorithm.  

When the loss is quadratic, the resulting problem is sometimes coined “minimal partition problem”.  

An example with the smoothed Kullback–Leibler is provided with [GNU Octave or Matlab](#gnu-octave-or-matlab) interface, on a task of _spatial regularization of semantic classification of a 3D point cloud_.  

## Documentation

### Directory tree
    .   
    ├── include/        C++ headers, with some doc  
    ├── octave/         GNU Octave or Matlab code  
    │   ├── doc/        some documentation  
    │   └── mex/        MEX C++ interfaces
    ├── pcd-prox-split/ git submodule preconditionned forward-Douglas–Rachford 
    │                   algorithm (required only for directionnaly 
    │                   differentiable cases and example data)
    ├── python/         Python code  
    │   ├── cpython/    C Python interfaces  
    │   └── wrappers/   python wrappers and documentation  
    ├── src/            C++ sources  
    └── wth-element/    git submodule for weighted quantiles search
                        (required only for cp_d1_ql1b) 


### C++ documentation
Requires `C++11`.  
Be sure to have OpenMP enabled with your compiler to enjoy parallelization. Note that, as of 2020, MSVC still does not support OpenMP 3.0 (published in 2008); consider switching to a decent compiler.  

The number of parallel threads used in parallel regions is crucial for good performance; it is roughly controlled by a preprocessor macro `MIN_OPS_PER_THREAD` which can be again set with`-D` compilation flag. A rule of thumb is to set it to `10000` on personal computers with a handful of cores, and up to `100000` for large computer clusters with tens of cores.  

The C++ classes are documented within the corresponding headers in `include/`.  

### Graph structure
Graph structures must be given as forward-star representation. For conversion from simple adjacency list representation, or for creation from scratch for regular N-dimensionnal grids (2D for images, 3D for volumes, etc.), see the `pcd-prox-split/grid-graph` git submodule.  

### GNU Octave or Matlab
See the script `compile_parallel_cut_pursuit_mex.m` for typical compilation commands; it can be run directly from the GNU Octave interpreter, but Matlab users must set compilation flags directly on the command line `CXXFLAGS = ...` and `LDFLAGS = ...`.  

The integer type holding the components assignment is by defaut on 16 bits. For applications expecting a large number of components, this can be extended to 32 bits with the compilation option `-DCOMP_T_ON_32_BITS`.

Extensive documentation of the MEX interfaces can be found within dedicated `.m` files in `octave/doc/`.  

The script `example_prox_tv.m` exemplifies the use of [`Cp_prox_tv`](#cp_prox_tv-proximity-operator-of-the-graph-total-variation), on a task of _color image denoising_.  

The script `example_EEG.m` exemplifies the use of [`Cp_d1_ql1b`](#cp_d1_ql1b-quadratic-functional-ℓ1-norm-bounds-and-graph-total-variation), on a task of _brain source identification from electroencephalography_.  

The scripts `example_labeling_3D.m` and `example_labeling_3D_d0.m` exemplify the use of, respectively, [`Cp_d1_lsx`](#cp_d1_lsx-separable-loss-simplex-constraints-and-graph-total-variation) and [`Cp_d0_dist`](#cp_d0_dist-separable-distance-and-weighted-contour-length), on a task of _spatial regularization of semantic classification of a 3D point cloud_.  

### Python
Requires `numpy` package.  
See the script `setup.py` for compiling modules with `distutils`; on UNIX systems, it can be directly interpreted as `python setup.py build_ext`.  

The integer type holding the components assignment is by defaut on 16 bits. For applications expecting a large number of components, this can be extended to 32 bits with the compilation option `-DCOMP_T_ON_32_BITS`.

Extensive documentation of the Python wrappers can be found in the corresponding `.py` files.  

The script `example_prox_tv.py` exemplifies the use of [`Cp_prox_tv`](#cp_prox_tv-proximity-operator-of-the-graph-total-variation), on a task of _color image denoising_.  

The script `example_EEG.py` exemplifies the use of [`Cp_d1_ql1b`](#cp_d1_ql1b-quadratic-functional-ℓ1-norm-bounds-and-graph-total-variation), on a task of _brain source identification from electroencephalography_.  

The scripts `example_labeling_3D.py` and `example_labeling_3D_d0.py` exemplify the use of, respectively, [`Cp_d1_lsx`](#cp_d1_lsx-separable-loss-simplex-constraints-and-graph-total-variation) and [`Cp_d0_dist`](#cp_d0_dist-separable-distance-and-weighted-contour-length), on a task of _spatial regularization of semantic classification of a 3D point cloud_.  

## References
L. Landrieu and G. Obozinski, [Cut Pursuit: Fast Algorithms to Learn Piecewise Constant Functions on Weighted Graphs](http://epubs.siam.org/doi/abs/10.1137/17M1113436), 2017.  

H. Raguet and L. Landrieu, [Cut-pursuit Algorithm for Regularizing Nonsmooth Functionals with Graph Total Variation](https://1a7r0ch3.github.io/cp/), 2018.  

Y. Boykov and V. Kolmogorov, An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision, IEEE Transactions on Pattern Analysis and Machine Intelligence, 2004.

## License
This software is under the GPLv3 license.
