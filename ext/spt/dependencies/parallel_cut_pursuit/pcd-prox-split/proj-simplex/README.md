### Orthogonal projection over the simplex:

    {x ∈ ℝ<sup>d</sup> | ∀_d_, _x_<sub>_d_</sub> ≥ 0  and  ∑<sub>_d_</sub> _x_<sub>_d_</sub> = _a_} ,
     
possibly within a diagonal metric defined by _m_<sup>-1</sup> as

   ⟨_x_, _y_⟩<sub>_m_<sup>-1</sup></sub> = ⟨_x_, diag(_m_<sup>-1</sup>) _y_⟩ = ∑<sub>_d_</sub> _x_<sub>_d_</sub> _y_<sub>_d_</sub> /_m_<sub>_d_</sub>

i.e. _m_ is the vector of the *inverses* of the diagonal entries of the 
matrix of the desired metric.

Based on Condat's modification (simple version) of Michelot (1986)'s algorithm  
Work possibly on different vectors in parallel with OpenMP API  

### Directory tree
    .   
    ├── include/    C++ headers, with some doc  
    ├── octave/     for GNU Octave or Matlab  
    │   ├── doc/    some documentation  
    │   └── mex/    MEX API  
    └── src/        C++ sources  

### C++
Be sure to have OpenMP enabled with your compiler to enjoy parallelization. Note that, as of 2020, MSVC still does not support OpenMP 3.0 (published in 2008); consider switching to a decent compiler.  

The number of parallel threads used in parallel regions is crucial for good performance; it is roughly controlled by a macro `MIN_OPS_PER_THREAD` which can be set by usual `D` compilation flag. A rule of thumb is to set it to `10000` on personnal computers with a handful of cores, and up to `100000` for large computer clusters with tens of cores.  

The C++ routines are documented within the corresponding headers in `include/`.  


### GNU Octave or Matlab
The MEX interfaces are documented within dedicated `.m` files in `mex/doc/`.  

See `octave/compile_mex.m` for typical compilation commands; it can be run directly from the GNU Octave interpreter.  

### Reference and license
L. Condat, Fast Projection onto the Simplex and the ℓ<sub>1</sub> Ball,
Mathematical Programming, 2016, 158, 575-585  

This software is under the GPLv3 license.  

Hugo Raguet 2016, 2018  
