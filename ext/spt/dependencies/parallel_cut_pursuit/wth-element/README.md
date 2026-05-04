### w-th Element: Weighted n-th Element

The _weighted rank interval_ of an element _e_ can be defined as the interval

     [wsum(_e_), wsum(_e_) + weight(_e_)[ ,

where wsum(_e_) is the cumulative sum of the weights of all elements comparing
lower to _e_, and weight(_e_) is the weight associated to _e_.  
The _w-th element_ is then the element whose weighted rank interval contains _w_;
note that if all weights are equal to unity, the _w_-th element with _w_ = _n_ reduces to the _n_-th element (starting count at 0).  

Based on quickselect algorithm.  

### Directory tree
    .   
    ├── include/    C++ header, actually #include'ing the sources, with some doc  
    ├── octave/     for GNU Octave or Matlab  
    │   ├── doc/    some documentation  
    │   └── mex/    MEX API  
    └── src/        C++ sources

### C++
The C++ routines are documented within the corresponding headers in `include/`.  

### GNU Octave or Matlab
The MEX interface is documented within dedicated `.m` files in `mex/doc/`.  

See `octave/compile_wth_element_mex.m` for typical compilation commands; it can be run directly from the GNU Octave interpreter.  

### References and license
This software is under the GPLv3 license.  

Hugo Raguet 2018  
