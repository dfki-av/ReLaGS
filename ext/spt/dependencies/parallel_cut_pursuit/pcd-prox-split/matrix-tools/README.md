## Some tools for dealing with linear operators

`operator_norm_matrix`: compute the square operator norm of a real matrix, ║_A_<sup>*</sup>_A_║, using _power method_.  

`symmetric_equilibration_jacobi`: extract the inverse square root of the diagonal of _A_<sup>*</sup>_A_ 

`symmetric_equilibration_bunch`: diagonal ℓ<sub>∞</sub> norm equilibration with Bunch method

Work possibly in parallel with OpenMP API  

### Directory tree
    .   
    ├── include/    C++ headers, with some doc  
    └── src/        C++ sources  

### References

R. Von Mises and H. Pollaczek-Geiringer, Praktische Verfahren der Gleichungsauflösung, Zeitschrift für Angewandte Mathematik und Mechanik, 1929, 9, 152-164  

J. R. Bunch, Equilibration of Symmetric Matrices in the Max-Norm, Journal of the ACM, 1971, 18, 566-572  

### License
This software is under the GPLv3 license.  

Hugo Raguet 2016, 2018  
