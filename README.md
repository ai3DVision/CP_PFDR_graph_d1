# Cut-Pursuit with Preconditioned Forward-Douglas–Rachford for Minimizing Graph Total Variation with Additional Nondifferentiable Terms

Routines in C/C++.  
Parallel implementation with OpenMP API.  
MEX API for interface with GNU Octave or MATLAB.  
Boost API for interface with Python.

## General problem statement
This extension of the [cut-pursuit algorithm](https://github.com/loicland/cut-pursuit) minimizes functionals structured over a weighted graph _G_ = (_V_, _E_, _w_)

    _F_: _x_ ↦ _f_(_x_) + ∑<sub>_v_ ∈ _V_</sub> _g_<sub>_v_</sub>(_x_<sub>_v_</sub>) +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub> ║<i>x</i><sub>_u_</sub> − _x_<sub>_v_</sub>║ ,    

where _x_ ∈ ℍ<sup>_V_</sup> for some base vector space ℍ, _f_ is differentiable, and for all _v_ ∈ _V_, _g_<sub>_v_</sub> admits _directional derivatives_ on every points of its domain and every directions.  

The cut-pursuit algorithm seeks partitions __*V*__ of the set of vertices _V_, constituting the constant connected components of the solution, by successively solving the corresponding problem, structured over the reduced graph __*G*__ = (__*V*__, __*E*__), that is

  arg min<sub>_ξ_ ∈ ℍ<sup>__*V*__</sup></sub>
    _F_(_x_) ,    such that ∀ _U_ ∈ __*V*__, ∀ _u_ ∈ _U_, _x_<sub>_u_</sub> = _ξ_<sub>_U_</sub> ,

and then refining the partition.  
A key requirement is thus the ability to solve the reduced problem, which often have the exact same structure as the original one, but with much less vertices |__*V*__| ≪ |_V_|. If the solution of the original problem has only few constant connected components in comparison to the number of vertices, the cut-pursuit strategy can speed-up minimization by several order of magnitude.  
The [preconditioned forward-Douglas–Rachford](https://1a7r0ch3.github.io/pgfb/) splitting algorithm is often well suited for such minimization, when the functionals are convex. We provide also MEX API for using it directly to solve the original problem; just remove the prefix `CP_` in the methods listed below.  

For the nonconvex case where the norm on the difference in the graph total variation is replaced by a ℓ<sub>0</sub> norm, see [L0 cut-pursuit](https://github.com/loicland/cut-pursuit) repository, by Loïc Landrieu.

## Available Routines

We provide implementations for a wide range of applications in convex cases, often used in signal processing or machine learning. We specify here the routines available with the MEX API, they are not all availabel in python yet. C/C++ routines are available separately (see below, [§ Files and Folders](#files-and-folders) ).  

### Quadratic functional with ℓ<sub>1</sub>-norm regularization
The base space is ℍ = ℝ, and the general form is  

    _F_: _x_ ↦  1/2 ║<i>y</i> − _A_<i>x</i>║<sup>2</sup> +
 ∑<sub>_v_ ∈ _V_</sub> _λ_<sub>_v_</sub> |_x_<sub>_v_</sub>| +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub>
 |_x_<sub>_u_</sub> − _x_<sub>_v_</sub>| ,  

where _y_ ∈ ℝ<sup>_n_</sup>, and _A_: ℝ<sup>_n_</sup> → ℝ<sup>_V_</sup> is a linear operator, and 
_λ_ ∈ ℝ<sup>_V_</sup> and _w_ ∈ ℝ<sup>_E_</sup> are regularization weights.  
This combination of ℓ<sub>1</sub> norm and total variation is often coined _fused LASSO_.  
Implementation also supports an additional positivity constraint
  ∑<sub>_v_ ∈ _V_</sub>  _ι_<sub>ℝ<sub>+</sub></sub>(_x_<sub>_v_</sub>),
where _ι_<sub>ℝ<sub>+</sub></sub> is the convex indicator of ℝ<sub>+</sub> : x ↦ 0 if _x_ ≥ 0, +∞ otherwise.  

Currently, _A_ must be provided as a matrix.  
There are several particular cases:  
 - _n_ ≪ |_V_|, call on `CP_PFDR_graph_quadratic_d1_l1_mex`  
 - _n_ ≥ |_V_|, call on `CP_PFDR_graph_quadratic_d1_l1_AtA_mex`
with precomputations by _A_<sup>\*</sup>   
 - _A_ is diagonal or _f_ is a square weighted ℓ<sub>2</sub> norm, call on
`CP_PFDR_graph_l22_d1_l1_mex`  

### Quadratic functional with box constraints
The base space is ℍ = ℝ, and the general form is  

    _F_: _x_ ↦ 1/2 ║<i>y</i> − _A_ <i>x</i>║<sup>2</sup> +
 ∑<sub>_v_ ∈ _V_</sub> _ι_<sub>\[_m_,_M_\]</sub>(_x_<sub>_v_</sub>) +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub>
 |_x_<sub>_u_</sub> − _x_<sub>_v_</sub>| ,  

where _y_ ∈ ℝ<sup>_n_</sup>, and _A_: ℝ<sup>_n_</sup> → ℝ<sup>_V_</sup> is a linear operator, _w_ ∈ ℝ<sup>_E_</sup> are regularization weights, and _ι_<sub>\[_m_,_M_\]</sub> is the convex indicator of \[_m_,_M_\] : x ↦ 0 if _m_ ≤ _x_ ≤ _M_, +∞ otherwise.  

Currently, _A_ must be provided as a matrix.  
There are several particular cases:  
 - _n_ ≪ |_V_|, call on `CP_PFDR_graph_quadratic_d1_bounds_mex`  
 - _n_ ≥ |_V_|, call on `CP_PFDR_graph_quadratic_d1_bounds_AtA_mex`
with precomputations by _A_<sup>\*</sup>   
 - _A_ is diagonal or _f_ is a square weighted ℓ<sub>2</sub> norm, call on
`CP_PFDR_graph_l22_d1_bounds_mex`  

### Separable loss with simplex constraints
The base space is ℍ = ℝ<sup>_K_</sup>, where _K_ is a set of labels, and the general form is  

    _F_: _x_ ↦  _f_(_y_, _x_) +
 ∑<sub>_v_ ∈ _V_</sub> _ι_<sub>Δ<sub>_K_</sub></sub>(_x_<sub>_v_</sub>) +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub>
 ∑<sub>_k_ ∈ _K_</sub> |_x_<sub>_u_,_k_</sub> − _x_<sub>_v_,_k_</sub>| ,  

where _y_ ∈ ℝ<sup>_K_ ⨯ _V_</sup>, _f_ is a loss functional (see below), _w_ ∈ ℝ<sup>_E_</sup> are regularization weights, and _ι_<sub>Δ<sub>_K_</sub></sub> is the convex indicator of the simplex
Δ<sub>_K_</sub> = {_x_ ∈ ℝ<sup>_K_</sup> | ∑<sub>k</sub> x<sub>k</sub> = 1 and ∀ k, x<sub>k</sub> ≥ 0}: x ↦ 0 if _x_ ∈ Δ<sub>_K_</sub>, +∞ otherwise. 

The following loss functionals are available, all implemented in the routine
`CP_PFDR_graph_loss_d1_simplex_mex`:
 - linear, _f_(_y_, _x_) = − ∑<sub>_v_ ∈ _V_</sub> ∑<sub>_k_ ∈ _K_</sub> _x_<sub>_v_,_k_</sub> _y_<sub>_v_,_k_</sub>
 - quadratic, _f_(_y_, _x_) = ∑<sub>_v_ ∈ _V_</sub> ∑<sub>_k_ ∈ _K_</sub> (_x_<sub>_v_,_k_</sub> − _y_<sub>_v_,_k_</sub>)<sup>2</sup>
 - smoothed Kullback–Leibler, _f_(_y_, _x_) = ∑<sub>_v_ ∈ _V_</sub>
KL(_α_ _u_ + (1 − _α_) _y_<sub>_v_</sub>, _α_ _u_ + (1 − _α_) _x_<sub>_v_</sub>),  
where _α_ ∈ \]0,1\[,
_u_ ∈ Δ<sub>_K_</sub> is the uniform discrete distribution over _K_,
and
KL: (_p_, _q_) ↦ ∑<sub>_k_ ∈ _K_</sub> _p_<sub>_k_</sub> log(_p_<sub>_k_</sub>/_q_<sub>_k_</sub>).  


## Documentation
See [§ Available Routines](#available-routines) for the problems currently implemented.  

### Directory tree
    .   
    ├── data/       some illustrative data  
    ├── include/    C/C++ headers, with some doc  
    ├── octave/     GNU Octave/MATLAB code  
    │   ├── doc/    some documentation  
    │   └── mex/    MEX API  
    ├── python/     python code and API  
    └── src/        C/C++ sources  

### C/C++
The C/C++ routines are documented within the corresponding headers in `include/`.  

### GNU Octave/MATLAB
The MEX interfaces are documented within dedicated `.m` files in `mex/doc/`.  
Beware that currently, inputs are not checked, wrong input types and sizes lead to segmentation faults or aberrant results.  
See `mex/compile_mex.m` for typical UNIX compilation commands.  

The script `example_EEG_CP.m` exemplifies the problem [§ Quadratic functional with ℓ<sub>1</sub>-norm regularization](#quadratic-functional-with-ℓ1-norm-regularization), on a task of _brain source identification with electroencephalography_, described in the [references](#references).

### Python
Currently, only the routine for the problem described in [§ Quadratic functional with ℓ<sub>1</sub>-norm regularization](#quadratic-functional-with-ℓ1-norm-regularization) is implemented.  
Some documentation can be found in the header of `CP_quadratic_l1_py.cpp`.  

Make sure that your version of boost is at least 1.63, and that it is compiled with python support.  
For compilation with cmake, the provided `CMakeLists.txt` file assumes python 3; this can be easily modified within the file. Compile with  

    cd python  
    cmake .  
    make   

The script `example_EEG_CP.py` exemplifies the problem [§ Quadratic functional with ℓ<sub>1</sub>-norm regularization](#quadratic-functional-with-ℓ1-norm-regularization), on a task of _brain source identification with electroencephalography_, described in the [references](#references). The script requires the libraries `matplotlib` (tested with 2.2) and `PyQt4`.

You can compile the library in a given environment by replacing the ```cmake``` command by the following:
```
cmake . -DPYTHON_LIBRARY=$ENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$ENV/include/python3.6m -DBOOST_INCLUDEDIR=$ENV/include
```
with ```$ENV``` the path to the target environment (for anaconda it can be found with ```locate anaconda```). Adapt the version of python to the one used in your environment (for anaconda you can find it with ```locate anaconda3/lib/libpython```).

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
<td width="20%"><img src="data/ground_truth.png" width="100%"/></td>
<td width="10%"></td>
<td width="20%"><img src="data/brain_activity.png" width="100%"/></td>
<td width="10%"></td>
<td width="20%"><img src="data/brain_sources.png" width="100%"/></td>
<td width="10%"></td>
</tr></table>

Data courtesy of Ahmad Karfoul and Isabelle Merlet, LTSI, INSERM U1099.  

## References
[1] H. Raguet and L. Landrieu, [Cut-pursuit Algorithm for Regularizing Nonsmooth Functionals with Graph Total Variation](https://1a7r0ch3.github.io/cp/).

[2] H. Raguet, [A Note on the Forward-Douglas-Rachford Splitting Algorithm for Monotone Inclusion and Convex Optimization](https://1a7r0ch3.github.io/fdr/).
