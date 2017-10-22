# Cut-Pursuit with Preconditioned Forward-Douglas–Rachford for minimizing graph total variation with additional nondifferentiable terms
Routines in C/C++, with mex API for interface with GNU Octave or MATLAB.  
Parallel implementation with OpenMP API.  

## General problem statement
This extension of the [cut-pursuit algorithm](https://github.com/loicland/cut-pursuit) allows to minimize functionals structured over a graph _G_ = (_V_, _E_)

    _F_(_x_) = _f_(_x_) + ∑<sub>_v_ ∈ _V_</sub> _g_<sub>_v_</sub>(_x_<sub>_v_</sub>) +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _λ_<sub>(_u_,_v_)</sub> ║<i>x</i><sub>_u_</sub> − _x_<sub>_v_</sub>║ ,    

where _x_ ∈ ℍ<sup>_V_</sup> for some base vector space ℍ, _f_ is differentiable, and for all _v_ ∈ _V_, _g_<sub>_v_</sub> admits _directional derivatives_ on every points of its domain and every directions.  

The cut-pursuit algorithm seeks partitions __*V*__ = (_U_<sub>1</sub>,...,_U_<sub>|__*V*__|</sub>) of the set of vertices _V_, constituting the constant connected components of the solution, by successively solving the corresponding problem, structured over the reduced graph __*G*__ = (__*V*__, __*E*__), that is

  arg min<sub>_ξ_ ∈ ℍ<sup>__*V*__</sup></sub>
    _F_(_x_) ,   such that ∀ _U_ ∈ __*V*__, ∀ _u_ ∈ _U_, _x_<sub>_u_</sub> = _ξ_<sub>_U_</sub> ,

and then refining the partition.  
A key requirement is thus the ability to solve the reduced problem, which often have the exact same structure as the original one, but with much less vertices |__*V*__| ≪ |_V_|. If the solution of the original problem has only few constant connected components in comparison to the number of vertices, the cut-pursuit strategy can speed-up minimization by several order of magnitude.  
The [preconditioned forward-Douglas–Rachford](https://1a7r0ch3.github.io/pgfb/) splitting algorithm is often well suited for such minimization, when the functionals are convex. We provide also mex API for using it directly to solve the original problem; just remove the prefix `CP_` in the methods listed below.  

For some nonconvex cases, where the norm on the difference in the graph total variation is replaced by a ℓ<sub>0</sub>-norm, see [L0 cut-pursuit](https://github.com/loicland/cut-pursuit) repository, by Loïc Landrieu.

## Available Routines

We provide implementations for a wide range of applications in convex cases, often used in signal processing or machine learning. We specify here the mex API, but the C/C++ routines are available separately (see below, § Mex Routines).  

### Quadratic functional with ℓ<sub>1</sub>-norm regularisation
The base space is ℍ = ℝ, and the general form is  

    _F_(_x_) = 1/2 ║<i>y</i> − _A_<i>x</i>║<sup>2</sup> +
 ∑<sub>_v_ ∈ _V_</sub> _λ_<sup>(ℓ)</sup><sub>_v_</sub> |_x_<sub>_v_</sub>| +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _λ_<sup>(δ)</sup><sub>(_u_,_v_)</sub>
 |_x_<sub>_u_</sub> − _x_<sub>_v_</sub>| ,  

where _y_ ∈ ℝ<sup>_n_</sup>, and _A_: ℝ<sup>_n_</sup> → ℝ<sup>_V_</sup> is a linear operator, and 
_λ_<sup>(ℓ)</sup> ∈ ℝ<sup>_V_</sup> and _λ_<sup>(δ)</sup> ∈ ℝ<sup>_E_</sup> are regularization weights.  

Currently, _A_ must be provided as a matrix.  
There are several particular cases:  
 - _n_ ≪ |_V_|, call on `CP_PFDR_graph_quadratic_d1_l1_mex`  
 - _n_ ≥ |_V_|, call on `CP_PFDR_graph_quadratic_d1_l1_AtA_mex`
with precomputations by _A_<sup>\*</sup>   
 - _A_ is diagonal or _f_ is a square weighted ℓ<sub>2</sub>-norm, call on
`CP_PFDR_graph_l22_d1_l1_mex`  

### Quadratic functional with box constraint
The base space is ℍ = ℝ, and the general form is  

    _F_(_x_) = 1/2 ║<i>y</i> − _A_ <i>x</i>║<sup>2</sup> +
 ∑<sub>_v_ ∈ _V_</sub> _ι_<sub>\[_m_,_M_\]</sub>(_x_<sub>_v_</sub>) +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _λ_<sub>(_u_,_v_)</sub>
 |_x_<sub>_u_</sub> − _x_<sub>_v_</sub>| ,  

where _y_ ∈ ℝ<sup>_n_</sup>, and _A_: ℝ<sup>_n_</sup> → ℝ<sup>_V_</sup> is a linear operator, _λ_ ∈ ℝ<sup>_E_</sup> are regularization weights, and _ι_<sub>\[_m_,_M_\]</sub> is the convex indicator of \[_m_,_M_\] : x ↦ 0 if _m_ ≤ _x_ ≤ _M_, and +∞ otherwise.  

Currently, _A_ must be provided as a matrix.  
There are several particular cases:  
 - _n_ ≪ |_V_|, call on `CP_PFDR_graph_quadratic_d1_bounds_mex`  
 - _n_ ≥ |_V_|, call on `CP_PFDR_graph_quadratic_d1_l1_bounds_mex`
with precomputations by _A_<sup>\*</sup>   
 - _A_ is diagonal or _f_ is a square weighted ℓ<sub>2</sub>-norm, call on
`CP_PFDR_graph_l22_d1_bounds_mex`  

### Separable loss with simplex constraint
The base space is ℍ = ℝ<sup>_K_</sup>, where _K_ is a set of labels, and the general form is  

    _F_(_x_) = f(_y_, _x_) +
 ∑<sub>_v_ ∈ _V_</sub> _ι_<sub>_Δ_<sub>_K_</sub></sub>(_x_<sub>_v_</sub>) +
 ∑<sub>(_u_,_v_) ∈ _E_</sub> _λ_<sub>(_u_,_v_)</sub> +
 ∑<sub>_k_ ∈ _K_</sub> |_x_<sub>_u_,_k_</sub> − _x_<sub>_v_,_k_</sub>| ,  

where _y_ ∈ ℝ<sup>_K_ ⨯ _V_</sup>, _f_ is a loss functional (see below), _λ_ ∈ ℝ<sup>_E_</sup> are regularization weights, and _ι_<sub>_Δ_<sub>_K_</sub></sub> is the convex indicator of the simplex
_Δ_<sub>_K_</sub> = {_x_ ∈ ℝ<sup>_K_</sup> | ∑<sub>k</sub> x<sub>k</sub> = 1 and ∀ k, x<sub>k</sub> ≥ 0}.  

The following loss functionals are available, all implemented in the routine
`CP_PFDR_graph_loss_d1_simplex_mex`:
 - linear, f(_y_, _x_) = − ∑<sub>_v_ ∈ _V_</sub> ∑<sub>_k_ ∈ _K_</sub> _x_<sub>_v_,_k_</sub> _y_<sub>_v_,_k_</sub>
 - quadratic, f(_y_, _x_) = ∑<sub>_v_ ∈ _V_</sub> ∑<sub>_k_ ∈ _K_</sub> (_x_<sub>_v_,_k_</sub> − _y_<sub>_v_,_k_</sub>)<sup>2</sup>
 - smoothed Kullback–Leibler for some _α_, ∑<sub>_v_ ∈ _V_</sub>
KL(_α_ _u_ + (1 − _α_) _y_<sub>_v_</sub>, _α_ _u_ + (1 − _α_) _x_<sub>_v_</sub>),
where _u_ ∈ ℝ<sup>_K_</sup> is the uniform discrete distribution over _K_, and
KL(_p_, _q_) = ∑<sub>_k_ ∈ _K_</sub> _p_<sub>_k_</sub> log(_p_<sub>_k_</sub>/_q_<sub>_k_</sub>).  

## Mex Routines
Within directory `mex/`  
    C/C++ sources are in directory `src/`  
    headers are in directory `include/`  
    mex API are in directory `api/`  
    some documentation in directory `doc/`  

See `compile_mex.m` for typical UNIX compilation commands.

## Documentation
The mex interfaces are documented in `mex/doc/` within dedicated `.m` files.  
The C/C++ routines are documented in `mex/include/` within the corresponding headers.  

## References
H. Raguet and L. Landrieu, Cut-pursuit algorithm for nonsmooth functionals regularized by graph total variation, in preparation.

H. Raguet, A note on the forward-Douglas-Rachford splitting algorithm, and application to convex optimization, [to appear](https://1a7r0ch3.github.io/pgfb/).
