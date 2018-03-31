function [Cv, rP, CP_it, Time, Obj, Dif] = CP_PFDR_graph_loss_d1_simplex_mex(Q, al, Eu, Ev, La_d1, CP_difTol, CP_itMax, PFDR_rho, PFDR_condMin, PFDR_difRcd, PFDR_difTol, PFDR_itMax, verbose)
%
%        [Cv, rP, CP_it, Time, Obj, Dif] = CP_PFDR_graph_loss_d1_simplex_mex(Q, al, Eu, Ev, La_d1, CP_difTol, CP_itMax, PFDR_rho, PFDR_condMin, PFDR_difRcd, PFDR_difTol, PFDR_itMax, verbose)
% Minimize functionals on graphs of the form:
%
%        F(p) = f(p) + ||p||_{d1,La_d1} + i_{simplex}(p)
%
% where for each vertex, p_v is a vector of length K,
%       f is a data-fidelity loss (depending on q and parameter al, see below)
%       ||p||_{d1,La_d1} = sum_{k, uv in E} la_d1_uvk |p_uk - p_vk|,
%   and i_{simplex} is the standard simplex constraint over each vertex,
%       for all v, (for all k, p_vk >= 0) and sum_k p_vk = 1,
%
% using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
% splitting algorithm.
%
%
% INPUTS: (warning: real numeric type is either single or double, not both)
% Q     - observed probabilities, K-by-V array (real)
%         strictly speaking, those are not required to lie on the
%         simplex, especially with linear fidelity term (al = 0)
% al    - scalar defining the data-fidelity loss function
%         al = 0, linear:
%                   f(p) = - <q, p>,
%         with  <q, p> = sum_{k,v} q_kv p_kv;
%         0 < al < 1, smoothed Kullback-Leibler divergence:
%                   f(p) = sum_v KLa(q_v||p_v),
%         with KLa(q_v||p_v) = KL(au + (1-a)q_v || au + (1-a)p_v),
%         where KL is the regular Kullback-Leibler divergence,
%               u is the uniform discrete distribution over {1,...,K},
%               and a = al is the smoothing parameter.
%         Up to a constant - H(au + (1-a)q_v))
%             = sum_{k} (a/K + (1-a)q_v) log(a/K + (1-a)q_v),
%         we have KLa(q_v||p_v)
%             = - sum_{k} (a/K + (1-a)q_v) log(a/K + (1-a)p_v);
%        al >= 1, quadratic:
%                 f(p) = al/2 ||q - p||_{l2}^2,
%        with  ||q - p||_{l2}^2 = sum_{k,v} (q_kv - p_kv)^2.
% Eu    - for each edge, C-style index of one vertex, array of length E (int32)
% Ev    - for each edge, C-style index of the other vertex, array of length E (int32)
%         The graph should be connected, with edges with nonzero penalization
%         coefficients. If it is not the case, the optimal values on isolated
%         components are independent, so they should be computed separately.
% La_d1 - d1 penalization coefficients for each edge, array of length E (real)
%
% [CP]
% difTol - stopping criterion on iterate evolution.
%          If difTol < 1, algorithm stops if relative changes of X (in l1 norm)
%          is less than difTol. If difTol >= 1, algorithm stops if less than
%          difTol maximum-likelihood labels have changed.
%          1e-2 is a conservative value; 1e-3 or less can give
%          better precision but with longer computational time
% itMax  - maximum number of iterations (graph cut and subproblem)
%          10 cuts solve accurately most problems
% it     - adress of an integer keeping track of iteration (cut) number
%
% [PFDR]
% rho     - relaxation parameter, 0 < rho < 2
%           1 is a conservative value; 1.5 often speeds up convergence
% condMin - parameter ensuring stability of preconditioning 0 < condMin <= 1
%           1 is a conservative value; 0.1 or 0.01 might enhance preconditioning
% difRcd  - reconditioning criterion on iterate evolution.
%           if difTol < 1, reconditioning occurs if all coordinates of P 
%           change by less than difRcd. difRcd is then divided by 10.
%           If difTol >= 1, reconditioning occurs if less than
%           difRcd maximum-likelihood labels have changed. difRcd is then 
%           divided by 10.
%           0 (no reconditioning) is a conservative value, 10*difTol or 
%           100*difTol might speed up convergence. reconditioning might 
%           temporarily draw minimizer away from solution, and give bad
%           subproblem solution
% difTol  - stopping criterion on iterate evolution.
%           If difTol < 1, algorithm stops if relative changes of X (in l1 norm)
%           is less than difTol. If difTol >= 1, algorithm stops if less than
%           difTol maximum-likelihood labels have changed.
%           1e-3*CP_difTol is a conservative value.
% itMax   - maximum number of iterations
%           1e4 iterations provides enough precision for most subproblems
%
% verbose - if nonzero, display information on the progress, every 'verbose'
%           iterations during subproblem resolution
%
% OUTPUTS:
% Cv    - assignement of each vertex of the minimizer to an homogeneous connected
%         component of the graph, numbered from 0 to (rV - 1)
%         array of length V (int32)
% rP    - values of each homogeneous connected components of the minimizer, 
%         array of size K-by-rV (real)
%         The actual minimizer is then reconstructed as P = rP(:,Cv+1);
% CP_it - actual number of iterations performed
% Time  - if requested, the elapsed time along iterations (itMax + 1 values)
% Obj   - if requested, the values of the objective functional along 
%         iterations (itMax + 1 values)
% Dif   - if requested, the iterate evolution along iterations (see difTol)
%
% Parallel implementation with OpenMP API.
%
% Typical compilation command (UNIX):
% mex CXXFLAGS="\$CXXFLAGS -DMEX -fopenmp -DNDEBUG" ...
%     LDFLAGS="\$LDFLAGS -fopenmp" ...
%     api/CP_PFDR_graph_loss_d1_simplex_mex.cpp ...
%     src/CP_PFDR_graph_loss_d1_simplex.cpp ...
%     src/PFDR_graph_loss_d1_simplex.cpp ...
%     src/graph.cpp src/maxflow.cpp src/proj_simplex_metric.cpp ...
%     -output bin/CP_PFDR_graph_loss_d1_simplex_mex
%
% Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
% Nonsmooth Functionals with Graph Total Variation.
%
% Hugo Raguet 2017
