/*=============================================================================
 *
 * Minimize functionals on graphs of the form:
 *
 *        F(p) = f(p) + ||p||_{d1,La_d1} + i_{simplex}(p)
 *
 * where for each vertex, p_v is a vector of length K,
 *       f is a data-fidelity loss (depending on q and parameter al, see below)
 *       ||p||_{d1,La_d1} = sum_{k, uv in E} la_d1_uvk |p_uk - p_vk|,
 *   and i_{simplex} is the standard simplex constraint over each vertex,
 *       for all v, (for all k, p_vk >= 0) and sum_k p_vk = 1,
 *
 * using cut-pursuit approach with preconditioned forward-Douglas-Rachford 
 * splitting algorithm.
 *
 * Parallel implementation with OpenMP API.
 *
 * Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
 * Nonsmooth Functionals with Graph Total Variation.
 *
 * Hugo Raguet 2017
 *===========================================================================*/
#ifndef CP_PFDR_GRAPH_LOSS_D1_SIMPLEX_H
#define CP_PFDR_GRAPH_LOSS_D1_SIMPLEX_H

template <typename real> struct CPls_Restart;
/* structure for "warm restart"
 * for more information, see implemention of CP_PFDR_graph_loss_d1_simplex
 * G   - graph structure with V nodes and E edges
 * Vc  - list of vertices within each connected component
 * rVc - cumulative sum of the components sizes (in an array of length (rV + 1)) */

template <typename real> struct CPls_Restart<real>*
create_CPls_Restart(const int K, const int V, const int E, const real al, \
    int *rV, int *Cv, real **rP, const real *Q, const int *Eu, const int *Ev);

template <typename real>
void free_CPls_Restart(struct CPls_Restart<real> *CP_restart);

template <typename real>
void CP_PFDR_graph_loss_d1_simplex(const int K, const int V, const int E, \
    const real al, int *rV, int *Cv, real **rP, const real *Q, \
    const int *Eu, const int *Ev, const real *La_d1, \
    const real CP_difTol, const int CP_itMax, int *CP_it, \
    const real PFDR_rho, const real PFDR_condMin, \
    const real PFDR_difRcd, const real PFDR_difTol, const int PFDR_itMax, \
    double *Time, real *Obj, real *Dif, const int verbose, \
    struct CPls_Restart<real> *CP_restart);
/* 25 arguments:
 * K, V, E    - number of classes, of vertices, of (undirected) edges
 * al         - scalar defining the data-fidelity loss function
 *              al = 0, linear:
 *                        f(p) = - <q, p>,
 *                with  <q, p> = sum_{k,v} q_kv p_kv;
 *              0 < al = a < 1, smoothed Kullback-Leibler divergence:
 *                        f(p) = sum_v KLa(q_v||p_v),
 *                with KLa(q_v||p_v) = KL(au + (1-a)q_v || au + (1-a)p_v),
 *                where KL is the regular Kullback-Leibler divergence,
 *                      u is the uniform discrete distribution over {1,...,K},
 *                      and a = al is the smoothing parameter.
 *                Up to a constant - H(au + (1-a)q_v))
 *                    = sum_{k} (a/K + (1-a)q_v) log(a/K + (1-a)q_v),
 *                we have KLa(q_v||p_v)
 *                    = - sum_{k} (a/K + (1-a)q_v) log(a/K + (1-a)p_v);
 *              al = 1, quadratic:
 *                        f(p) = 1/2 ||q - p||_{l2}^2,
 *              with  ||q - p||_{l2}^2 = sum_{k,v} (q_kv - p_kv)^2.
 * rV         - adress of an integer keeping track of the number of homogeneous
 *              connected components of the minimizer
 * Cv         - assignement of each vertex of the minimizer to an homogeneous connected
 *              component of the graph, array of length V
 * rP         - adress of a pointer keeping track of the values of each
 *              homogeneous connected components of the minimizer (in a K-by-rV array,
 *              column major format)
 * Q          - observed probabilities, K-by-V array, column major format
 *              strictly speaking, those are not required to lie on the simplex
 * Eu         - for each edge, index of one vertex, array of length E
 * Ev         - for each edge, index of the other vertex, array of length E
 *              Every vertex should belong to at least one edge. If it is not the
 *              case, the optimal value of an isolated vertex is independent
 *              from the other vertices, so it should be removed from the problem.
 * La_d1      - d1 penalization coefficients, strictly positive, array of length E
 *
 * [CP]
 * difTol     - stopping criterion on iterate evolution.
 *              If  difTol < 1, algorithm stops relative changes of X (in l1 
 *              norm) is less than difTol. If difTol >= 1, algorithm stops if
 *              less than difTol maximum-likelihood labels have changed
 *              1e-2 is a conservative value; 1e-3 or less can give
 *              better precision but with longer computational time
 * itMax      - maximum number of iterations (graph cut and subproblem)
 *              10 cuts solve accurately most problems
 * it         - adress of an integer keeping track of iteration (cut) number
 *
 * [PFDR]
 * rho        - relaxation parameter, 0 < rho < 2
 *              1 is a conservative value; 1.5 often speeds up convergence
 * condMin    - parameter ensuring stability of preconditioning 0 < condMin <= 1
 *              1 is a conservative value; 0.1 or 0.01 might enhance preconditioning
 * difRcd     - reconditioning criterion on iterate evolution.
 *              if difTol < 1, reconditioning occurs if all coordinates of P 
 *              change by less than difRcd. difRcd is then divided by 10.
 *              If difTol >= 1, reconditioning occurs if less than
 *              difRcd maximum-likelihood labels have changed. difRcd is then 
 *              divided by 10.
 *              0 (no reconditioning) is a conservative value, 10*difTol or 
 *              100*difTol might speed up convergence. reconditioning might 
 *              temporarily draw minimizer away from solution, and give bad
 *              subproblem solution
 * difTol     - stopping criterion on iterate evolution.
 *              If  difTol < 1, algorithm stops if relative changes of X (in
 *              l1 norm) is less than difTol. If difTol >= 1, algorithm stops 
 *              if less than difTol maximum-likelihood labels have changed.
 *              1e-3*CP_difTol is a conservative value.
 * itMax      - maximum number of iterations
 *              1e4 iterations provides enough precision for most subproblems
 *
 * Time       - if not NULL, records elapsed time, array of length (CP_itMax + 1)
 * Obj        - if not NULL and N is nonzero, records the values of the objective
 *              functional, array of length (CP_itMax + 1)
 * Dif        - if not NULL, records the iterate evolution (see CP_difTol),
 *              array of length CP_itMax
 * verbose    - if nonzero, display information on the progress, every 'verbose'
 *              iterations
 * CP_restart - pointer to structure (see above) for "warm restart"; rV, Cv 
 *              and rP should be initialized in coherence with rVc, Vc; if the
 *              objective is monitored, the first value is not computed (Obj[0]
 *              not written). set to NULL for no warm restart */
#endif
