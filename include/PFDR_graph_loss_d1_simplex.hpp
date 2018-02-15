/*=============================================================================
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
 * using preconditioned forward-Douglas-Rachford algorithm.
 *
 * Parallel implementation with OpenMP API.
 * 
 * Reference: H. Raguet, A Note on the Forward-Douglas-Rachford Splitting for
 * Monotone Inclusion and Convex Optimization.
 * 
 * Hugo Raguet 2016
 *===========================================================================*/
#ifndef PFDR_GRAPH_LOSS_D1_SIMPLEX_H
#define PFDR_GRAPH_LOSS_D1_SIMPLEX_H

template <typename real>
void PFDR_graph_loss_d1_simplex(const int K, const int V, const int E, \
    const real al, const real *La_f, real *P, const real *Q, \
    const int *Eu, const int *Ev, const real *La_d1, \
    const real rho, const real condMin, \
    real difRcd, const real difTol, const int itMax, int *it, \
    real *Obj, real *Dif, const int verbose);
/* 19 arguments
 * K, V, E  - number of classes, of vertices, of (undirected) edges
 * al, La_f - scalar defining the data-fidelity loss function,
 *            and weights modifying it (only for al > 0), array of length V;
 *            set La_f to NULL for no weight
 *            al = 0, linear:
 *                      f(p) = - <q, p>,
 *              with  <q, p> = sum_{k,v} q_kv p_kv;
 *            0 < al = a < 1, smoothed Kullback-Leibler divergence:
 *                      f(p) = sum_v La_f_v KLa(q_v||p_v),
 *              with KLa(q_v||p_v) = KL(au + (1-a)q_v || au + (1-a)p_v),
 *              where KL is the regular Kullback-Leibler divergence,
 *                    u is the uniform discrete distribution over {1,...,K},
 *                    and a = al is the smoothing parameter.
 *              Up to a constant - H(au + (1-a)q_v))
 *                  = sum_{k} (a/K + (1-a)q_v) log(a/K + (1-a)q_v),
 *              we have KLa(q_v||p_v)
 *                  = - sum_{k} (a/K + (1-a)q_v) log(a/K + (1-a)p_v);
 *            al = 1, quadratic:
 *                      f(p) = 1/2 ||q - p||_{l2,La_f}^2,
 *            with  ||q - p||_{l2,La_f}^2 = sum_{k,v} La_f_v (q_kv - p_kv)^2.
 * P        - minimizer, K-by-V array, column major format
 * Q        - observed probabilities, K-by-V array, column major format
 *            strictly speaking, those are not required to lie on the
 *            simplex, especially with linear fidelity term (al = 0) for which
 *            no weights La_f are taken into account
 * Eu       - for each edge, index of one vertex, array of length E
 * Ev       - for each edge, index of the other vertex, array of length E
 *            Every vertex should belong to at least one edge with a nonzero
 *            penalization coefficient. If it is not the case, the optimal 
 *            value of an isolated vertex is independent from the other 
 *            vertices, so it should be removed from the problem.
 * La_d1    - d1 penalization coefficients for each edge, array of length E
 * rho      - relaxation parameter, 0 < rho < 2
 *            1 is a conservative value; 1.5 often speeds up convergence
 * condMin  - parameter ensuring stability of preconditioning 0 < condMin <= 1
 *            1 is a conservative value; 0.1 or 0.01 might enhance preconditioning
 * difRcd   - reconditioning criterion on iterate evolution.
 *            if difTol < 1, reconditioning occurs relative changes of X (in l1
 *            norm) is less than difRcd. difRcd is then divided by 10.
 *            If difTol >= 1, reconditioning occurs if less than
 *            difRcd maximum-likelihood labels have changed. difRcd is then 
 *            divided by 10.
 *            0 (no reconditioning) is a conservative value, 10*difTol or 
 *            100*difTol might speed up convergence. reconditioning might 
 *            temporarily draw minimizer away from solution; it is advised to 
 *            monitor objective value when using reconditioning
 * difTol   - stopping criterion on iterate evolution.
 *            If  difTol < 1, algorithm stops if relative changes of X (in l1
 *            norm) is less than difTol. If difTol >= 1, algorithm stops if
 *            less than difTol maximum-likelihood labels have changed.
 *            1e-3 is a typical value; 1e-4 or less can give better
 *            precision but with longer computational time.
 * itMax    - maximum number of iterations
 * it       - adress of an integer keeping track of iteration number
 * Obj      - if not NULL, records the value of the objective functional,
 *            array of length (itMax + 1)
 * Dif      - if not NULL, record the iterate evolution (see difTol),
 *            array of length itMax 
 * verbose  - if nonzero, display information on the progress, every 'verbose'
 *            iterations */
#endif
