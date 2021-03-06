/*==================================================================
 * Hugo Raguet 2016
 *================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <cmath> /* for log function, only for computing KLa loss */
#ifdef _OPENMP
    #include <omp.h>
    /* rough minimum number of operations per thread */
    #define MIN_OPS_PER_THREADS 1000
#endif
#ifdef MEX
    #include "mex.h"
    #define FLUSH mexEvalString("drawnow expose")
#else
    #define FLUSH fflush(stdout)
#endif
#include "../include/proj_simplex.hpp"
#include "../include/PFDR_graph_loss_d1_simplex.hpp"

/* constants of the correct type */
#define ZERO ((real) 0.)
#define ONE ((real) 1.)
#define TWO ((real) 2.)
#define HALF ((real) 0.5)
#define ALMOST_TWO ((real) 1.9)
#define TENTH ((real) 0.1)

/* nops is a rough estimation of the total number of operations 
 * max_threads is the maximum number of jobs which can be performed in 
 * parallel */
static int compute_num_threads(const int nops, const int max_threads)
{
#ifdef _OPENMP
    const int m = (omp_get_num_procs() < max_threads) ?
        omp_get_num_procs() : max_threads;
    int n = 1 + (nops - MIN_OPS_PER_THREADS)/MIN_OPS_PER_THREADS;
    return (n < m) ? n : m;
#else
    return 1;
#endif
}

template<typename real>
static void print_progress(char *msg, int it, int itMax, const real dif, \
                                        const real difTol, const real difRcd)
{
    int k = 0;
    while (msg[k++] != '\0'){ printf("\b"); }
    sprintf(msg, "iteration %d (max. %d)\n", it, itMax);
    if (difTol > ZERO || difRcd > ZERO){
        if (difTol >= ONE){
            sprintf(msg, "%slabel evolution %d (recond. %d; tol. %d)\n", \
                                msg, (int) dif, (int) difRcd, (int) difTol);
        }else{
            sprintf(msg, "%siterate evolution %g (recond. %g; tol. %g)\n", \
                                                 msg, dif, difRcd, difTol);
        }
    }
    printf("%s", msg);
    FLUSH;
}

template<typename real>
static void preconditioning(const int K, const int V, const int E, \
    const real al, const real *La_f, const real *P, const real *Q, \
    const int *Eu, const int *Ev, const real *La_d1, \
    real *Ga, real *GaQ, real *Zu, real *Zv, real *Wu, real *Wv, \
    real *W_d1u, real *W_d1v, real *Th_d1, const real rho, const real condMin)
/* 21 arguments 
 * for initialization:
 *      Zu, Zv are NULL
 * for reconditioning:
 *      Zu, Zv are the current auxiliary variables for d1 */
{
    /**  control the number of threads with Open MP  **/
    const int ntVK = compute_num_threads(V*K, V);
    const int ntEK = compute_num_threads(E*K, E);
    const int ntKE = compute_num_threads(K*E, K);

    /**  initialize general variables  **/
    int u, v, e, k, i; /* index edges and vertices */
    real a, b, c; /* general purpose temporary real scalars */
    real *Aux; /* auxiliary pointer */
    real al_K, al_1, al_K_al_1;
    if (ZERO < al && al < ONE){ /* constants for KLa loss */
        al_K = al/K;
        al_1 = ONE - al;
        al_K_al_1 = al_K/al_1;
    }

    if (Zu != NULL){ /* reconditioning */
        /**  retrieve original metric 
         **  normalized after last preconditioning  **/
        if (al == ONE){ /* quadratic loss, GaQ is La_f Ga */
            if (La_f == NULL){
                for (v = 0; v < V*K; v++){ Ga[v] = GaQ[v]; }
            }else{
                #pragma omp parallel for private(v, k, a) schedule(static) \
                    num_threads(ntVK)
                for (v = 0; v < V; v++){
                    a = ONE/La_f[v];
                    for (k = 0; k < K; k++){ Ga[v*K+k] = a*GaQ[v*K+k]; }
                }
            }
        }else if (al > ZERO){ /* KLa loss */
            if (La_f == NULL){
                #pragma omp parallel for private(v) schedule(static) \
                    num_threads(ntVK)
                for (v = 0; v < V*K; v++){ Ga[v] = GaQ[v]/(al_K + al_1*Q[v]); }
            }else{
                for (v = 0; v < V; v++){
                    a = ONE/La_f[v];
                    for (k = 0; k < K; k++){ Ga[v*K+k] = a*GaQ[v*K+k]/(al_K + al_1*Q[v*K+k]); }
                }
            }
        }else{ /* linear loss, GaQ = Ga*Q, but some Q are zero */
            #pragma omp parallel for private(u, v, k, i, a) schedule(static) \
                num_threads(ntVK)
            for (u = 0; u < V; u++){
                v = u*K;
                /* use highest value of Q to improve accuracy */
                i = 0;
                a = Q[v];
                for (k = 1; k < K; k++){
                    if (a < Q[v+k]){
                        a = Q[v+k];
                        i = k;
                    }
                }
                /* retrieve normalization coefficient */
                a = GaQ[v+i]/a/Ga[v+i];
                for (k = 0; k < K; k++){ Ga[v+k] *= a; } 
            }
        }
        /**  get the auxiliary subgradients  **/
        #pragma omp parallel for private(e, i, u, v, k) schedule(static) \
            num_threads(ntEK)
        for (e = 0; e < E; e++){
            u = Eu[e]*K;
            v = Ev[e]*K;
            i = e*K;
            for (k = 0; k < K; k++){ 
                if (al == ZERO){ /* linear loss, grad = -Q */
                    Zu[i] = (Wu[i]/Ga[u])*(P[u] + GaQ[u] - Zu[i]);
                    Zv[i] = (Wv[i]/Ga[v])*(P[v] + GaQ[v] - Zv[i]);
                }else if (al == ONE){ /* quadratic loss, grad = La_f (P - Q) */
                    Zu[i] = (Wu[i]/Ga[u])*(P[u] - GaQ[u]*(P[u] - Q[u]) - Zu[i]);
                    Zv[i] = (Wv[i]/Ga[v])*(P[v] - GaQ[v]*(P[v] - Q[v]) - Zv[i]);
                }else{ /* dKLa/dp_k = -(1-a)(a/K + (1-a)q_k)/(a/K + (1-a)p_k) */
                    Zu[i] = (Wu[i]/Ga[u])*(P[u] + GaQ[u]/(al_K_al_1 + P[u]) - Zu[i]);
                    Zv[i] = (Wv[i]/Ga[v])*(P[v] + GaQ[v]/(al_K_al_1 + P[v]) - Zv[i]);
                }
                i++; u++; v++;
            }
        }
    }
    
    /**  compute the Hessian  **/
    if (al == ZERO){ /* linear loss, H = 0 */
        for (v = 0; v < V*K; v++){ Ga[v] = ZERO; }
    }else if (al == ONE){ /* quadratic loss, H = La_f */
        if (La_f == NULL){
            for (v = 0; v < V*K; v++){ Ga[v] = ONE; }
        }else{
            for (v = 0; v < V; v++){
                a = La_f[v];
                for (k = 0; k < K; k++){ Ga[v*K+k] = a; }
            }
        }
    }else{ /* d^2KLa/dp_k^2 = (1-a)^2 (a/K + (1-a)q_k)/(a/K + (1-a)p_k)^2 */
        if (La_f == NULL){
            #pragma omp parallel for private(v, a) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V*K; v++){
                a = (al_K_al_1 + P[v]);
                Ga[v] = (al_K + al_1*Q[v])/(a*a);
            }
        }else{
            #pragma omp parallel for private(v, a, b, k) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V; v++){
                b = La_f[v];
                for (k = 0; k < K; k++){
                    a = (al_K_al_1 + P[v*K+k]);
                    Ga[v*K+k] = b*(al_K + al_1*Q[v*K+k])/(a*a);
                }
            }
        }
    }

    /**  d1 contribution and splitting weights  **/
    if (al == ZERO){ /* linear case, compute directly pseudo hessian */
        Aux = Ga;
    }else{ /* use GaQ as temporary storage */
        for (v = 0; v < V*K; v++){ GaQ[v] = ZERO; } 
        Aux = GaQ;
    }
    /* this task cannot be easily parallelized along the edges */
    #pragma omp parallel for private(k, e, u, v, i, a) schedule(static) \
        num_threads(ntKE)
    for (k = 0; k < K; k++){
        i = k;
        for (e = 0; e < E; e++){
            u = Eu[e]*K + k;
            v = Ev[e]*K + k;
            if (Zu == NULL){ /* first preconditioning */
                a = La_d1[e];
            }else{ /* reconditioning */
                a = P[u] - P[v];
                if (a < ZERO){ a = -a; }
                if (a < condMin){ a = condMin; }
                a = La_d1[e]/a;
            }
            Aux[u] += a;
            Aux[v] += a;
            Wu[i] = a;
            Wv[i] = a;
            i += K;
        }
    }
    if (al > ZERO){ /* add contribution to the Hessian */
        #pragma omp parallel for private(v) schedule(static) num_threads(ntVK)
        for (v = 0; v < V*K; v++){ Ga[v] += Aux[v]; }
    }
    /* inverse the sum of the weights */
    #pragma omp parallel for private(v) schedule(static) num_threads(ntVK)
    for (v = 0; v < V*K; v++){ Aux[v] = ONE/Aux[v]; }
    /* make splitting weights sum to unity */
    #pragma omp parallel for private(e, i, u, v, k) schedule(static) \
        num_threads(ntEK)
    for (e = 0; e < E; e++){
        u = Eu[e]*K;
        v = Ev[e]*K;
        i = e*K;
        for (k = 0; k < K; k++){
            Wu[i] *= Aux[u];
            Wv[i] *= Aux[v];
            i++; u++; v++;
        }
    }

    /**  inverse the approximate of the Hessian  **/
    if (al > ZERO){
        #pragma omp parallel for private(v) schedule(static) num_threads(ntVK)
        for (v = 0; v < V*K; v++){ Ga[v] = ONE/Ga[v]; }
    } /* linear case already inverted */

    /**  convergence condition on the metric  **/
    a = ALMOST_TWO*(TWO - rho);
    if (al == ONE){ /* quadratic loss, L = La_f */
        if (La_f != NULL){
            #pragma omp parallel for private(v, k, c) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V; v++){
                c = a/La_f[v];
                for (k = 0; k < K; k++){ if (Ga[v*K+k] > c){ Ga[v*K+k] = c; } }
            }
        }else if (a < ONE){
            #pragma omp parallel for private(v) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V*K; v++){ if (Ga[v] > a){ Ga[v] = a; } }
        } /* else Ga already less than 1/al */
    }else if (al > ZERO){ /* KLa loss, Lk = max_{0<=p_k<=1} d^2KLa/dp_k^2
                           *              = (1-a)^2 (a/K + (1-a)q_k)/(a/K)^2 */
        if (La_f == NULL){
            b = ONE/(al_K_al_1*al_K_al_1);
            #pragma omp parallel for private(v, c) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V*K; v++){
                c = a/((al_K + al_1*Q[v])*b);
                if (Ga[v] > c){ Ga[v] = c; }
            }
        }else{
            #pragma omp parallel for private(v, k, b, c) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V; v++){
                b = La_f[v]*ONE/(al_K_al_1*al_K_al_1);
                for (k = 0; k < K; k++){
                    c = a/((al_K + al_1*Q[v*K+k])*b);
                    if (Ga[v*K+k] > c){ Ga[v*K+k] = c; }
                }
            }
        }
    } /* linear loss, L = 0 */

    /**  precompute some quantities  **/
    if (al > ZERO){ /* weights and thresholds for d1 prox */
        #pragma omp parallel for private(e, i, u, v, k, a, b) \
            schedule(static) num_threads(ntEK)
        for (e = 0; e < E; e++){
            u = Eu[e]*K;
            v = Ev[e]*K;
            i = e*K;
            a = La_d1[e];
            for (k = 0; k < K; k++){
                W_d1u[i] = Wu[i]/Ga[u];
                W_d1v[i] = Wv[i]/Ga[v];
                b = W_d1u[i] + W_d1v[i];
                Th_d1[i] = a*b/(W_d1u[i]*W_d1v[i]);
                W_d1u[i] /= b;
                W_d1v[i] /= b;
                i++; u++; v++;
            }
        }
    } /* linear loss: weights are all 1/2 and thresholds are all 2 */
    /* metric and first order information */
    if (al == ZERO){ /* linear loss, grad = -Q */
        #pragma omp parallel for private(v) schedule(static) num_threads(ntVK)
        for (v = 0; v < V*K; v++){ GaQ[v] = Ga[v]*Q[v]; }
    }else if (al == ONE){ /* quadratic loss, GaQ is La_f Ga */
        if (La_f == NULL){
            for (v = 0; v < V*K; v++){ GaQ[v] = Ga[v]; }
        }else{
            #pragma omp parallel for private(v, k, a) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V; v++){
                a = La_f[v];
                for (k = 0; k < K; k++){ GaQ[v*K+k] = a*Ga[v*K+k]; }
            }
        }
    }else{ /* dKLa/dp_k = -(1-a)(a/K + (1-a)q_k)/(a/K + (1-a)p_k) */
        if (La_f == NULL){
            #pragma omp parallel for private(v) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V*K; v++){ GaQ[v] = Ga[v]*(al_K + al_1*Q[v]); }
        }else{
            #pragma omp parallel for private(v, k, a) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V; v++){
                a = La_f[v];
                for (k = 0; k < K; k++){ GaQ[v*K+k] = a*Ga[v*K+k]*(al_K + al_1*Q[v*K+k]); }
            }
        }
    }

    if (Zu != NULL){ /**  update auxiliary variables  **/
        #pragma omp parallel for private(e, i, u, v, k) schedule(static) \
            num_threads(ntEK)
        for (e = 0; e < E; e++){
            u = Eu[e]*K;
            v = Ev[e]*K;
            i = e*K;
            for (k = 0; k < K; k++){ 
                if (al == ZERO){ /* linear loss, grad = -Q */
                    Zu[i] = P[u] + GaQ[u] - (Ga[u]/Wu[i])*Zu[i];
                    Zv[i] = P[v] + GaQ[v] - (Ga[v]/Wv[i])*Zv[i];
                }else if (al == ONE){ /* quadratic loss, grad = La_f (P - Q) */
                    Zu[i] = P[u] - GaQ[u]*(P[u] - Q[u] + Zu[i]/Wu[i]);
                    Zv[i] = P[v] - GaQ[v]*(P[v] - Q[v] + Zv[i]/Wv[i]);
                }else{ /* dKLa/dp_k = -(1-a)(a/K + (1-a)q_k)/(a/K + (1-a)p_k) */
                    Zu[i] = P[u] + GaQ[u]/(al_K_al_1 + P[u]) - (Ga[u]/Wu[i])*Zu[i];
                    Zv[i] = P[v] + GaQ[v]/(al_K_al_1 + P[v]) - (Ga[v]/Wv[i])*Zv[i];
                }
                i++; u++; v++;
            }
        }
    }

    /** normalize metric to avoid machine precision trouble
     ** when projecting onto simplex  **/
    #pragma omp parallel for private(u, v, k, a) schedule(static) \
        num_threads(ntVK)
    for (u = 0; u < V; u++){
        v = u*K;
        a = Ga[v];
        for (k = 1; k < K; k++){ if (Ga[v+k] > a){ a = Ga[v+k]; } }
        for (k = 0; k < K; k++){ Ga[v+k] /= a; }
    }
}

template <typename real>
void PFDR_graph_loss_d1_simplex(const int K, const int V, const int E, \
    const real al, const real *La_f, real *P, const real *Q, \
    const int *Eu, const int *Ev, const real *La_d1, \
    const real rho, const real condMin, \
    real difRcd, const real difTol, const int itMax, int *it, \
    real *Obj, real *Dif, const int verbose)
/* 19 arguments */
{
    /***  initialize general variables  ***/
    if (verbose){ printf("Initializing constants and variables... "); FLUSH; }
    int u, v, e, i, k; /* index edges and vertices */
    real a, b, c; /* general purpose temporary real scalars */
    const real one = ONE; /* argument for simplex projection */
    real al_K, al_1, al_K_al_1;
    if (ZERO < al && al < ONE){ /* constants for KLa loss */
        al_K = al/K;
        al_1 = ONE - al;
        al_K_al_1 = al_K/al_1;
    }

    /***  control the number of threads with Open MP  ***/
    const int ntVK = compute_num_threads(V*K, V);
    const int ntEK = compute_num_threads(E*K, E);
    const int ntKE = compute_num_threads(K*E, K);

    /**  allocates general purpose arrays  **/
    real *Ga = (real*) malloc(K*V*sizeof(real)); /* descent metric */
    real *GaQ = (real*) malloc(K*V*sizeof(real)); /* metric and first order information */
    /* store explicit step */
    real *FP = (real*) malloc(K*V*sizeof(real));
    /* auxiliary variables for generalized forward-backward */
    real *Zu = (real*) malloc(K*E*sizeof(real));
    real *Zv = (real*) malloc(K*E*sizeof(real));
    /* splitting weights for generalized forward-backward */
    real *Wu = (real*) malloc(K*E*sizeof(real));
    real *Wv = (real*) malloc(K*E*sizeof(real));
    real *W_d1u, *W_d1v, *Th_d1;
    if (al > ZERO){ /* weights and thresholds for d1 prox */
        W_d1u = (real*) malloc(K*E*sizeof(real));
        W_d1v = (real*) malloc(K*E*sizeof(real));
        Th_d1 = (real*) malloc(K*E*sizeof(real));
    }else{
        W_d1u = W_d1v = Th_d1 = NULL;
    }
    /* initialize p *//* assumed already initialized */
    /* initialize, for all i, z_i = x */
    #pragma omp parallel for private(e, u, v, k, i) schedule(static) \
        num_threads(ntEK)
    for (e = 0; e < E; e++){
        u = Eu[e]*K;
        v = Ev[e]*K;
        i = e*K;
        for (k = 0; k < K; k++){
            Zu[i] = P[u];
            Zv[i] = P[v];
            i++; u++; v++;
        }
    }
    if (verbose){ printf("done.\n"); FLUSH; }

    /***  preconditioning  ***/
    if (verbose){ printf("Preconditioning... "); FLUSH; }
    preconditioning<real>(K, V, E, al, La_f, P, Q, Eu, Ev, La_d1, Ga, GaQ, \
                      NULL, NULL, Wu, Wv, W_d1u, W_d1v, Th_d1, rho, condMin);
    if (verbose){ printf("done.\n"); FLUSH; }

    /***  forward-Douglas-Rachford  ***/
    if (verbose){ printf("Preconditioned forward-Douglas-Rachford algorithm\n"); FLUSH; }
    /* initialize */
    int itMsg, it_ = 0;
    real dif, *P_ = NULL; /* store last iterate */
    char msg[256];
    dif = (difTol > difRcd) ? difTol : difRcd;
    if (difTol > ZERO || difRcd > ZERO || Dif != NULL){
        if (difTol >= ONE){
            P_ = (real*) malloc(V*sizeof(real));
            /* compute maximum-likelihood labels */
            #pragma omp parallel for private(u, v, k, a) schedule(static) \
                num_threads(ntVK)
            for (u = 0; u < V; u++){
                v = u*K;
                a = P[v];
                P_[u] = (real) 0;
                for (k = 1; k < K; k++){
                    if (P[v+k] > a){
                        a = P[v+k];
                        P_[u] = (real) k;
                    }
                }
            }
        }else{
            P_ = (real*) malloc(K*V*sizeof(real));
            for (v = 0; v < K*V; v++){ P_[v] = P[v]; }
        }
    }
    if (verbose){
        msg[0] = '\0';
        itMsg = 0;
    }

    /***  main loop  ***/
    while (true){

        /**  objective functional value  **/
        if (Obj != NULL){ 
            a = ZERO;
            if (al == ZERO){ /* linear loss */
                #pragma omp parallel for private(v) reduction(+:a) \
                    schedule(static) num_threads(ntVK)
                for (v = 0; v < V*K; v++){ a -= P[v]*Q[v]; }
            }else if (al == ONE){ /* quadratic loss */
                if (La_f == NULL){
                    #pragma omp parallel for private(v, c) reduction(+:a) \
                        schedule(static) num_threads(ntVK)
                    for (v = 0; v < V*K; v++){
                        c = P[v] - Q[v];
                        a += c*c;
                    }
                }else{
                    #pragma omp parallel for private(v, k, i, b, c) \
                        reduction(+:a) schedule(static) num_threads(ntVK)
                    for (v = 0; v < V; v++){
                        i = v*K; 
                        b = ZERO;
                        for (k = 0; k < K; k++){
                            c = P[i+k] - Q[i+k];
                            b += c*c;
                        }
                        a += La_f[v]*b;
                    }
                }
                a *= HALF;
            }else{ /* KLa loss */
                if (La_f == NULL){
                    #pragma omp parallel for private(v, c) reduction(+:a) \
                        schedule(static) num_threads(ntVK)
                    for (v = 0; v < V*K; v++){
                        c = al_K + al_1*Q[v];
                        a += c*log(c/(al_K + al_1*P[v]));
                    }
                }else{
                    #pragma omp parallel for private(v, k, b, c) \
                        reduction(+:a) schedule(static) num_threads(ntVK)
                    for (v = 0; v < V; v++){
                        b = ZERO; 
                        for (k = 0; k < K; k++){
                            c = al_K + al_1*Q[v*K+k];
                            b += c*log(c/(al_K + al_1*P[v*K+k]));
                        }
                        a += La_f[v]*b;
                    }
                }
            }
            Obj[it_] = a;
            /* ||x||_{d1,La_d1} */
            a = ZERO;
            #pragma omp parallel for private(e, u, v, b, c) reduction(+:a) \
                schedule(static) num_threads(ntEK)
            for (e = 0; e < E; e++){
                u = Eu[e]*K;
                v = Ev[e]*K;
                b = ZERO;
                for (k = 0; k < K; k++){
                    c = P[u] - P[v];
                    if (c < ZERO){ b -= c; }
                    else{ b += c; }
                    u++; v++;
                }
                a += La_d1[e]*b;
            }
            Obj[it_] += a;
        }

        /**  progress and stopping criterion  **/
        if (verbose && itMsg++ == verbose){
            print_progress<real>(msg, it_, itMax, dif, difTol, difRcd);
            itMsg = 1;
        }
        if (it_ == itMax || dif < difTol){ break; }

        /**  reconditioning  **/
        if (dif < difRcd){
            if (verbose){
                print_progress<real>(msg, it_, itMax, dif, difTol, difRcd);
                printf("Reconditioning... ");
                FLUSH;
                msg[0] = '\0';
            }
            preconditioning<real>(K, V, E, al, La_f, P, Q, Eu, Ev, La_d1, Ga, GaQ, \
                            Zu, Zv, Wu, Wv, W_d1u, W_d1v, Th_d1, rho, condMin);
            difRcd *= TENTH;
            if (verbose){ printf("done.\n"); FLUSH; }
        }

        /**  forward and backward steps on auxiliary variables  **/
        /* explicit step */
        if (al == ZERO){ /* linear loss, grad = -Q */
            #pragma omp parallel for private(v) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V*K; v++){
                FP[v] = TWO*P[v] + GaQ[v];
            }
        }else if (al == ONE){ /* quadratic loss, grad = al(P - Q) */
            #pragma omp parallel for private(v) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V*K; v++){
                FP[v] = TWO*P[v] - GaQ[v]*(P[v] - Q[v]);
            }
        }else{ /* dKLa/dp_k = -(1-a)(a/K + (1-a)q_k)/(a/K + (1-a)p_k) */
            #pragma omp parallel for private(v) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V*K; v++){
                FP[v] = TWO*P[v] + GaQ[v]/(al_K_al_1 + P[v]);
            }
        }

        /* implicit step for d1 penalization */
        #pragma omp parallel for private(e, u, v, k, i, a, b, c) \
            schedule(static) num_threads(ntEK)
        for (e = 0; e < E; e++){
            u = Eu[e]*K;
            v = Ev[e]*K;
            i = e*K;
            for (k = 0; k < K; k++){
                a = FP[u] - Zu[i];
                b = FP[v] - Zv[i];
                if (al == ZERO){ /* weights are all 1/2 and thresholds are all 2 */
                    c = HALF*(a + b); /* weighted average */
                    a = a - b; /* finite difference */
                    /* soft thresholding, update and relaxation */
                    if (a > TWO){
                        a = HALF*(a - TWO);
                        Zu[i] += rho*(c + a - P[u]);
                        Zv[i] += rho*(c - a - P[v]);
                    }else if (a < -TWO){
                        a = HALF*(a + TWO);
                        Zu[i] += rho*(c + a - P[u]);
                        Zv[i] += rho*(c - a - P[v]);
                    }else{
                        Zu[i] += rho*(c - P[u]);
                        Zv[i] += rho*(c - P[v]);
                    }
                }else{
                    c = W_d1u[i]*a + W_d1v[i]*b; /* weighted average */
                    a = a - b; /* finite difference */
                    /* soft thresholding, update and relaxation */
                    if (a > Th_d1[i]){
                        a -= Th_d1[i];
                        Zu[i] += rho*(c + W_d1v[i]*a - P[u]);
                        Zv[i] += rho*(c - W_d1u[i]*a - P[v]);
                    }else if (a < -Th_d1[i]){
                        a += Th_d1[i];
                        Zu[i] += rho*(c + W_d1v[i]*a - P[u]);
                        Zv[i] += rho*(c - W_d1u[i]*a - P[v]);
                    }else{
                        Zu[i] += rho*(c - P[u]);
                        Zv[i] += rho*(c - P[v]);
                    }
                }
                i++; u++; v++;
            }
        }

        /** average **/
        for (v = 0; v < V*K; v++){ P[v] = ZERO; }
        /* this task cannot be easily parallelized along the edges */
        #pragma omp parallel for private(k, e, i) schedule(static) \
            num_threads(ntKE)
        for (k = 0; k < K; k++){
            i = k;
            for (e = 0; e < E; e++){
                P[Eu[e]*K+k] += Wu[i]*Zu[i];
                P[Ev[e]*K+k] += Wv[i]*Zv[i];
                i += K;
            }
        }

        /**  projection on simplex  **/
        proj_simplex_metric<real>(P, Ga, K, V, V, &one, 1);

        /**  iterate evolution  **/
        if (difTol > ZERO || difRcd > ZERO || Dif != NULL){
            dif = ZERO;
            if (difTol >= ONE){
                #pragma omp parallel for private(u, v, k, i, a) \
                    reduction(+:dif) schedule(static) num_threads(ntVK)
                for (u = 0; u < V; u++){
                    v = u*K;
                    /* get maximum likelihood label */
                    a = P[v];
                    i = 0;
                    for (k = 1; k < K; k++){
                        if (P[v+k] > a){
                            a = P[v+k];
                            i = k;
                        }
                    }
                    /* compare with previous and update */
                    a = (real) i;
                    if (a != P_[u]){
                        dif += ONE;
                        P_[u] = a;
                    }
                }
            }else{
                /* max reduction available in C since OpenMP 3.1 and gcc 4.7 */
                #pragma omp parallel for private(v, a) schedule(static) \
                    num_threads(ntVK) /* reduction(max:dif) */ reduction(+:dif)
                for (v = 0; v < V*K; v++){
                    a = P_[v] - P[v];
                    if (a < ZERO){ a = -a; }
                    /* if (a > dif){ dif = a; } */ /* max norm */
                    dif += a; /* relative l1 norm evolution */
                    P_[v] = P[v];
                }
                dif /= V; /* relative l1 norm evolution */
            }
            if (Dif != NULL){ Dif[it_] = dif; }
        }

        it_++;
    } /* endwhile (true) */

    /* final information */
    *it = it_;
    if (verbose){
        print_progress<real>(msg, it_, itMax, dif, difTol, difRcd);
        FLUSH;
    }

    /* free stuff */
    free(Ga);
    free(GaQ);
    free(FP);
    free(Zu);
    free(Zv);
    free(Wu);
    free(Wv);
    free(W_d1u);
    free(W_d1v);
    free(Th_d1);
    free(P_);
}

/* instantiate for compilation */
template void PFDR_graph_loss_d1_simplex<float>(const int, const int, const int, \
        const float, const float*, float*, const float*, const int*, const int*, \
        const float*, const float, const float, float, const float, \
        const int, int*, float*, float*, const int);

template void PFDR_graph_loss_d1_simplex<double>(const int, const int, const int, \
        const double, const double*, double*, const double*, const int*, const int*, \
        const double*, const double, const double, double, const double, \
        const int, int*, double*, double*, const int);
