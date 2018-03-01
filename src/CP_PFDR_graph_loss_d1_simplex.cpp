/*=============================================================================
 * Hugo Raguet 2016
 *===========================================================================*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <float.h>
#ifdef _OPENMP
    #include <omp.h>
    /* rough minimum number of operations per thread */
    #define MIN_OPS_PER_THREADS 1000
#endif
#ifdef MEX
    #include "mex.h"
    #define FLUSH mexEvalString("drawnow expose")
    #define CALLOC mxCalloc
    #define FREE mxFree
#else
    #define FLUSH fflush(stdout)
    #define CALLOC calloc
    #define FREE free
#endif
#include "../include/graph.hpp" /* Boykov-Kolmogorov graph class */
#include "../include/PFDR_graph_loss_d1_simplex.hpp"

/* constants of the correct type */
#define ZERO ((real) 0.)
#define ONE ((real) 1.)
#define TWO ((real) 2.)
#define HALF ((real) 0.5)
#define TRUE ((uint8_t) 1)
#define FALSE ((uint8_t) 0)

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
static void print_progress(int it, int itMax, double t, \
    const int rV, const int rE, const real dif, const real difTol)
{
    printf("\n\tCut pursuit iteration %d (max. %d)\n", it, itMax);
    if (difTol >= ONE){
        printf("\tlabel evolution %d (tol. %d)\n", (int) dif, (int) difTol);
    }else if (difTol > ZERO){
        printf("\titerate evolution %g (tol. %g)\n", dif, difTol);
    }
    printf("\t%d connected component(s), %d reduced edge(s)\n", rV, rE);
    if (t > 0.){ printf("\telapsed time %.1f s\n", t); }
    FLUSH;
}

template<typename real>
static void initialize(const int K, const int V, const int E, \
    const real al, int *rV, int *Cv, real **rP, const real *Q, \
    const int *Eu, const int *Ev, \
    Graph<real, real, real> **G, int **Vc, int **rVc, real *obj)
{
    /**  initialize general variables  **/
    int u, v, e, i, k; /* vertices, edges, indices */
    real a, b, c; /* general purpose temporary real scalar */
    real *rQ;
    real al_K, al_1;
    if (ZERO < al && al < ONE){ /* constants for KLa loss */
        al_K = al/K;
        al_1 = ONE - al;
    }

    /** control the number of threads with Open MP **/
    const int ntVK = compute_num_threads(V*K, V);
    const int ntKV = compute_num_threads(V*K, K);;

    /**  construct graph **/
    Graph<real, real, real> *H = new Graph<real, real, real>(V, E); 
    H->add_node(V);
    /* d1 edges */
    for (e = 0; e < E; e++){ H->add_edge(Eu[e], Ev[e], ZERO, ZERO); }
    /* source/sink edges */
    for (v = 0; v < V; v++){ H->add_tweights(v, ZERO, ZERO); }
    *G = H;

    /**  solve 'unisimplicial' problem  **/
    *rP = (real*) CALLOC(K, sizeof(real));
    rQ = (real*) calloc(K, sizeof(real));
    #pragma omp parallel for private(k, v) schedule(static) num_threads(ntKV)
    for (k = 0; k < K; k++){
        for (v = k; v < K*V; v += K){ rQ[k] += Q[v]; }  /* sum over vertices */
    }
    if (al == ZERO){ /* linear loss, optimum at simplex corner */
        a = rQ[i = 0];
        for (k = 1; k < K; k++){ if (rQ[k] > a){ a = rQ[i = k]; } }
        for (k = 0; k < K; k++){ (*rP)[k] = (k == i) ? ONE : ZERO; }
    }else{ /* quadratic or KLa loss, optimum at barycenter */
        for (k = 0; k < K; k++){ (*rP)[k] = rQ[k]/V; }
    }

    /**  assign every vertex to the unique component  **/
    *rV = 1;
    for (v = 0; v < V; v++){ Cv[v] = 0; }
    *Vc = (int*) malloc(V*sizeof(int));
    for (v = 0; v < V; v++){ (*Vc)[v] = v; }
    *rVc = (int*) malloc(2*sizeof(int));
    (*rVc)[0] = 0; (*rVc)[1] = V;
    
    /**  objective functional value  **/
    if (obj != NULL){
        a = ZERO;
        if (al == ZERO){ /* linear loss */
            a = -rQ[i]; /* i is still the maximum index */
        }else if (al == ONE){ /* quadratic loss */
            #pragma omp parallel for private(k, v, b, c) reduction(+:a) \
                schedule(static) num_threads(ntKV)
            for (k = 0; k < K; k++){
                b = (*rP)[k];
                for (v = 0; v < V*K; v++){
                    c = Q[v] - b;
                    a += c*c;
                }
                a *= HALF;
            }
        }else{ /* KLa loss */
            #pragma omp parallel for private(k, v, b, c) reduction(+:a) \
                schedule(static) num_threads(ntKV)
            for (k = 0; k < K; k++){
                b = (*rP)[k];
                for (v = 0; v < V*K; v++){
                    c = al_K + al_1*Q[v];
                    a += c*log(c/(al_K + al_1*b));
                }
            }
        }
        *obj = a;
    }
    free(rQ);
}

template <typename real> struct CPls_Restart
{
    Graph<real, real, real> *G;
    int *Vc;
    int *rVc;
};

template <typename real> struct CPls_Restart<real>*
create_CPls_Restart(const int K, const int V, const int E, const real al, \
                    int *rV, int *Cv, real **rP, const real *Q, \
                    const int *Eu, const int *Ev)
{
    struct CPls_Restart<real> *CP_restart = 
        (struct CPls_Restart<real>*) malloc(sizeof(struct CPls_Restart<real>));
    Graph<real, real, real> *G;
    real *Vc, *rVc;
    initialize(K, V, E, al, rV, Cv, rP, Q, Eu, Ev, &G, &Vc, &rVc, NULL);
    CP_restart->G = G;
    CP_restart->Vc = Vc;
    CP_restart->rVc = rVc;
    return CP_restart;
}

template <typename real> void free_CPls_Restart(struct CPls_Restart<real> *CP_restart)
{
    delete CP_restart->G;
    free(CP_restart->Vc);
    free(CP_restart->rVc);
    free(CP_restart);
}

/* switch between exact directional derivative and an underestimation
 * unfortunately using the actual derivative seems less efficient;
 * maybe more descent directions should be searched */
/* #define EXACT_DERIV 0 */ /* not used anymore */

template <typename real>
void CP_PFDR_graph_loss_d1_simplex(const int K, const int V, const int E, \
    const real al, int *rV, int *Cv, real **rP, const real *Q, \
    const int *Eu, const int *Ev, const real *La_d1, \
    const real CP_difTol, const int CP_itMax, int *CP_it, \
    const real PFDR_rho, const real PFDR_condMin, \
    const real PFDR_difRcd, const real PFDR_difTol, const int PFDR_itMax, \
    double *Time, real *Obj, real *Dif, const int verbose, \
    struct CPls_Restart<real> *CP_restart)
/* 25 arguments */
{
    /***  initialize main graph and general variables  ***/
    if (verbose){
        printf("\tInitializing constants, variables, graph structure " \
               "and solution of reduced problem... ");
        FLUSH;
    }
    int s, t, u, v, w, e, ru, rv, re, i, j, k, l, n; /* vertices, edges, indices, ... */
    typename Graph<real, real, real>::arc_id ee; /* pointer over edge in graph structure */
    real a, b, c, d; /* general purpose temporary real scalars */
    const real one = ONE; /* argument for simplex projection */
    real al_K, al_1, al_K_al_1;
    if (ZERO < al && al < ONE){ /* constants for KLa loss */
        al_K = al/K;
        al_1 = ONE - al;
        al_K_al_1 = al_K/al_1;
    }

    /**  precision on finite differences  **/
    switch (sizeof(real)){
        case sizeof(float) :
            a = (real) FLT_EPSILON;
            break;
        case sizeof(double) :
            a = (real) DBL_EPSILON;
            break;
        case sizeof(long double) :
            a = (real) LDBL_EPSILON;
            break;
        default :
            a = (real) FLT_EPSILON;
    }
    b = (CP_difTol >= ONE) ? CP_difTol/V : CP_difTol;
    c = (PFDR_difTol >= ONE) ? PFDR_difTol/V : PFDR_difTol;
    c = (b < c) ? b : c;
    const real eps = (ZERO < c < a) ? c : a;

    /** monitor elapsing time **/
    double timer = 0.;
    struct timespec time0, timeIt;
    if (Time != NULL){ clock_gettime(CLOCK_MONOTONIC, &time0); }

    /** control the number of threads with Open MP **/
    const int ntVK = compute_num_threads(V*K, V);
    const int ntE = compute_num_threads(E, E);
    int ntrV, ntrEK, ntrVK, ntVK_rV;
    ntrV = ntrEK = ntrVK = ntVK_rV = 1;

    /**  some meaningful pointers  **/
    real *DfS; /* gradient of differentiable part */
    real *rQ, *rLa_d1, *rLa_f; /* reduced observation or gradient,
                                * reduced penalizations, quadratic weights */
    rQ = rLa_d1 = rLa_f = NULL;
    int *rEu, *rEv, *rEc; /* for reduced edge set */
    rEu = rEv = NULL;
    int *Vc; /* list of vertices within each connected component */
    int *rVc; /* cumulative sum of the components sizes */
    /* #if EXACT_DERIV
        int *Div, *Djv; / * indices for descent directions * /
    #endif */
    int *rDi, *Djv; /* indices for descent directions */

    /***  cut pursuit initialization  ***/
    int rV_, rE = 0; /* number of vertices and edges in reduced graph */
    Graph<real, real, real> *G;
    if (CP_restart == NULL){
        initialize(K, V, E, al, &rV_, Cv, rP, Q, Eu, Ev, &G, &Vc, &rVc, Obj);
        rQ = (real*) malloc(K*sizeof(real));
    }else{ /**  warm restart  **/
        rV_ = *rV;
        G = CP_restart->G;
        Vc = CP_restart->Vc;
        rVc = CP_restart->rVc;
        rQ = (real*) malloc(rV_*K*sizeof(real));
        /* Cv, rP are supposed to be initialized accordingly */
    }
    if (verbose){ printf("done.\n"); FLUSH; }

    /***  cut pursuit  ***/
    /* initialize */
    int PFDR_it = PFDR_itMax, CP_it_ = 0;
    int *Cv_ = NULL; /* store last iterate partition */
    real dif, *rP_ = NULL; /* store last iterate values */
    real *rPu, *rPv, *rQv, *DfSv; /* auxiliary pointer to vertices */
    const real *Qv;
    int *L; /* for sorting labels within components */
    dif = CP_difTol > ONE ? CP_difTol : ONE;
    if (CP_difTol > ZERO || Dif != NULL){
        if (CP_difTol >= ONE){
            rP_ = (real*) malloc(rV_*sizeof(real));
            /* compute maximum-likelihood labels */
            #pragma omp parallel for private(rv, rPv, a, k) schedule(static) \
                num_threads(ntrVK)
            for (rv = 0; rv < rV_; rv++){
                rPv = (*rP) + rv*K;
                a = rPv[0];
                rP_[rv] = (real) 0;
                for (k = 1; k < K; k++){
                    if (rPv[k] > a){
                        a = rPv[k];
                        rP_[rv] = (real) k;
                    }
                }
            }
        }else{
            rP_ = (real*) malloc(rV_*K*sizeof(real));
            for (rv = 0; rv < rV_*K; rv++){ rP_[rv] = (*rP)[rv]; }
        }
        Cv_ = (int*) malloc(V*sizeof(int));
        for (v = 0; v < V; v++){ Cv_[v] = Cv[v]; }
    }
    /***  main loop  ***/
    while (true){

        /**  elapsed time  **/
        if (Time != NULL){
            clock_gettime(CLOCK_MONOTONIC, &timeIt);
            timer = timeIt.tv_sec - time0.tv_sec;
            timer += (timeIt.tv_nsec - time0.tv_nsec)/1000000000.;
            Time[CP_it_] = timer;
        }

        /**  stopping criteria and information  **/
        if (verbose){
            print_progress<real>(CP_it_, CP_itMax, timer, rV_, rE, dif, CP_difTol);
        }
        if (CP_it_ == CP_itMax || dif < CP_difTol){ break; }
        
        /***  steepest cuts  ***/
        if (verbose){ printf("\tSteepest cut:\n"); FLUSH; }

        /**  compute gradient of loss term  **/ 
        if (verbose){ printf("\tCompute gradient of differentiable part... "); FLUSH; }
        DfS = (real*) malloc(V*K*sizeof(real));
        if (al == ZERO){ /* linear loss, grad = -Q */
            #pragma omp parallel for private(v) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V*K; v++){ DfS[v] = -Q[v]; }
        }else if (al == ONE){ /* quadratic loss, grad = P - Q */
            #pragma omp parallel for private(v, DfSv, rPv, Qv, k) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V; v++){
                DfSv = DfS + v*K;
                rPv = (*rP) + Cv[v]*K;
                Qv = Q + v*K;
                for (k = 0; k < K; k++){ DfSv[k] = rPv[k] - Qv[k]; }
            }
        }else{ /* dKLa/dp_k = -(1-a)(a/K + (1-a)q_k)/(a/K + (1-a)p_k) */
            #pragma omp parallel for private(v, DfSv, rPv, Qv, k) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V; v++){
                DfSv = DfS + v*K;
                rPv = (*rP) + Cv[v]*K;
                Qv = Q + v*K;
                for (k = 0; k < K; k++){
                    DfSv[k] = -(al_K + al_1*Qv[k])/(al_K_al_1 + rPv[k]);
                }
            }
        }

        /**  add the differentiable d1 contribution  **/ 
        #pragma omp parallel for \
            private(u, v, k, ee, e, a, d, rPu, rPv, DfSv) schedule(static) \
            num_threads(ntVK)
        for (v = 0; v < V; v++){
            rPv = (*rP) + Cv[v]*K;
            DfSv = DfS + v*K;
            for (ee = G->nodes[v].first; ee; ee = ee->next){
                if (ee->is_active){
                    u = (int) (ee->head - G->nodes);
                    rPu = (*rP) + Cv[u]*K;
                    e = ((int) (ee - G->arcs))/2;
                    a = La_d1[e];
                    for (k = 0; k < K; k++){
                        d = rPv[k] - rPu[k];
                        if (d > eps){ DfSv[k] += a; }
                        else if (d < -eps){ DfSv[k] -= a; }
                    }
                }
            }
        }
        if (verbose){ printf("done.\n"); FLUSH; }

        #if 0
        /**  cut: direction at vertex v is -Uiv + Ujv, where Uk is the vector 
         **  of all zeros except 1 at coordinate k, iv is the steepest nonzero
         **  ascent label and j is the steepest nonunity descent label at
         **  vertex v  **/
        if (verbose){ printf("\tConstruct graph for min-cut/max-flow... "); FLUSH; }
        /* find labels i, j and set the source/sink capacities Uj - Ui */
        #if EXACT_DERIV
            Div = (int*) malloc(V*sizeof(int));
            Djv = (int*) malloc(V*sizeof(int));
        #endif
        #pragma omp parallel for private(v, rPv, DfSv, a, i, j, k) \
            schedule(static) num_threads(ntVK)
        for (v = 0; v < V; v++){
                rPv = (*rP) + Cv[v]*K;
                DfSv = DfS + v*K;
                /* find steepest nonzero ascent label */
                i = 0;
                while (rPv[i] <= eps){ i++; }
                a = DfSv[i];
                for (k = 0; k < K; k++){
                    if (rPv[k] > eps && DfSv[k] > a){ a = DfSv[i = k]; }
                }
                /* find steepest descent nonunity label */
                j = 0;
                while (rPv[j] >= ONE - eps){ j++; }
                a = DfSv[j];
                for (k = 0; k < K; k++){
                    if (rPv[k] < ONE - eps && DfSv[k] < a){ a = DfSv[j = k]; }
                }
                #if EXACT_DERIV
                    Div[v] = i;
                    Djv[v] = j;
                #endif
                G->nodes[v].tr_cap = DfSv[j] - DfSv[i];
        }

        /* set the d1 edge capacities */
        #if EXACT_DERIV
            /* horizontal and source/sink capacities are modified according to
             * Kolmogorov & Zabih (2004) ; E(u,v) is decomposed as
             * E(0,0) | E(0,1)   A|B          0 | 0    0|D-C   0|E
             * --------------- = --- =  A  + ------- + ----- + --- , E = B+C-A-D
             * E(1,0) | E(1,1)   C|D         C-A|C-A   0|D-C   0|0
             *
             *                         cst +   unary terms   + binary term     */
            /* #pragma omp parallel for private() num_threads(ntE)
             * this task cannot be easily parallelized */
            for (e = 0; e < E; e++){
                n = 2*e; /* index in directed edge list */
                u = Eu[e];
                v = Ev[e];
                /* E(0,0) is 0, a is E(0,1), b is E(1,0), c is E(1,1) */
                if (G->arcs[n].is_active){ /* must check for Pui = Pvi */
                    a = b = c = ZERO; 
                    ru = Cv[u];
                    rv = Cv[v]; 
                    rPu = (*rP) + ru*K;
                    rPv = (*rP) + rv*K;
                    /**  contribution of -Ui  **/
                    i = Div[u]; /* iu */
                    j = Div[v]; /* iv */
                    /* (Uiu)_iu versus (Uiv)_iu */
                    d = rPu[i] - rPv[i];
                    if (-eps <= d && d <= eps){
                        b += La_d1[e];
                        if (i != j){ c += La_d1[e]; }
                    }
                    /* (Uiu)_iv versus (Uiv)_iv */
                    d = rPu[j] - rPv[j];
                    if (-eps <= d && d <= eps){
                        a += La_d1[e];
                        if (i != j){ c += La_d1[e]; }
                    }
                    /**  contribution of +Uj  **/
                    i = Djv[u]; /* ju */
                    j = Djv[v]; /* jv */
                    /* (Uju)_ju versus (Ujv)_ju */
                    d = rPu[i] - rPv[i];
                    if (-eps <= d && d <= eps){
                        b += La_d1[e];
                        if (i != j){ c += La_d1[e]; }
                    }
                    /* (Uju)_jv versus (Ujv)_jv */
                    d = rPu[j] - rPv[j];
                    if (-eps <= d && d <= eps){
                        a += La_d1[e];
                        if (i != j){ c += La_d1[e]; }
                    }
                }else{ /* not active: this ensures Pu = Pv */
                    a = b = TWO*La_d1[e];
                    c = ZERO;
                    if (Div[u] != Div[v]){ c += TWO*La_d1[e]; }
                    if (Djv[u] != Djv[v]){ c += TWO*La_d1[e]; }
                }
                /* arbitrarily chosen orientation: u -> v */
                G->nodes[u].tr_cap += b; /* E(1,0)-E(0,0) = b */
                G->nodes[v].tr_cap += c - b; /* E(1,1)-E(1,0) = c-b */
                G->arcs[n].r_cap = a + b - c; /* E(1,0)+E(0,1)-E(1,1)-E(0,0) = b+a-c */
                G->arcs[n+1].r_cap = ZERO;
            }
        #else
            #pragma omp parallel for private(e, n) schedule(static) \
                num_threads(ntE)
            for (e = 0; e < E; e++){
                n = 2*e; /* index in directed edge list */
                if (G->arcs[n].is_active){
                    G->arcs[n].r_cap = G->arcs[n+1].r_cap = ZERO;
                }else{
                    G->arcs[n].r_cap = G->arcs[n+1].r_cap = TWO*La_d1[e];
                }
            }
        #endif
        if (verbose){ printf("done.\n"); FLUSH; }

        /* find min cut and activate corresponding edges */
        if (verbose){ printf("\tFind min cut and activate corresponding edges... "); FLUSH; }
        G->maxflow();
        s = 0; /* edge activation counter */
        #pragma omp parallel for private(e, u, v, i) reduction(+:s) \
            schedule(static) num_threads(ntE)
        for (e = 0; e < E; e++){
            i = 2*e; /* index in directed edge list */
            if (!G->arcs[i].is_active){
                u = Eu[e];
                v = Ev[e];
                #if EXACT_DERIV
                    if ((G->what_segment(u) != G->what_segment(v)) ||
                        ((!G->nodes[v].parent || G->nodes[v].is_sink) &&
                         /* both +1; their descent directions might disagree */
                         (Div[u] != Div[v] || Djv[u] != Djv[v])))
                #else
                    if (G->what_segment(u) != G->what_segment(v))
                #endif
                {
                    G->arcs[i].is_active = G->arcs[i+1].is_active = TRUE;
                    s++;
                }
            }
        }
        if (verbose){ printf("done.\n"); FLUSH; }
        #endif

        rDi = (int*) malloc(rV_*sizeof(int));
        Djv = (int*) calloc(V, sizeof(int));

        /* find most confident label for each connected component */
        #pragma omp parallel for private(v, rPv, a, i, k) schedule(static) \
            num_threads(ntVK)
        for (rv = 0; rv < rV_; rv++){
            rPv = (*rP) + rv*K;
            i = 0;
            a = rPv[0];
            for (k = 1; k < K; k++){
                if (rPv[k] > a){ a = rPv[i = k]; }
            }
            rDi[rv] = i;
        }
    
        /* iterate over all alternative labels */
        for (n = 1; n < K; n++){
            if (verbose){ printf("\talpha-expansion #%d/#%d: ", n, (K - 1)); FLUSH; }
            if (verbose){ printf("set capacities... "); FLUSH; }
            /* set the source/sink capacities */
            #pragma omp parallel for private(rv, DfSv, i, j, k, v, s, t) \
                schedule(static) num_threads(ntVK)
            for (rv = 0; rv < rV_; rv++){
                i = rDi[rv];
                j = n > i ? n : (n - 1);
                /* run along the component ru */
                for (v = Vc[s = rVc[rv]], t = rVc[rv+1]; s < t; v = Vc[++s]){
                    DfSv = DfS + v*K;
                    k = Djv[v];
                    /* cost for changing dv to Uj - Ui */
                    if (k == 0){ G->nodes[v].tr_cap = DfSv[j] - DfSv[i]; }
                    else if (k == n){ G->nodes[v].tr_cap = ZERO; }
                    else if (k > i){ G->nodes[v].tr_cap = DfSv[j] - DfSv[k]; }
                    else{ G->nodes[v].tr_cap = DfSv[j] - DfSv[k-1]; }
                }
            }
            /* set the d1 edge capacities */
            /* #pragma omp parallel for private(e, i) schedule(static) \
             *     num_threads(ntE)
             * this task cannot be easily parallelized */
            for (e = 0; e < E; e++){
                i = 2*e; /* index in directed edge list */
                if (G->arcs[i].is_active){ /* ignore weird equality cases */
                    G->arcs[i].r_cap = G->arcs[i+1].r_cap = ZERO;
                }else{
                /* horizontal and source/sink capacities are modified according to
                 * Kolmogorov & Zabih (2004); E(u,v) is decomposed as
                 * E(0,0) | E(0,1)   A|B          0 | 0    0|D-C   0|E
                 * --------------- = --- =  A  + ------- + ----- + --- , E = B+C-A-D
                 * E(1,0) | E(1,1)   C|D         C-A|C-A   0|D-C   0|0
                 *
                 *                         cst +   unary terms   + binary term     */
                    u = Eu[e];
                    v = Ev[e];
                    j = Djv[u];
                    k = Djv[v];
                    /* a is E(0,0), b is E(0,1), c is E(1,0), E(1,1) is 0 */
                    a = (j == k) ? ZERO : TWO*La_d1[e];
                    /* E(0,1) is for changing dv to Uj - Ui while keeping du identical */
                    /* if (j == n){ b = ZERO; } */
                        /* impossible with only one alpha-expansion cycle */
                    /* else{ */ b = TWO*La_d1[e]; /* } */
                    /* E(1,0) is for changing du to Uj - Ui while keeping dv identical */
                    /* if (k == n){ b = ZERO; } */
                        /* impossible with only one alpha-expansion cycle */
                    /* else{ */ c = TWO*La_d1[e]; /* } */
                    /* arbitrarily chosen orientation: u -> v */
                    G->nodes[u].tr_cap += c - a; /* E(1,0)-E(0,0) = c-a */
                    G->nodes[v].tr_cap -= c; /* E(1,1)-E(1,0) = -c */
                    G->arcs[i].r_cap = b + c - a; /* E(1,0)+E(0,1)-E(1,1)-E(0,0) = b+c-a */
                    G->arcs[i+1].r_cap = ZERO;
                }
            }
            if (verbose){ printf("done; "); FLUSH; }
            /* find min cut and update descent label accordingly */
            if (verbose){ printf("find min-cut/max-flow... "); FLUSH; }
            G->maxflow();
            #pragma omp parallel for private(v) schedule(static) \
                num_threads(ntVK)
            for (v = 0; v < V; v++){
                if (!G->nodes[v].parent || G->nodes[v].is_sink){ Djv[v] = n; }
            }
            if (verbose){ printf("done.\n"); FLUSH; }
        }

        /* activate edges correspondingly */
        s = 0; /* edge activation counter */
        #pragma omp parallel for private(e, i) reduction(+:s) \
            schedule(static) num_threads(ntE)
        for (e = 0; e < E; e++){
            i = 2*e; /* index in directed edge list */
            if (!G->arcs[i].is_active && (Djv[Eu[e]] != Djv[Ev[e]])){
                G->arcs[i].is_active = G->arcs[i+1].is_active = TRUE;
                s++;
            }
        }

        /* free steepest cut stuff */
        free(DfS);
        /* #if EXACT_DERIV
            free(Div);
            free(Djv);
        #endif */
        free(rDi);
        free(Djv);
        if (verbose){ printf("\t%d new activated edge(s).\n", s); FLUSH; }

        /***  check for no activation  ***/
        if (s == 0){ /**  recomputing everything is not worth  **/
            if (CP_difTol > ZERO || Dif != NULL){
                dif = ZERO;
                if (Dif != NULL){ Dif[CP_it_] = dif; } 
            }
            CP_it_++;
            if (Obj != NULL){ Obj[CP_it_] = Obj[CP_it_-1]; }
            continue;
        }else{ /* reduced values will be recomputed */
            FREE(*rP);
        }

        /***  compute reduced graph  ***/
        if (verbose){ printf("\tConstruct reduced problem... "); FLUSH; }
        /**  compute connected components  **/
        /* cleanup assigned components */
        for (v = 0; v < V; v++){ Cv[v] = -1; }
        rV_ = 0; /* current connected component */
        n = 0; /* number of vertices already assigned */
        i = 0; /* index of vertices currently exploring */
        rVc = (int*) realloc(rVc, (V + 1)*sizeof(int));
        rVc[0] = 0;
        /* depth first search */
        for (u = 0; u < V; u++){
            if (Cv[u] != -1){ continue; } /* already assigned */
            Cv[u] = rV_; /* assign to current component */
            Vc[n++] = u; /* put in connected components list */
            while (i < n){
                v = Vc[i++]; 
                /* add neighbors to the connected components list */
                for (ee = G->nodes[v].first; ee; ee = ee->next){
                    if (!ee->is_active){
                        w = (int) (ee->head - G->nodes);
                        if (Cv[w] != -1){ continue; }  /* already assigned */
                        Cv[w] = rV_; /* assign to current component */
                        Vc[n++] = w; /* put in connected components list */
                    }
                }
            } /* the current connected component is complete */
            rVc[++rV_] = n;
        }
        /* update cumulative components size and number of parallel threads */
        rVc = (int*) realloc(rVc, (rV_ + 1)*sizeof(int));
        ntrV = compute_num_threads(rV_, rV_);
        ntrVK = compute_num_threads(rV_*K, rV_);
        ntVK_rV = compute_num_threads(V*K, rV_);
        
        /**  compute reduced connectivity and penalizations  **/
        /* rEc has two purposes
         * 1) keep track of the number of edges going out of a component
         * 2) maps neighboring components to reduced edges
         * note that both are simultaneously possible because we consider
         * undirected edges, indexed by (ru, rv) such that ru < rv */
        rEc = (int*) malloc(rV_*sizeof(int));
        rEv = (int*) malloc(E*sizeof(int));
        rLa_d1 = (real*) malloc(E*sizeof(real));
        for (rv = 0; rv < rV_; rv++){ rEc[rv] = -1; }
        rE = 0; /* current number of reduced edges */
        n = 0; /* keep track of previous edge number */
        /* iterate over components */
        for (ru = 0; ru < rV_; ru++){
            i = 1; /* flag signalling isolated components */
            /* run along the component ru */
            for (u = Vc[s = rVc[ru]], t = rVc[ru+1]; s < t; u = Vc[++s]){
                for (ee = G->nodes[u].first; ee; ee = ee->next){
                    if (!ee->is_active){ continue; }
                    e = ((int) (ee - G->arcs))/2;
                    a = La_d1[e];
                    if (a == ZERO){ continue; }
                    i = 0; /* a nonzero edge involving ru exists */
                    v = (int) (ee->head - G->nodes);
                    rv = Cv[v];
                    if (rv < ru){ continue; } /* count only undirected edges */
                    re = rEc[rv];
                    if (re == -1){ /* new edge */
                        rEv[rE] = rv;
                        rLa_d1[rE] = a;
                        rEc[rv] = rE++;
                    }else{ /* edge already exists */
                        rLa_d1[re] += a;
                    }
                }
            }
            if (i){ /* isolated components should be treated separately!
                     * for now; we just link them to themselves to ensure that
                     * they will be taken into account by PFDR */
                rEv[rE] = ru;
                rLa_d1[rE++] = eps;
            }else{
                for (; n < rE; n++){ rEc[rEv[n]] = -1; } /* reset rEc */
            }
            rEc[ru] = rE;
        }
        /* update reduced edge list and number of parallel threads */
        rEv = (int*) realloc(rEv, rE*sizeof(int));
        rEu = (int*) malloc(rE*sizeof(int));
        rLa_d1 = (real*) realloc(rLa_d1, rE*sizeof(real));
        re = 0;
        for (ru = 0; ru < rV_; ru++){ while(re < rEc[ru]){ rEu[re++] = ru; } }
        free(rEc);
        ntrEK = compute_num_threads(rE*K, rE);

        /**  compute reduced observations  **/
        *rP = (real*) CALLOC(rV_*K, sizeof(real));
        rQ = (real*) malloc(rV_*K*sizeof(real));
        if (al > ZERO){ rLa_f = (real*) malloc(rV_*sizeof(real)); }
        #pragma omp parallel for private(rv, s, t, v, Qv, rPv, i, k, a) \
            schedule(static) num_threads(ntVK_rV)
        for (rv = 0; rv < rV_; rv++){
            rPv = (*rP) + rv*K; /* sum along the component rv */
            for (v = Vc[s = rVc[rv]], t = rVc[rv+1]; s < t; v = Vc[++s]){
                Qv = Q + v*K;
                for (k = 0; k < K; k++){ rPv[k] += Qv[k]; }
            }
            /* put reduced observation in rQ and unisimplicial optimum in rP */
            rQv = rQ + rv*K;
            if (al == ZERO){ /* linear loss
                * unisimplicial optimum at simplex corner 
                * reduced obervation is the sum */
                a = rPv[i = 0];
                for (k = 1; k < K; k++){ if (rPv[k] > a){ a = rPv[i = k]; } }
                for (k = 0; k < K; k++){
                    rQv[k] = rPv[k];
                    rPv[k] = (k == i) ? ONE : ZERO;
                }
            }else{ /* quadratic or KLa loss
                * unisimplicial optimum at barycenter 
                * 'equivalent' reduced observation is barycenter with weights */
                i = rVc[rv+1] - rVc[rv];
                for (k = 0; k < K; k++){
                    rQv[k] = rPv[k]/i;
                    rPv[k] = rQv[k];
                }
                rLa_f[rv] = i;
            }
        }

        if (verbose){
            printf("%d connected component(s), %d reduced edge(s).\n", rV_, rE);
            FLUSH;
        }

        /***  preconditioned forward-Douglas-Rachford  ***/
        if (verbose){
            printf("\tSolve reduced problem:\n");
            FLUSH;
        }
        PFDR_graph_loss_d1_simplex<real>(K, rV_, rE, al, rLa_f, *rP, rQ, \
            rEu, rEv, rLa_d1, PFDR_rho, PFDR_condMin, PFDR_difRcd, \
            PFDR_difTol, PFDR_itMax, &PFDR_it, NULL, NULL, verbose);

        /***  merge neighboring components with almost equal values  ***/
        /* this only deactivates edges, components are not updated yet */
        s = 0; /* edge deactivation counter */
        #pragma omp parallel for private(e, i, a, rPu, rPv, k, d) \
            reduction(+:s) schedule(static) num_threads(ntE)
        for (e = 0; e < E; e++){
            i = 2*e; /* index in directed edge list */
            if (G->arcs[i].is_active){
                a = ZERO; /* store max difference */
                rPu = (*rP) + Cv[Eu[e]]*K;
                rPv = (*rP) + Cv[Ev[e]]*K;
                for (k = 0; k < K; k++){
                    d = rPu[k] - rPv[k];
                    if (d < ZERO){ d = -d; }
                    if (d > a){ a = d; }
                }
                if (a <= eps){
                    G->arcs[i].is_active = G->arcs[i+1].is_active = FALSE;
                    s++;
                }
            }
        }
        if (verbose){ printf("\t%d deactivated edge(s).\n", s); FLUSH; }

        /***  progress  ***/
        /**  iterate evolution  **/
        if (CP_difTol > ZERO || Dif != NULL){
            dif = ZERO;
            if (CP_difTol >= ONE){
                #pragma omp parallel for private(rv, rPv, a, i, k, v, s, t) \
                    reduction(+:dif) schedule(static) num_threads(ntVK_rV)
                for (rv = 0; rv < rV_; rv++){
                    /* get maximum likelihood label */
                    rPv = (*rP) + rv*K;
                    a = rPv[0];
                    i = 0;
                    for (k = 1; k < K; k++){
                        if (rPv[k] > a){ a = rPv[i = k]; }
                    }
                    a = (real) i;
                    /* run along the component rv */
                    for (v = Vc[s = rVc[rv]], t = rVc[rv+1]; s < t; v = Vc[++s]){
                    /* compare with previous, and update component assignation */
                        if (a != rP_[Cv_[v]]){ dif += ONE; }
                        Cv_[v] = rv;
                    }
                }
                /* update maximum likelihood label */
                rP_ = (real*) realloc(rP_, rV_*sizeof(real));
                #pragma omp parallel for private(rv, rPv, a, i, k) \
                    schedule(static) num_threads(ntrVK)
                for (rv = 0; rv < rV_; rv++){
                    /* recompute maximum likelihood label */
                    rPv = (*rP) + rv*K;
                    a = rPv[0];
                    i = 0;
                    for (k = 1; k < K; k++){
                        if (rPv[k] > a){ a = rPv[i = k]; }
                    }
                    rP_[rv] = (real) i;
                }
            }else{
                /* max reduction available in C since OpenMP 3.1 and gcc 4.7 */
                #pragma omp parallel for private(v, rv, rPu, rPv, d, k) \
                    /* reduction(max:dif) */ reduction(+:dif) \
                    schedule(static) num_threads(ntVK)
                for (v = 0; v < V; v++){
                    rv = Cv[v];
                    rPu = (*rP) + rv*K;
                    rPv = rP_ + Cv_[v]*K;
                    a = ZERO;
                    for (k = 0; k < K; k++){
                        d = rPu[k] - rPv[k];
                        if (d < ZERO){ d = -d; }
                        /* if (d > dif){ dif = d; } */ /* max norm */
                        dif += d; /* relative l1 norm evolution */
                    }
                    Cv_[v] = rv;
                }
                dif /= V; /* relative l1 norm evolution */
                rP_ = (real*) realloc(rP_, rV_*K*sizeof(real));
                for (rv = 0; rv < rV_*K; rv++){ rP_[rv] = (*rP)[rv]; }
            }
            if (Dif != NULL){ Dif[CP_it_] = dif; } 
        }
        CP_it_++;
    
        /**  objective functional value  **/
        if (Obj != NULL){
            a = ZERO;
            if (al == ZERO){ /* linear loss */
                #pragma omp parallel for private(rv) reduction(+:a) \
                    schedule(static) num_threads(ntrVK)
                for (rv = 0; rv < rV_*K; rv++){ a -= (*rP)[rv]*rQ[rv]; }
            }else if (al == ONE){ /* quadratic loss */
                #pragma omp parallel for private(v, Qv, rPv, k, c) \
                    reduction(+:a) schedule(static) num_threads(ntVK)
                for (v = 0; v < V; v++){
                    rPv = (*rP) + Cv[v]*K;
                    Qv = Q + v*K;
                    for (k = 0; k < K; k++){
                        c = rPv[k] - Qv[k];
                        a += c*c;
                    }
                }
                a *= HALF;
            }else{ /* KLa loss */
                #pragma omp parallel for private(v, Qv, rPv, k, c) \
                    reduction(+:a) schedule(static) num_threads(ntVK)
                for (v = 0; v < V; v++){
                    rPv = (*rP) + Cv[v]*K;
                    Qv = Q + v*K;
                    for (k = 0; k < K; k++){
                        c = al_K + al_1*Qv[k];
                        a += c*log(c/(al_K + al_1*rPv[k]));
                    }
                }
            }
            Obj[CP_it_] = a;
            /* ||x||_{d1,La_d1} */
            a = ZERO;
            #pragma omp parallel for private(re, rPu, rPv, k, b, c) \
                reduction(+:a) schedule(static) num_threads(ntrEK)
            for (re = 0; re < rE; re++){
                rPu = (*rP) + rEu[re]*K;
                rPv = (*rP) + rEv[re]*K;
                b = ZERO;
                for (k = 0; k < K; k++){
                    c = rPu[k] - rPv[k];
                    if (c < ZERO){ c = -c; }
                    b += c;
                }
                a += rLa_d1[re]*b;
            }
            Obj[CP_it_] += a;
        }

        /***  free reduced graph stuff  ***/
        free(rEu);
        free(rEv);
        free(rLa_d1);
        free(rLa_f);
        free(rQ);

    } /* endwhile (true) */

    /* final information */
    *rV = rV_;
    *CP_it = CP_it_;

    /* free CP stuff, or store for warm restart */
    if (CP_restart != NULL){
        CP_restart->G = G;
        CP_restart->Vc = Vc;
        CP_restart->rVc = rVc;
    }else{
        delete G;
        free(Vc);
        free(rVc);
    }

    /* free remaining stuff */
    free(rP_);
    free(Cv_);
}

/* instantiate for compilation */
template void CP_PFDR_graph_loss_d1_simplex<float>(const int, const int, const int, \
    const float, int*, int*, float**, const float*, const int*, const int*, \
    const float*, const float, const int, int*, const float, \
    const float, const float, const float, const int, double*, float*, float*, \
    const int, struct CPls_Restart<float> *CP_restart);

template void CP_PFDR_graph_loss_d1_simplex<double>(const int, const int, const int, \
    const double, int*, int*, double**, const double*,const int*, const int*, \
    const double*, const double, const int, int*, const double, \
    const double, const double, const double, const int, double*, double*, double*, \
    const int, struct CPls_Restart<double> *CP_restart);
