/*
 * Python wrapper for Cut Pursuit for solving inverse problems regularized with
 * graph total variation and l1 penalty and positivity constraint
 * Loic Landrieu 2018
 *
 * minimize the following functional over a graph G = (V, E)
 *
 * F(x) = 1/2 ||A x - obs||^2
 *      + sum_{(u,v) in E)}{edge_weight(u,v) * |x_u - x_v|}
 *      + sum_{v in V)}{l1_weight(v) * |x_v|}
 * with positivity constrainty of x
 *
 * SYNTAX
 * cV, rX = CP_quadratic_l1(obs, source, target, edge_weight, A, l1_weight)
 *
 * INPUTS
 * numpy array must share the same dtype and be either f4, f8, float32, float64
 * obs = observation vector
 *      vector, numpy array of shape (N,)
 * source = source vertices of edges E
 *      vector, numpy array of shape (E,)
 * target = target vertices of edges E
 *      vector, numpy array of shape (E,)
 * edge_weight = weight associated with each edge in the TV regularization
 *      vector, numpy array of shape (E,)
 *      scalar, in which case all weights are the same
 * A = design matrix of the inverse problem
 *      matrix, numpy array of shape (N,V)
 *      vector, numpy array of shape (V,), in which case A is a diagonal matrix
 *      scalar, in which case A is the identity matrix (the value of the scalar is discarded)
 * l1_weight = weight associated with each vertex in the L1 regularization
 *      vector, numpy array of shape (V,)
 *      scalar, in which case all weights are the same
 * optional parameter (see below) = positivity, PFDR_rho, PFDR_condMin, CP_difTol,
 *          PFDR_difRcd, PFDR_difTol, CP_itMax, PFDR_itMax, verbose
 *
 * OUPUTS
 * cV = index of the component of each vertex
 *     numpy array of shape (V,)
 * rX = value associated to each constant component
 *      numpy array of shape (rV,) with rV the number of constant components
 * x = rX[cV]
 *
 * OPTIONS
 * Modify with caution, see the paper for details.
 * double positivity : if nonzero, add positivity constraints
 * int CP_itMax : maximum number of cut pursuit iterations (default : 10)
 * int PFDR_itMax : maximum number of PFDR iterations for the reduced problems (default : 1e4)
 * int verbose : if nonzero, display reduced problem progress information  (default : 0)
 * double CP_difTol : stopping criterion on iterate evolution (default : 1e-3)
 * double PFDR_difTol : stopping criterion on iterate evolution for the reduced problem (default : 1e-4)
 * double PFDR_rho : relaxation parameter for PFDR, 0 < rho < 2 (default : 1)
 * T PFDR_condMin : small positive parameter ensuring stability of preconditioning (default : 1e3)
 * T PFDR_difRcd : reconditioning criterion (default : 0, no reconditioning)
 *
 * see examples.py for examples of usage
*/

#include <iostream>
#include <stdexcept>
#include <typeinfo>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <numpy/ndarrayobject.h>
#include "boost/tuple/tuple.hpp"
#include <../include/CP_PFDR_graph_quadratic_d1_l1.hpp>

#define DEFAULT_positivity ((int) 0)
#define DEFAULT_PFDR_rho ((double) 1.0)
#define DEFAULT_PFDR_condMin ((double) 1e-3)
#define DEFAULT_CP_difTol ((double) 1e-3)
#define DEFAULT_PFDR_difRcd ((double) 0)
#define DEFAULT_PFDR_difTol ((double) 1e-4)
#define DEFAULT_CP_itMax ((int) 10)
#define DEFAULT_PFDR_itMax ((int) 1e4)
#define DEFAULT_verbose ((int) 0)

namespace bpn = boost::python::numpy;
namespace bp =  boost::python;

enum A_mode {identity, diagonal, matrix};

template<class T>
struct to_py_tuple
{//converts output to a python tuple
    static bp::tuple convert(const int V, const int * Cv, const int rV, const T * rX){

        npy_intp dims_V = V;
        PyObject * Cv_py = PyArray_SimpleNew(1, &dims_V, NPY_INT);
        void * Cv_data = PyArray_DATA((PyArrayObject*)Cv_py);
        memcpy(Cv_data, Cv, dims_V * sizeof(int));

        npy_intp dims_rV = rV;
        PyObject * rX_py;
        if (typeid(T) == typeid(float))
            {rX_py= PyArray_SimpleNew(1, &dims_rV, NPY_FLOAT32);}
        else if (typeid(T) == typeid(double))
            {rX_py= PyArray_SimpleNew(1, &dims_rV, NPY_FLOAT64);}
        else
        {throw std::invalid_argument("Type unknown");}
        void * rX_data = PyArray_DATA((PyArrayObject*)rX_py);
        memcpy(rX_data, rX, dims_rV * sizeof(T));

        return boost::python::make_tuple(bp::handle<>(bp::borrowed(Cv_py))
                                       , bp::handle<>(bp::borrowed(rX_py)));
    }
};

template<class T>
bp::tuple CP_quadratic_l1_from_array(int N, int V, int E, int edge_weight_length, int l1_weight_length, A_mode a_mode
                       , T * obs_data, T * A_data , const int * source_data, const int * target_data
                       , T * edge_weight_data, T * l1_weight_data, const int positivity
                       , T PFDR_rho, T PFDR_condMin, T CP_difTol, T PFDR_difRcd, T PFDR_difTol
                       , int CP_itMax, int PFDR_itMax, int verbose)
{   //call the c+ function with the proper type
    T * A_data_ = nullptr, * obs_data_ = nullptr;
    //--- adapt variable inputs ---
    if (a_mode == diagonal)
    {   //need to multiply A and obs by A - without effecting the inputs!
        A_data_ = (T* )malloc(V*sizeof(T));
        obs_data_ = (T* )malloc(V*sizeof(T));
        std::memcpy(obs_data_, obs_data, V * sizeof(T));
        std::memcpy(A_data_, A_data, V * sizeof(T));
        for (int i = 0; i < V; i++)
        {
            obs_data_[i] = obs_data[i] * A_data[i];
            A_data_[i] = A_data[i] * A_data[i];
        }
    }
    if (edge_weight_length == 1)
    {   //create an uniform vector
        float scalar = edge_weight_data[0];
        edge_weight_data = (T* )malloc(E*sizeof(T));
        for (int i = 0; i < V; i++) {edge_weight_data[i] = scalar;}
    }
    if (l1_weight_length == 1)
    {   //create an uniform vector
        float scalar = l1_weight_data[0];
        l1_weight_data = (T* )malloc(V*sizeof(T));
        for (int i = 0; i < V; i++) {l1_weight_data[i] = scalar;}
    }
    //--- outputs ---
    T * rX;
    int Cv[V];
    int rV;
    int CP_it[1];
    double * Time = NULL;
    T * Obj = NULL;
    T * Dif = NULL;
    //---call to c++ function---
    if (a_mode == identity)
    {
        CP_PFDR_graph_quadratic_d1_l1<T>(V, E, N, &rV, Cv, &rX, obs_data, NULL, \
            source_data, target_data, edge_weight_data, l1_weight_data, positivity, CP_difTol, CP_itMax, CP_it, PFDR_rho, \
            PFDR_condMin, PFDR_difRcd, PFDR_difTol, PFDR_itMax, Time, Obj, \
            Dif, verbose, NULL);
    }else if (a_mode == diagonal)
    {
            CP_PFDR_graph_quadratic_d1_l1<T>(V, E, N, &rV, Cv, &rX, obs_data_, A_data_, \
                source_data, target_data, edge_weight_data, l1_weight_data, positivity, CP_difTol, CP_itMax, CP_it, PFDR_rho, \
                PFDR_condMin, PFDR_difRcd, PFDR_difTol, PFDR_itMax, Time, Obj, \
                Dif, verbose, NULL);
    }else
    {
        CP_PFDR_graph_quadratic_d1_l1<T>(V, E, N, &rV, Cv, &rX, obs_data, A_data, \
            source_data, target_data, edge_weight_data, l1_weight_data, positivity, CP_difTol, CP_itMax, CP_it, PFDR_rho, \
            PFDR_condMin, PFDR_difRcd, PFDR_difTol, PFDR_itMax, Time, Obj, \
            Dif, verbose, NULL);
    }
    //--- freeing memory ---
    if (a_mode == diagonal)
    {
        free(A_data_);
        free(obs_data_);
    }
    if (edge_weight_length == 1)
    {
        free(edge_weight_data);
    }
    if (l1_weight_length == 1)
    {
        free(l1_weight_data);
    }
    //--- converting ---
    return to_py_tuple<T>::convert(V, Cv, rV, rX);
}

bp::tuple CP_quadratic_l1(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      , const bpn::ndarray & edge_weight, const bpn::ndarray & A
                      , const bpn::ndarray & l1_weight, const int positivity
                      , double PFDR_rho, double PFDR_condMin, double CP_difTol, double PFDR_difRcd, double PFDR_difTol
                      , int CP_itMax, int PFDR_itMax, int verbose)
{
    //---read and check size ---
    int N = bp::len(obs);
    const int E = bp::len(source);
    int V;
    A_mode a_mode;
    const int dim_A = A.get_nd();
    int n_lin_A = A.shape(0);
    int n_col_A = 1;

    if (dim_A==2) {n_col_A = A.shape(1);};

    if (n_col_A==1)
    {
        if (n_lin_A == 1)
        {
            a_mode = identity;
        }
        else if (n_lin_A == N) {
            a_mode = diagonal;
        }
        else {throw std::invalid_argument("A should be either a scalar, vector of size N, or a N-by-V matrix ");}
        V = N;
        N = 0;
    } else if (n_col_A>1)
    {
        if (n_lin_A != N) {throw std::invalid_argument("A should be either a scalar, a vector of size N, or a N-by-V matrix ");}
        a_mode = matrix;
        V = n_col_A;
    }
    else{
        throw std::invalid_argument("A should be either a scalar, a vector of size N, or a N-by-V matrix ");
    }
    //---detect type of inputs ---
    bpn::dtype data_type = obs.get_dtype();
    std::string data_classname = bp::extract<std::string>(bp::str(data_type));
    if (!data_classname.compare("float32") || !data_classname.compare("f4"))
    {
        return CP_quadratic_l1_from_array<float>(N, V, E, bp::len(edge_weight), bp::len(l1_weight), a_mode
                                                  , reinterpret_cast<float*>(obs.get_data())
                                                  , reinterpret_cast<float*>(A.get_data())
                                                  , reinterpret_cast<int*>(source.get_data())
                                                  , reinterpret_cast<int*>(target.get_data())
                                                  , reinterpret_cast<float*>(edge_weight.get_data())
                                                  , reinterpret_cast<float*>(l1_weight.get_data())
                                                  , positivity
                                                  ,(float) PFDR_rho, (float) PFDR_condMin, (float) CP_difTol
                                                  ,(float) PFDR_difRcd, (float) PFDR_difTol, CP_itMax, PFDR_itMax, verbose);
    }
    else if (!data_classname.compare("float64") || !data_classname.compare("f8"))
    {
        return CP_quadratic_l1_from_array<double>(N, V, E, bp::len(edge_weight), bp::len(l1_weight), a_mode
                                                  , reinterpret_cast<double*>(obs.get_data())
                                                  , reinterpret_cast<double*>(A.get_data())
                                                  , reinterpret_cast<int*>(source.get_data())
                                                  , reinterpret_cast<int*>(target.get_data())
                                                  , reinterpret_cast<double*>(edge_weight.get_data())
                                                  , reinterpret_cast<double*>(l1_weight.get_data())
                                                  , positivity
                                                  ,(double) PFDR_rho, (double) PFDR_condMin, (double) CP_difTol
                                                  ,(double) PFDR_difRcd, (double) PFDR_difTol, CP_itMax, PFDR_itMax, verbose);
    }
    else
    {
       {throw std::invalid_argument("Type unknown, must be float32, float64, f4, or f8.");}
    }
}

bp::tuple CP_quadratic_l1_1(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      , double edge_weight, const bpn::ndarray & A
                      , const bpn::ndarray & l1_weight, const int positivity
                      , double PFDR_rho, double PFDR_condMin, double CP_difTol, double PFDR_difRcd, double PFDR_difTol
                      , int CP_itMax, int PFDR_itMax, int verbose)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray edge_weight_ = bpn::zeros(shape, data_type);
    edge_weight_[0] = edge_weight;
    return CP_quadratic_l1(obs, source, target, edge_weight_, A, l1_weight, positivity
                          ,PFDR_rho, PFDR_condMin, CP_difTol, PFDR_difRcd, PFDR_difTol, CP_itMax, PFDR_itMax, verbose);
}

bp::tuple CP_quadratic_l1_2(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      ,  const bpn::ndarray & edge_weight, double A
                      , const bpn::ndarray & l1_weight, const int positivity
                      , double PFDR_rho, double PFDR_condMin, double CP_difTol, double PFDR_difRcd, double PFDR_difTol
                      , int CP_itMax, int PFDR_itMax, int verbose)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray A_ = bpn::zeros(shape, data_type);
    A_[0] = 0; //value of scalar discarded
    return CP_quadratic_l1(obs, source, target, edge_weight, A_, l1_weight, positivity
                         ,PFDR_rho, PFDR_condMin, CP_difTol, PFDR_difRcd, PFDR_difTol, CP_itMax, PFDR_itMax, verbose);
}

bp::tuple CP_quadratic_l1_3(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      ,  const bpn::ndarray & edge_weight, const bpn::ndarray & A
                      , double l1_weight, const int positivity
                      , double PFDR_rho, double PFDR_condMin, double CP_difTol, double PFDR_difRcd, double PFDR_difTol
                      , int CP_itMax, int PFDR_itMax, int verbose)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray l1_weight_ = bpn::zeros(shape, data_type);
    l1_weight_[0] = l1_weight;
    return CP_quadratic_l1(obs, source, target, edge_weight, A, l1_weight_, positivity
                         ,PFDR_rho, PFDR_condMin, CP_difTol, PFDR_difRcd, PFDR_difTol, CP_itMax, PFDR_itMax, verbose);
}

bp::tuple CP_quadratic_l1_4(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      , double edge_weight, double A
                      , const bpn::ndarray & l1_weight, const int positivity
                      , double PFDR_rho, double PFDR_condMin, double CP_difTol, double PFDR_difRcd, double PFDR_difTol
                      , int CP_itMax, int PFDR_itMax, int verbose)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray edge_weight_ = bpn::zeros(shape, data_type);
    bpn::ndarray A_ = bpn::zeros(shape, data_type);
    edge_weight_[0] = edge_weight;
    A_[0] = 0; //value of scalar discarded
    return CP_quadratic_l1(obs, source, target, edge_weight_, A_, l1_weight, positivity
                         ,PFDR_rho, PFDR_condMin, CP_difTol, PFDR_difRcd, PFDR_difTol, CP_itMax, PFDR_itMax, verbose);
}

bp::tuple CP_quadratic_l1_5(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      ,  double edge_weight, const bpn::ndarray & A
                      , double l1_weight, const int positivity
                      , double PFDR_rho, double PFDR_condMin, double CP_difTol, double PFDR_difRcd, double PFDR_difTol
                      , int CP_itMax, int PFDR_itMax, int verbose)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray edge_weight_ = bpn::zeros(shape, data_type);
    bpn::ndarray l1_weight_ = bpn::zeros(shape, data_type);
    edge_weight_[0] = edge_weight;
    l1_weight_[0] = l1_weight;
    return CP_quadratic_l1(obs, source, target, edge_weight_, A, l1_weight_, positivity
                         ,PFDR_rho, PFDR_condMin, CP_difTol, PFDR_difRcd, PFDR_difTol, CP_itMax, PFDR_itMax, verbose);
}

bp::tuple CP_quadratic_l1_6(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      ,  const bpn::ndarray & edge_weight, double A
                      , double l1_weight, const int positivity
                      , double PFDR_rho, double PFDR_condMin, double CP_difTol, double PFDR_difRcd, double PFDR_difTol
                      , int CP_itMax, int PFDR_itMax, int verbose)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray l1_weight_ = bpn::zeros(shape, data_type);
    bpn::ndarray A_ = bpn::zeros(shape, data_type);
    l1_weight_[0] = l1_weight;
    A_[0] = 0; //value of scalar discarded
    return CP_quadratic_l1(obs, source, target, edge_weight, A_, l1_weight_, positivity
                         ,PFDR_rho, PFDR_condMin, CP_difTol, PFDR_difRcd, PFDR_difTol, CP_itMax, PFDR_itMax, verbose);
}

bp::tuple CP_quadratic_l1_7(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      ,  double edge_weight, double A, double l1_weight
                      , const int positivity, double PFDR_rho, double PFDR_condMin, double CP_difTol, double PFDR_difRcd, double PFDR_difTol
                      , int CP_itMax, int PFDR_itMax, int verbose)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray edge_weight_ = bpn::zeros(shape, data_type);
    bpn::ndarray l1_weight_ = bpn::zeros(shape, data_type);
    bpn::ndarray A_ = bpn::zeros(shape, data_type);
    edge_weight_[0] = edge_weight;
    l1_weight_[0] = l1_weight;
    A_[0] = 0; //value of scalar discarded
    return CP_quadratic_l1(obs, source, target, edge_weight_, A_, l1_weight_, positivity
                         , PFDR_rho, PFDR_condMin, CP_difTol, PFDR_difRcd, PFDR_difTol, CP_itMax, PFDR_itMax, verbose);
}

BOOST_PYTHON_MODULE(libCP)
{
    _import_array();
    bpn::initialize();
    bp::def("CP_quadratic_l1", CP_quadratic_l1
            , (bp::args("positivity")=DEFAULT_positivity, bp::args("PFDR_rho")=DEFAULT_PFDR_rho
            , bp::args("PFDR_condMin")=DEFAULT_PFDR_condMin, bp::args("CP_difTol")=DEFAULT_CP_difTol
            , bp::args("PFDR_difRcd")=DEFAULT_PFDR_difRcd, bp::args("PFDR_difTol")=DEFAULT_PFDR_difTol
            , bp::args("CP_itMax")=DEFAULT_CP_itMax, bp::args("PFDR_itMax")=DEFAULT_PFDR_itMax
            , bp::args("verbose")=DEFAULT_verbose));
    bp::def("CP_quadratic_l1", CP_quadratic_l1_1
            , (bp::args("positivity")=DEFAULT_positivity, bp::args("PFDR_rho")=DEFAULT_PFDR_rho
            , bp::args("PFDR_condMin")=DEFAULT_PFDR_condMin, bp::args("CP_difTol")=DEFAULT_CP_difTol
            , bp::args("PFDR_difRcd")=DEFAULT_PFDR_difRcd, bp::args("PFDR_difTol")=DEFAULT_PFDR_difTol
            , bp::args("CP_itMax")=DEFAULT_CP_itMax, bp::args("PFDR_itMax")=DEFAULT_PFDR_itMax
            , bp::args("verbose")=DEFAULT_verbose));
    bp::def("CP_quadratic_l1", CP_quadratic_l1_2
            , (bp::args("positivity")=DEFAULT_positivity, bp::args("PFDR_rho")=DEFAULT_PFDR_rho
            , bp::args("PFDR_condMin")=DEFAULT_PFDR_condMin, bp::args("CP_difTol")=DEFAULT_CP_difTol
            , bp::args("PFDR_difRcd")=DEFAULT_PFDR_difRcd, bp::args("PFDR_difTol")=DEFAULT_PFDR_difTol
            , bp::args("CP_itMax")=DEFAULT_CP_itMax, bp::args("PFDR_itMax")=DEFAULT_PFDR_itMax
            , bp::args("verbose")=DEFAULT_verbose));
    bp::def("CP_quadratic_l1", CP_quadratic_l1_3
            , (bp::args("positivity")=DEFAULT_positivity, bp::args("PFDR_rho")=DEFAULT_PFDR_rho
            , bp::args("PFDR_condMin")=DEFAULT_PFDR_condMin, bp::args("CP_difTol")=DEFAULT_CP_difTol
            , bp::args("PFDR_difRcd")=DEFAULT_PFDR_difRcd, bp::args("PFDR_difTol")=DEFAULT_PFDR_difTol
            , bp::args("CP_itMax")=DEFAULT_CP_itMax, bp::args("PFDR_itMax")=DEFAULT_PFDR_itMax
            , bp::args("verbose")=DEFAULT_verbose));
    bp::def("CP_quadratic_l1", CP_quadratic_l1_4
            , (bp::args("positivity")=DEFAULT_positivity, bp::args("PFDR_rho")=DEFAULT_PFDR_rho
            , bp::args("PFDR_condMin")=DEFAULT_PFDR_condMin, bp::args("CP_difTol")=DEFAULT_CP_difTol
            , bp::args("PFDR_difRcd")=DEFAULT_PFDR_difRcd, bp::args("PFDR_difTol")=DEFAULT_PFDR_difTol
            , bp::args("CP_itMax")=DEFAULT_CP_itMax, bp::args("PFDR_itMax")=DEFAULT_PFDR_itMax
            , bp::args("verbose")=DEFAULT_verbose));
    bp::def("CP_quadratic_l1", CP_quadratic_l1_5
            , (bp::args("positivity")=DEFAULT_positivity, bp::args("PFDR_rho")=DEFAULT_PFDR_rho
            , bp::args("PFDR_condMin")=DEFAULT_PFDR_condMin, bp::args("CP_difTol")=DEFAULT_CP_difTol
            , bp::args("PFDR_difRcd")=DEFAULT_PFDR_difRcd, bp::args("PFDR_difTol")=DEFAULT_PFDR_difTol
            , bp::args("CP_itMax")=DEFAULT_CP_itMax, bp::args("PFDR_itMax")=DEFAULT_PFDR_itMax
            , bp::args("verbose")=DEFAULT_verbose));
    bp::def("CP_quadratic_l1", CP_quadratic_l1_6
            , (bp::args("positivity")=DEFAULT_positivity, bp::args("PFDR_rho")=DEFAULT_PFDR_rho
            , bp::args("PFDR_condMin")=DEFAULT_PFDR_condMin, bp::args("CP_difTol")=DEFAULT_CP_difTol
            , bp::args("PFDR_difRcd")=DEFAULT_PFDR_difRcd, bp::args("PFDR_difTol")=DEFAULT_PFDR_difTol
            , bp::args("CP_itMax")=DEFAULT_CP_itMax, bp::args("PFDR_itMax")=DEFAULT_PFDR_itMax
            , bp::args("verbose")=DEFAULT_verbose));
    bp::def("CP_quadratic_l1", CP_quadratic_l1_7
            , (bp::args("positivity")=DEFAULT_positivity, bp::args("PFDR_rho")=DEFAULT_PFDR_rho
            , bp::args("PFDR_condMin")=DEFAULT_PFDR_condMin, bp::args("CP_difTol")=DEFAULT_CP_difTol
            , bp::args("PFDR_difRcd")=DEFAULT_PFDR_difRcd, bp::args("PFDR_difTol")=DEFAULT_PFDR_difTol
            , bp::args("CP_itMax")=DEFAULT_CP_itMax, bp::args("PFDR_itMax")=DEFAULT_PFDR_itMax
            , bp::args("verbose")=DEFAULT_verbose));
}

