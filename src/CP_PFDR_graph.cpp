/*
 * Python wrapper for Cut Pursuit for solving inverse problems regularized with
 * graph total variation and l1 penalty and positivity constraint
 * Loic Landrieu 2018

 * minimize the following functional over a graph G = (V, E)
 *
 * F(x) = 1/2 ||A x - obs||^2
 *      + sum_{(u,v) in E)}{edge_weight(u,v) * |x_u - x_v|}
 *      + sum_{v in V)}{l1_weight(v) * |x_v|}
 * with positivity constrainty of x

 * SYNTAX
 * cV, rX = CP_PFDR_graph(obs, source, target, edge_weight, A, l1_weight, positivity, speed)
 *
 * INPUTS
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
 *      scalar, in which case A is a scalar matrix
 * l1_weight = weight associated with each vertex in the L1 regularization
 *      vector, numpy array of shape (V,)
 *      scalar, in which case all weights are the same
 * positivity : if nonzero, add positivity constraints
 *      scalar
 * speed : define the speed setting
 *      scalar
 *      0 -> slow but precise
 *      1 -> standard
 *      2 -> fast but less precise (for prototyping)
 *
 * OUPUTS
 * cV = index of the component of each vertex
 *     numpy array of shape (V,)
 * rX = value associated to each constant component
 *      numpy array of shape (rV,) with rV the number of constant components
 *
 * x = rX[cV]
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


namespace bpn = boost::python::numpy;
namespace bp =  boost::python;

enum A_mode {identity, scalar, diagonal, matrix};

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
void set_speed(int speed, T * CP_difTol, int * CP_itMax, T * PFDR_difRcd, T * PFDR_difTol, int * PFDR_itMax)
{   //tune the parameters to the desired speed setting
    switch(speed) {
        case 0 :
            std::cout << "Speed setting = SLOW" << std::endl;
            *CP_difTol = 1e-4;
            *CP_itMax = 25;
            *PFDR_difRcd = 1e-4;
            *PFDR_difTol = 1e-6;
            *PFDR_itMax = 1e5;
            break;
        case 2 :
            std::cout << "Speed setting = FAST" << std::endl;
            *CP_difTol = 1e-2;
            *CP_itMax = 5;
            *PFDR_difRcd = 1e-2;
            *PFDR_difTol = 1e-3;
            *PFDR_itMax = 1e3;
            break;
        default:
            std::cout << "Speed setting = NORMAL" << std::endl;
            *CP_difTol = 1e-3;
            *CP_itMax = 10;
            *PFDR_difRcd = 0;
            *PFDR_difTol = 1e-4;
            *PFDR_itMax = 1e4;
            break;
    }
}

template<class T>
bp::tuple CP_PFDR_graph_from_array(int N, int V, int E, int edge_weight_length, int l1_weight_length, A_mode a_mode
                       , T * obs_data, T * A_data , const int * source_data, const int * target_data
                       , T * edge_weight_data, T * l1_weight_data, const int positivity, const int speed)
{   //call the c+ function with the proper type
    T * A_data_ = nullptr, * obs_data_ = nullptr;
    //--- adapt variable inputs ---
    if (a_mode == scalar)
    {   //create a constant vector
        T scalar = A_data[0];
        A_data = (T* )malloc(V*sizeof(T));
        for (int i = 0; i < V; i++) {A_data[i] = scalar;}
    }
    if (a_mode == scalar || a_mode == diagonal)
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
    {   //create a constant vector
        float scalar = edge_weight_data[0];
        edge_weight_data = (T* )malloc(E*sizeof(T));
        for (int i = 0; i < V; i++) {edge_weight_data[i] = scalar;}
    }
    if (l1_weight_length == 1)
    {   //create a constant vector
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
    //--- speed parameters ---
    T PFDR_rho = 1.0;
    T PFDR_condMin = 1e-3;
    int verbose = 1000;
    T CP_difTol[1];
    int CP_itMax[1];
    T PFDR_difRcd[1];
    T PFDR_difTol[1];
    int PFDR_itMax[1];
    set_speed<T>(speed, CP_difTol, CP_itMax, PFDR_difRcd, PFDR_difTol, PFDR_itMax);
    //---call to c++ function---
    if (a_mode == scalar || a_mode == diagonal)
    {
        CP_PFDR_graph_quadratic_d1_l1<T>(V, E, N, &rV, Cv, &rX, obs_data_, A_data_, \
            source_data, target_data, edge_weight_data, l1_weight_data, positivity, *CP_difTol, *CP_itMax, CP_it, PFDR_rho, \
            PFDR_condMin, *PFDR_difRcd, *PFDR_difTol, *PFDR_itMax, Time, Obj, \
            Dif, verbose, NULL);
    }else
    {
        CP_PFDR_graph_quadratic_d1_l1<T>(V, E, N, &rV, Cv, &rX, obs_data, A_data, \
            source_data, target_data, edge_weight_data, l1_weight_data, positivity, *CP_difTol, *CP_itMax, CP_it, PFDR_rho, \
            PFDR_condMin, *PFDR_difRcd, *PFDR_difTol, *PFDR_itMax, Time, Obj, \
            Dif, verbose, NULL);
    }
    //--- freeing memory ---
    if (a_mode == scalar)
    {
        free(A_data);
    }
    if (a_mode == scalar || a_mode == diagonal)
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

bp::tuple CP_PFDR_graph(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      , const bpn::ndarray & edge_weight, const bpn::ndarray & A
                      , const bpn::ndarray & l1_weight, const int positivity, const int speed)
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
            a_mode = scalar;
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
    std::cout<<"Type detected = " << data_classname <<std::endl;

    if (!data_classname.compare("float32") || !data_classname.compare("f4"))
    {
        return CP_PFDR_graph_from_array<float>(N, V, E, bp::len(edge_weight), bp::len(l1_weight), a_mode
                                                  , reinterpret_cast<float*>(obs.get_data())
                                                  , reinterpret_cast<float*>(A.get_data())
                                                  , reinterpret_cast<int*>(source.get_data())
                                                  , reinterpret_cast<int*>(target.get_data())
                                                  , reinterpret_cast<float*>(edge_weight.get_data())
                                                  , reinterpret_cast<float*>(l1_weight.get_data())
                                                  , positivity, speed);
    }
    else if (!data_classname.compare("float64") || !data_classname.compare("f8"))
    {
        return CP_PFDR_graph_from_array<double>(N, V, E, bp::len(edge_weight), bp::len(l1_weight), a_mode
                                                  , reinterpret_cast<double*>(obs.get_data())
                                                  , reinterpret_cast<double*>(A.get_data())
                                                  , reinterpret_cast<int*>(source.get_data())
                                                  , reinterpret_cast<int*>(target.get_data())
                                                  , reinterpret_cast<double*>(edge_weight.get_data())
                                                  , reinterpret_cast<double*>(l1_weight.get_data())
                                                  , positivity, speed);
    }
    else
    {
       {throw std::invalid_argument("Type unknown, must be float32, float64, f4, or f8.");}
    }
}

bp::tuple CP_PFDR_graph_1(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      , double edge_weight, const bpn::ndarray & A
                      , const bpn::ndarray & l1_weight, const int positivity, const int speed)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray edge_weight_ = bpn::zeros(shape, data_type);
    edge_weight_[0] = edge_weight;
    return CP_PFDR_graph(obs, source, target, edge_weight_, A, l1_weight, positivity, speed);
}

bp::tuple CP_PFDR_graph_2(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      ,  const bpn::ndarray & edge_weight, double A
                      , const bpn::ndarray & l1_weight, const int positivity, const int speed)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray A_ = bpn::zeros(shape, data_type);
    A_[0] = A;
    return CP_PFDR_graph(obs, source, target, edge_weight, A_, l1_weight, positivity, speed);
}

bp::tuple CP_PFDR_graph_3(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      ,  const bpn::ndarray & edge_weight, const bpn::ndarray & A
                      , double l1_weight, const int positivity, const int speed)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray l1_weight_ = bpn::zeros(shape, data_type);
    l1_weight_[0] = l1_weight;
    return CP_PFDR_graph(obs, source, target, edge_weight, A, l1_weight_, positivity, speed);
}

bp::tuple CP_PFDR_graph_4(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      , double edge_weight, double A
                      , const bpn::ndarray & l1_weight, const int positivity, const int speed)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray edge_weight_ = bpn::zeros(shape, data_type);
    bpn::ndarray A_ = bpn::zeros(shape, data_type);
    edge_weight_[0] = edge_weight;
    A_[0] = A;
    return CP_PFDR_graph(obs, source, target, edge_weight_, A_, l1_weight, positivity, speed);
}

bp::tuple CP_PFDR_graph_5(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      ,  double edge_weight, const bpn::ndarray & A
                      , double l1_weight, const int positivity, const int speed)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray edge_weight_ = bpn::zeros(shape, data_type);
    bpn::ndarray l1_weight_ = bpn::zeros(shape, data_type);
    edge_weight_[0] = edge_weight;
    l1_weight_[0] = l1_weight;
    return CP_PFDR_graph(obs, source, target, edge_weight_, A, l1_weight_, positivity, speed);
}

bp::tuple CP_PFDR_graph_6(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      ,  const bpn::ndarray & edge_weight, double A
                      , double l1_weight, const int positivity, const int speed)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray l1_weight_ = bpn::zeros(shape, data_type);
    bpn::ndarray A_ = bpn::zeros(shape, data_type);
    l1_weight_[0] = l1_weight;
    A_[0] = A;
    return CP_PFDR_graph(obs, source, target, edge_weight, A_, l1_weight_, positivity, speed);
}

bp::tuple CP_PFDR_graph_7(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target
                      ,  double edge_weight, double A
                      , double l1_weight, const int positivity, const int speed)
{
    bpn::dtype data_type = obs.get_dtype();
    bp::tuple shape = bp::make_tuple(1);
    bpn::ndarray edge_weight_ = bpn::zeros(shape, data_type);
    bpn::ndarray l1_weight_ = bpn::zeros(shape, data_type);
    bpn::ndarray A_ = bpn::zeros(shape, data_type);
    edge_weight_[0] = edge_weight;
    l1_weight_[0] = l1_weight;
    A_[0] = A;
    return CP_PFDR_graph(obs, source, target, edge_weight_, A_, l1_weight_, positivity, speed);
}

BOOST_PYTHON_MODULE(libCP)
{
    _import_array();
    Py_Initialize();
    bpn::initialize();
    bp::def("CP_PFDR_graph", CP_PFDR_graph);
    bp::def("CP_PFDR_graph", CP_PFDR_graph_1);
    bp::def("CP_PFDR_graph", CP_PFDR_graph_2);
    bp::def("CP_PFDR_graph", CP_PFDR_graph_3);
    bp::def("CP_PFDR_graph", CP_PFDR_graph_4);
    bp::def("CP_PFDR_graph", CP_PFDR_graph_5);
    bp::def("CP_PFDR_graph", CP_PFDR_graph_6);
    bp::def("CP_PFDR_graph", CP_PFDR_graph_7);
}

