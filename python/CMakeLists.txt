# Python Interface for Cut Pursuit
# author: Loic Landrieu
# date: 2018

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

##############################
### Find required packages ###
##############################

find_package(Boost 1.65.0 COMPONENTS numpy3 python3 REQUIRED)
#for python2 replace the line above by the line below
#find_package(Boost 1.65.0 COMPONENTS numpy python REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package(PythonLibs)
message("PYTHON LIBRARIES ARE " ${PYTHON_LIBRARIES})
message("Boost LIBRARIES ARE " ${Boost_LIBRARIES})

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
LINK_DIRECTORIES(${PYTHON_LIBRARY_DIRS})

include_directories(../include)

file(GLOB CP_HEADERS ../include/*.hpp)

##############################
###      Build library     ###
##############################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -fopenmp -O3")

set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -O3 -DNDEBUG -Wall -fopenmp -std=c++11")

add_library(CP SHARED CP_quadratic_l1_py.cpp ../src/CP_PFDR_graph_quadratic_d1_l1.cpp ../src/operator_norm_matrix.cpp ../src/graph.cpp ../src/maxflow.cpp ../src/PFDR_graph_quadratic_d1_l1.cpp ${CP_HEADERS})

target_link_libraries(CP ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
