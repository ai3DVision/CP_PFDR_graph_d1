# Python Interface for Cut Pursuit
# author: Loic Landrieu
# date: 2018

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(CUT_PURSUIT)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules ${CMAKE_MODULE_PATH})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread -Wall -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -fopenmp -O3")
##############################
### Find required packages ###
##############################

include_directories(include)

find_package(Boost 1.65.0 COMPONENTS numpy REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package(PythonLibs)
message("PYTHON LIBRARIES ARE " ${PYTHON_LIBRARIES})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
LINK_DIRECTORIES(${PYTHON_LIBRARY_DIRS})

file(GLOB CP_HEADERS include/*.hpp)

message("HEADERS ARE " ${CP_HEADERS})

add_subdirectory(src)
