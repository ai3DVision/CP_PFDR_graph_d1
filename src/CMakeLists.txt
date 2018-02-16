set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -O3 -Wall -fopenmp -std=c++11")

add_library(CP SHARED CP_PFDR_graph.cpp CP_PFDR_graph_quadratic_d1_l1.cpp operator_norm_matrix.cpp graph.cpp maxflow.cpp  PFDR_graph_quadratic_d1_l1.cpp ${CP_HEADERS})
target_link_libraries(CP
	${CP_HEADERS}
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
    boost_python
    )
