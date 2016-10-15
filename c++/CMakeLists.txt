cmake_minimum_required(VERSION 2.8.3)
project(PersistenceFilter)


if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=gnu++0x)  #Tell the compiler to compile with C++11 
endif()


set(GSL_LIB_DEPENDS gsl gslcblas m)

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

#Build the PersistenceFilter library
add_library(persistence_filter
	${CMAKE_CURRENT_LIST_DIR}/src/persistence_filter.cc
	${CMAKE_CURRENT_LIST_DIR}/src/persistence_filter_utils.cc
)
target_link_libraries(persistence_filter
	${GSL_LIB_DEPENDS}
)


add_executable(persistence_filter_test src/persistence_filter_test.cc)
target_link_libraries(persistence_filter_test persistence_filter ${GSL_LIB_DEPENDS})
