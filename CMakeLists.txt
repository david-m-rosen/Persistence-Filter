cmake_minimum_required(VERSION 2.8)

#SET(CMAKE_BUILD_TYPE "DEBUG")
SET(CMAKE_BUILD_TYPE "RELEASE")
#SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
#SET(CMAKE_BUILD_TYPE "MINSIZEREL")

IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    message("Building in 'debug' mode...")
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

IF(CMAKE_BUILD_TYPE MATCHES RELEASE)
    message("Building in 'release' mode...")
ENDIF(CMAKE_BUILD_TYPE MATCHES RELEASE)

project(PersistenceFilter)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=gnu++0x)  # Tell the compiler to compile with C++11 
    add_definitions("-Wall")  # Display all warnings
endif()

set(GSL_LIB_DEPENDS gsl gslcblas m)

# Build the persistece_filter library
add_library(persistence_filter_utils SHARED
	${CMAKE_CURRENT_LIST_DIR}/c++/src/persistence_filter_utils.cc
)
target_include_directories(persistence_filter_utils PUBLIC ${CMAKE_CURRENT_LIST_DIR}/c++/include)

add_library(persistence_filter SHARED ${CMAKE_CURRENT_LIST_DIR}/c++/src/persistence_filter.cc)
target_include_directories(persistence_filter PUBLIC ${CMAKE_CURRENT_LIST_DIR}/c++/include)
target_link_libraries(persistence_filter
  persistence_filter_utils
  ${GSL_LIB_DEPENDS}
)

add_executable(persistence_filter_test c++/src/persistence_filter_test.cc)
target_link_libraries(persistence_filter_test persistence_filter persistence_filter_utils ${GSL_LIB_DEPENDS})

#BOOST PYTHON STUFF GOES HERE!!!
find_package(PythonLibs)
find_package(Boost)
IF(Boost_FOUND AND PYTHONLIBS_FOUND)
  SET(Boost_USE_STATIC_LIBS OFF)
  SET(Boost_USE_MULTITHREADED ON)
  SET(Boost_USE_STATIC_RUNTIME OFF)

  # Build the Python persistence_filter_utils library
  add_library(python_persistence_filter_utils SHARED python/python_persistence_filter_utils.cc)
  target_include_directories(python_persistence_filter_utils PUBLIC ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
  target_link_libraries(python_persistence_filter_utils persistence_filter_utils ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${GSL_LIB_DEPENDS})

  # Build the Python persistence_filter library
  add_library(python_persistence_filter SHARED python/python_persistence_filter.cc)
  target_include_directories(python_persistence_filter PUBLIC ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
  target_link_libraries(python_persistence_filter persistence_filter ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

ELSEIF(NOT Boost_FOUND)
  MESSAGE(STATUS "Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()
