cmake_minimum_required(VERSION 3.10)

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

message (STATUS "${CMAKE_CXX_COMPILER_ID}")

if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "GNUCXX")
    add_definitions(-std=gnu++0x)  # Tell the compiler to compile with C++11 
    add_definitions("-Wall")  # Display all warnings
else()
    add_definitions(-std=c++11)  # Tell the compiler to compile with C++11 
    add_definitions("-Wall")  # Display all warnings
endif()



if (APPLE)
  set(GSL_LIB_DEPENDS
    /usr/local/lib/libgsl.a
    /usr/local/lib/libgslcblas.a
    m
  )
else()
  set(GSL_LIB_DEPENDS
    gsl
    gslcblas
    m
)
endif()


include_directories(${CMAKE_CURRENT_LIST_DIR}/c++/include)

# Build the persistece_filter library
add_library(persistence_filter_utils SHARED
	${CMAKE_CURRENT_LIST_DIR}/c++/src/persistence_filter_utils.cc
)

target_link_libraries(persistence_filter_utils
  ${GSL_LIB_DEPENDS}
)

add_library(persistence_filter SHARED ${CMAKE_CURRENT_LIST_DIR}/c++/src/persistence_filter.cc)
target_link_libraries(persistence_filter
  ${GSL_LIB_DEPENDS}
  persistence_filter_utils
)


add_executable(persistence_filter_test c++/src/persistence_filter_test.cc)
target_link_libraries(persistence_filter_test persistence_filter persistence_filter_utils ${GSL_LIB_DEPENDS})


#BOOST PYTHON STUFF GOES HERE!!!

find_package(Boost)
IF(Boost_FOUND)
  if (APPLE)
    include_directories("${Boost_INCLUDE_DIRS}" "/usr/local/opt/python@3.9/Frameworks/Python.framework/Versions/3.9/include/python3.9")
  else()
    include_directories("${Boost_INCLUDE_DIRS}" "/usr/include/python2.7")
  endif()
    
  SET(Boost_USE_STATIC_LIBS OFF)
  SET(Boost_USE_MULTITHREADED ON)
  SET(Boost_USE_STATIC_RUNTIME OFF)
  find_package(Boost COMPONENTS python3)
  find_package(PythonLibs 3.9 REQUIRED)

message(STATUS "${PYTHON_LIBRARIES}")

  # Build the Python persistence_filter_utils library
  add_library(python_persistence_filter_utils SHARED python/python_persistence_filter_utils.cc)
  target_link_libraries(python_persistence_filter_utils persistence_filter_utils ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} ${GSL_LIB_DEPENDS} )

  # Build the Python persistence_filter library
  add_library(python_persistence_filter SHARED python/python_persistence_filter.cc)
  target_link_libraries(python_persistence_filter persistence_filter ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} )


ELSEIF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()
