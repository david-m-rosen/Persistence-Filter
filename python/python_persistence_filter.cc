#include "persistence_filter.h"

#include <iostream>
#include <functional>
#include <boost/python.hpp>

using namespace boost::python;
using namespace std;

// Wraps a Python function that accepts and returns a single floating-point value as a C++ function.
std::function<double(double)> wrap_python_function_as_cpp(PyObject* python_function)
{
  auto call_python_function = [=] (double x) {return call<double>(python_function, x); };
  return call_python_function;									    
}

// Factory methods for instantiating C++ PersistenceFilters whose log-survival functions are C++-wrapped native Python functions

// A function that constructs a C++ PersistenceFilter class from a Python log-survival function
boost::shared_ptr<PersistenceFilter> persistence_filter_from_python(const object& python_log_survival_function)
{
  return boost::shared_ptr<PersistenceFilter>(new PersistenceFilter(wrap_python_function_as_cpp(python_log_survival_function.ptr())));
}

// A function that constructs a C++ PersistenceFilter class from a Python log-survival function and an initialization time
boost::shared_ptr<PersistenceFilter> persistence_filter_with_initialization_time_from_python(const object& python_log_survival_function, double init_time)
{
  return boost::shared_ptr<PersistenceFilter>(new PersistenceFilter(wrap_python_function_as_cpp(python_log_survival_function.ptr()), init_time));
}

// Python interface to the C++ PersistenceFilter implementation
BOOST_PYTHON_MODULE(libpython_persistence_filter)
{
  //def("test_passthrough", eval_function_at_point_5)

  class_<PersistenceFilter, boost::shared_ptr<PersistenceFilter> >("PersistenceFilter", no_init)  // Declare this class without a default constructor...

    // ... and then bind custom "factory methods" to the __init__ function that can wrap passed-in native Python functions as instances of std::function<double(double)> before being passed in to the C++ PersistenceFilter class's constructor.
    .def("__init__", make_constructor(&persistence_filter_from_python) )
    .def("__init__", make_constructor(&persistence_filter_with_initialization_time_from_python))
  
    .def("update", &PersistenceFilter::update)
    .def("predict", &PersistenceFilter::predict)
    .def("last_observation_time", &PersistenceFilter::last_observation_time)
    .def("initialization_time", &PersistenceFilter::initialization_time)
    .def("likelihood", &PersistenceFilter::likelihood)
    .def("evidence", &PersistenceFilter::evidence)
    .def("evidence_lower_sum", &PersistenceFilter::evidence_lower_sum)

    ;
}

