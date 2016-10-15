#include "persistence_filter_utils.h"
#include <boost/python.hpp>

using namespace boost::python;

BOOST_PYTHON_MODULE(libpython_persistence_filter_utils)
{
  def("log_general_purpose_survival_function", log_general_purpose_survival_function, "This function implements the survival function for the general-purpose survival-time prior developed in the RSS workshop paper 'Towards Lifelong Feature-Based Mapping in Semi-Static Environments'");

  def("logsum", logsum, "Computes log(x + y) from log(x), log(y) in a numerically stable way.");

  def("logdiff", logdiff, "Computes log(x - y) from log(x), log(y) in a numerically stable way.  Note that here we require x > y.");
}

