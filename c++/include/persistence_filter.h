#ifndef __PERSISTENCE_FILTER_H__
#define __PERSISTENCE_FILTER_H__

#include <functional>
#include <boost/optional.hpp>
#include <gsl/gsl_sf_exp.h>


/** This class implements the persistence filter algorithm for computing
 * Bayesian beliefs over the temporal persistence of features in semi-static
 * environments, as described in the paper "Towards Lifelong Feature-Based
 * Mapping in Semi-Static Environments", by D.M. Rosen, J. Mason, and J.J.
 * Leonard.
 */

class PersistenceFilter
{
 protected:

  /** The absolute time (wall-time) at which this filter was initialized*/
  double init_time_;

  /** The time of the last observation*/
  double tN_;

  /** The natural logarithm of the likelihood probability p(Y_{1:N} | t_N)*/
  double logpY_tN_;

  /** The natural logarithm of the lower evidence sum L(Y_{1:N}).  Note that we use a boost::optional value here, since L(Y_{1:0}) = 0 (i.e. this value is zero at initialization), for which the logarithm is undefined.  We initialize this running sum after the incorporation of the first observation.*/
  boost::optional<double> logLY_;

  /** The natural logarithm of the marginal (evidence) probability p(Y_{1:N})*/
  double logpY_;

  /** A function returning the natural logarithm of the survival function S_T() for the survival time prior p_T()*/
  std::function<double(double)> logS_;

  /** A function returning the natural logarithm of the time-shifted survival function S_T(t - initialization_time)*/
  std::function<double(double)> shifted_logS_;

  /** A helper function that computes the logarithm of the prior probability assigned to the range [t0, t1) by the shiftd survival time prior*/
  double shifted_logdF(double t1, double t0);


 public:
  /** One argument-constructor accepting a function that returns the logarithm of the survival function S_T() for the survival time prior p_T().*/
 PersistenceFilter(const std::function<double(double)>& log_survival_function, double initialization_time = 0.0) : init_time_(initialization_time),  tN_(initialization_time), logpY_tN_(0.0), logLY_(boost::none), logpY_(0.0), logS_(log_survival_function)

    {
      // GO GO GADGET LAMBDA CALCULUS!!!
      // A lambda function that we will use to shift observation and query times by the initialization time
      auto time_shifter = [](double time, double time_shift){ return time - time_shift; };

      // logS now
      shifted_logS_ = std::bind(logS_, std::bind(time_shifter, std::placeholders::_1, init_time_));
    }

  /** Updates the filter by incorporating a new detector output.  Here 'detector_output' is a boolean value output by the detector indicating whether the given feature was detected, 'observation_time' is the timestamp for the detection, and 'P_M' and 'P_F' give the detector's missed detection and false alarm probabilities for this observation, respectively.*/
  void update(bool detector_output, double observation_time, double P_M, double P_F);

  /** Compute the posterior feature persistence time p(X_t = 1 | Y_{1:N}) at time t >= tN (the time of the last observation).*/
  double predict(double prediction_time) const;

  /** Return the function computing the logarithm of the survival function.*/
  const std::function<double(double)>& logS() const
    {
      return logS_;
    }

  /** Return the function computing the logarithm of the shifted survival function*/
  const std::function<double(double)>& shifted_logS() const
    {
      return shifted_logS_;
    }

  /** Return the time of the last observation*/
  double last_observation_time() const
  {
    return tN_;
  }

  /** Return the absolute time (wall-time) at which this filter was initialized*/
  double initialization_time() const
  {
    return init_time_;
  }

  /** Return the likelihood probability p(Y_{1:N} | T >= t_N)*/
  double likelihood() const
  {
    return gsl_sf_exp(logpY_tN_);
  }

  /** Return the evidence probability p(Y_{1:N})*/
  double evidence() const
  {
    return gsl_sf_exp(logpY_);
  }

  /** Return the lower-sum probability L(Y_{1:N})*/
  double evidence_lower_sum() const
  {
    if(logLY_)
      return gsl_sf_exp(*logLY_);
    else
      return 0.0;
  }


  /** Nothing to do here*/
  ~PersistenceFilter() {}
};

#endif //__PERSISTENCE_FILTER_H__
