#include "persistence_filter.h"
#include "persistence_filter_utils.h"

#include <stdexcept>

#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_errno.h>



double PersistenceFilter::shifted_logdF(double t1, double t0)
{
  return logdiff(shifted_logS_(t0), shifted_logS_(t1));
}

void PersistenceFilter::update(bool detector_output, double observation_time, double P_M, double P_F)
{
  // Input checking:
  if(observation_time < tN_)
    {
      throw std::domain_error("Current observation must be at least as recent as the last incorporated observation (observation_time >= last_observation_time)");
    }

  if( (P_M < 0) || (P_M > 1) )
    {
      throw std::domain_error("Probability of missed detection must be between 0 and 1");
    }

  if( (P_F < 0) || (P_F > 1) )
    {
      throw std::domain_error("Probability of false alarm must be between 0 and 1");
    }

  // Perform the actual updates here...

  // Update the lower sum LY
  if(logLY_)  // If the running sum logLY has been previously initialized ...
    {
      // Update its value according to the paper
      *logLY_ = logsum(*logLY_, logpY_tN_ + shifted_logdF(observation_time, tN_)) + (detector_output ? gsl_sf_log(P_F) : gsl_sf_log(1 - P_F));
    }
  else
    {
      // This is the first observation we've incorporated, so initialize the lower running sum logLY here.
      // To do this, we use the following instantiation of equation (10) from the paper:
      //
      // L(Y_{1:1}) = p(Y_{1:1} | t_0)[F_T(t_1) - F_T(t_0)]
      //            = p(y_1 | t_0) * [F_T(t_1) - 0]
      //            = p(y_1 | t_0) * (1 - S_T(t_1)),
      //
      // so that
      // 
      // log Y_{1:1} = log p(y_1 | t_0) + log(1 - S_T(t_1))

      double log1_minus_ST;

      // Turn off GSL error handling -- we'll use our own for this computation
      gsl_error_handler_t* error_handler = gsl_set_error_handler_off();
      // Compute logarithm of 1 - S_T(t_1), using the fact that
      gsl_sf_result result;
      int status = gsl_sf_exp_e(shifted_logS_(observation_time), &result);
      // Reset original error handler here
      gsl_set_error_handler(error_handler);

      if(status == GSL_SUCCESS)
	{
	  log1_minus_ST = gsl_sf_log_1plusx(-result.val);
	}
      else
	{
	  // In this case, attempting to exponentiate logS_T(t_1) results in a numerical underflow,
	  // so we use the fact that for small x, log(1 - x) ~ -x, and therefore
	  
	  log1_minus_ST = 0.0;
	}

      logLY_ = boost::optional<double> ((detector_output ? gsl_sf_log(P_F) : gsl_sf_log(1 - P_F)) 
					+ log1_minus_ST);
    }

  // Postcondition:  At this point logLY is properly initialized.

  //Update the measurement likelihood pY_tN
  logpY_tN_ += (detector_output ? gsl_sf_log(1.0 - P_M) : gsl_sf_log(P_M));

  //Update the current observation time
  tN_ = observation_time;

  //Compute the marginal (evidence) probability pY
  logpY_ = logsum(*logLY_, logpY_tN_ + shifted_logS_(tN_));
}
 
double PersistenceFilter::predict(double prediction_time) const
{
  // Input checking
  if(prediction_time < tN_)
    {
      throw std::domain_error("Prediction time must be at least as recent as the last incorporated observation (prediction_time >= last_observation_time)");
    }
  
  double exp_arg = logpY_tN_ - logpY_ + shifted_logS_(prediction_time);
  gsl_error_handler_t* error_handler = gsl_set_error_handler_off();
  gsl_sf_result result;
  int status = gsl_sf_exp_e(exp_arg, &result);
  // Reset original error handler here
  gsl_set_error_handler(error_handler);

  if(status == GSL_SUCCESS)
    {
      return result.val;
    }
  else
    {
      return 0.0;
    }
}
