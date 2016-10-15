#include "persistence_filter_utils.h"

#include <algorithm> //To get swap function
#include <stdexcept>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_errno.h>

double log_general_purpose_survival_function(double t, double lambda_l, double lambda_u)
{
  // Input checking
  if(t < 0)
    {
      throw std::domain_error("Survival functions are defined on the nonnegative real line (t >= 0)");
    }

  if(lambda_l >= lambda_u)
    {
      throw std::domain_error("Parameter lambda_u must be greater than lambda_l");
    }

  // The actual computation...
  if(t > 0)
    {
      gsl_sf_result result;
      int status;
      double log_E1_lambda_l, log_E1_lambda_u;

      // Turn off GSL error handling -- we'll use our own for this computation
      gsl_error_handler_t* error_handler = gsl_set_error_handler_off();


      // COMPUTE log(E1(lambda_l * t))
      status = gsl_sf_expint_E1_e(lambda_l * t, &result);
      
      if(status == GSL_SUCCESS)
	{
	  log_E1_lambda_l = gsl_sf_log(result.val);
	}
      else
	{
	  // Attempting to compute E1(lamda_l * t) directly underflows, so we
	  // compute a close upper-bound approximation for log(E1(lambda_l * t))
	  // directly, using the bound E1(x) < exp(-x) * log(1 + 1/x)
	  log_E1_lambda_l = -lambda_l*t + gsl_sf_log(gsl_sf_log_1plusx( 1.0 / (lambda_l*t)));
	}

      status = gsl_sf_expint_E1_e(lambda_u * t, &result);
      
      if(status == GSL_SUCCESS)
	{
	  log_E1_lambda_u = gsl_sf_log(result.val);
	}
      else
	{
	  // Attempting to compute E_1(lamda_u * t) directly underflows, so we
	  // compute a close lower-bound approximation for log(E1(lambda_u * t))
	  // directly, using the bound (1/2) * exp(-x) log(1 + 2/x)
	  log_E1_lambda_u = -M_LN2 - lambda_u*t + gsl_sf_log(gsl_sf_log_1plusx(2 / (lambda_u*t)));
	}

      // Reset original error handler
      gsl_set_error_handler(error_handler);
      return logdiff(log_E1_lambda_l, log_E1_lambda_u) - gsl_sf_log(gsl_sf_log(lambda_u / lambda_l));
    }
  else
    {
      return 0;
    }
}

double logsum(double logx, double logy)
{
  //Ensure that logy <= logx by swapping them if necessary
  if(logy > logx)
    {
      std::swap(logx, logy);
    }

  // Now logy <= logx, so we can use the fact that 
  //
  // log(x + y) = log(x(1+ y /x)) = log(x) + log(1 + y/x)
  //
  // with y/x <= 1 to compute the logarithm of the sum stably

  // Turn off GSL error handling before computing this exponential -- we'll use our own for this computation
  gsl_error_handler_t* error_handler = gsl_set_error_handler_off(); 

  gsl_sf_result result;
  int status = gsl_sf_exp_e(logy - logx, &result);

  // Reset original error handler here
  gsl_set_error_handler(error_handler);

  if(status == GSL_SUCCESS)
    {
      return logx + gsl_sf_log_1plusx(result.val);
    }
  else
    {
      // If an error is thrown here, it must be because logy - logx is sufficiently negative to cause numerical underflow in the computation of the exponential.  
      // In that case, y << x, and therefore, using the Taylor series expansion for log(1 - z), we compute
      //
      // log(1 - y/x) ~ -y/x ~ 0 
  
      return logx;
    }
}

double logdiff(double logx, double logy)
{
  // Input checking
  if(logy > logx)
    {
      throw std::domain_error("logx must be greater than or equal to logy");
    }
  
  // We exploit the chain of equalities
  // log(x - y) = log(x * (1 - y/x)) = log(x) + log(1 - y/x) = log(1 + (-exp(logy - logx)))

  // Turn off GSL error handling before computing this exponential -- we'll use our own for this computation
  gsl_error_handler_t* error_handler = gsl_set_error_handler_off(); 

  gsl_sf_result result;
  int status = gsl_sf_exp_e(logy - logx, &result);

  // Reset original error handler here
  gsl_set_error_handler(error_handler);

  if(status == GSL_SUCCESS)
    {
      return logx + gsl_sf_log_1plusx(-result.val);
    }
  else
    {
      // If an error is thrown here, it must be because logy - logx is sufficiently negative to cause numerical underflow in the computation of the exponential.  
      // In that case, y << x, and therefore, using the Taylor series expansion for log(1 - z), we compute
      //	 
      // log(1 - y/x) ~ -y/x ~ 0 

      return logx;
    }
}
