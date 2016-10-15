#ifndef __PERSISTENCE_FILTER_UTILS_H__
#define __PERSISTENCE_FILTER_UTILS_H__

/**This function implements the survival function for the general-purpose survival-time prior developed in the RSS workshop paper "Towards Lifelong Feature-Based Mapping in Semi-Static Environments".*/
double log_general_purpose_survival_function(double t, double lambda_l, double lambda_u);

/**Computes log(x + y) from log(x), log(y) in a numerically stable way.*/
double logsum(double logx, double logy);
  
/**Computes log(x - y) from log(x), log(y) in a numerically stable way.  Note that here we require x > y.*/
double logdiff(double logx, double logy);


#endif //__PERSISTENCE_FILTER_UTILS_H__
