#include "persistence_filter.h"
#include "persistence_filter_utils.h"

#include <functional>
#include <iostream>
#include <gsl/gsl_sf_exp.h>


using namespace std;

double lambda_u = 1;
double lambda_l = .01;

double P_M = .2;
double P_F = .01;


std::function<double(double)> logS_T = std::bind(log_general_purpose_survival_function, std::placeholders::_1, lambda_l, lambda_u);

auto S_T = [](double t) { return gsl_sf_exp(logS_T(t)); };

int main(int argc, char* argv[])
{

  PersistenceFilter filter(logS_T);
  

  //We compute by hand the persistence filter posterior updates corrresponding to integrating a negative measurement at time t_1 = 1.0, a positive measurement at time t_2= 2.0, and a negative measurement at time t_3 = 3.0
  double t_1 = 1.0;
  double t_2 = 2.0;
  double t_3 = 3.0;

  // INCORPORATE FIRST OBSERVATION y_1 = 0 at t_1 = 1.0

  filter.update(false, t_1, P_M, P_F);  // Update the filter
  double filter_posterior1 = filter.predict(t_1);  // Compute posterior prediction

  //The likelihood p(y_1 = 0 | T >= t_1) = P_M.
  double pY1_t1 = P_M;
  
  //The evidence probability p(y_1 = 0) = p(y_1 = 0 | T >= t_1) * p(T >= t_1) + p(y_1 = 0 | T < t_1) * p(T < t_1)
  double pY1 = P_M * S_T(t_1) + (1 - P_F) * (1 - S_T(t_1));

  //Compute the posterior p(X_{t_1} = 1 | y_1 = 0) = p(y_1 = 0 | T >= t_1) / p(y_1 = 0) * p(T >= t_1)
  double posterior1 = (pY1_t1 / pY1) * S_T(t_1);

  cout<<"FILTER STATE AFTER INCORPORATING y_1 = 0 at time t_1 = 1.0"<<endl;
  cout<<"Filter likelihood p(y_1 = 0 | T > t_1) = "<<filter.likelihood()<<endl;
  cout<<"True likelihood p(y_1 = 0 | T > t_1) = "<<pY1_t1 << endl;
  cout<<"Filter evidence p(y_1 = 0) = "<<filter.evidence()<<endl;
  cout<<"True evidence p(y_1 = 0) = "<<pY1<<endl;
  cout<<"Filter posterior probability p(X_{t_1} = 1 | y_1 = 0) = "<<filter.predict(t_1)<<endl;
  cout<<"True posterior probability p(X_{t_1} = 1 | y_1 = 0) = "<<posterior1<<endl<<endl;




  // INCORPORATE SECOND OBSERVATION y_2 = 1 at t_2 = 2.0

  filter.update(true, t_2, P_M, P_F);  // Update the filter
  double filter_posterior2 = filter.predict(t_2);  // Compute posterior prediction

  //The likelihood p(y_1 = 0, y_2 = 1 | T >= t_2)
  double pY2_t2 = P_M * (1-P_M);

  // The evidence probability p(y_1 = 0, y_2 = 1) =
  // p(y_1 = 0, y_2 = 1 | T > t_2) * p(T > t_2) +
  // p(y_1 = 0, y_2 = 1 | t_1 <= T < t_2) * p(t_1 <= T < t_2) +
  // p(y_1 = 0, y_2 = 1 | T < t_1) * p(t < t_1)

  double pY2 = P_M * (1 - P_M) * S_T(t_2)
    + P_M * P_F * (S_T(t_1) - S_T(t_2))
    + (1 - P_F) * P_F * (1 - S_T(t_1));

  // Compute the posterior p(X_{t_2} = 2 | y_1 = 0, y_2 = 1) = p(y_1 = 0, y_2 = 1 | T >= t_2) / p(y_1 = 0, y_2 = 1) * p(T >= t_2)

  double posterior2 = (pY2_t2 / pY2) * S_T(t_2);


  cout<<"FILTER STATE AFTER INCORPORATING y_2 = 1 at time t_2 = 2.0"<<endl;
  cout<<"Filter likelihood p(y_1 = 0, y_2 = 1 | T > t_2) = "<<filter.likelihood()<<endl;
  cout<<"True likelihood p(y_1 = 0, y_2 = 1 | T > t_2) = "<<pY2_t2 << endl;
  cout<<"Filter evidence p(y_1 = 0, y_2 = 1) = "<<filter.evidence()<<endl;
  cout<<"True evidence p(y_1 = 0, y_2 = 1) = "<<pY2<<endl;
  cout<<"Filter posterior probability p(X_{t_2} = 1 | y_1 = 0, y_2 = 1) = "<<filter.predict(t_2)<<endl;
  cout<<"True posterior probability p(X_{t_2} = 1 | y_1 = 0, y_2 = 1) = "<<posterior2<<endl<<endl;




  // INCORPORATE THIRD OBSERVATION y_3 = 0 at t_3 = 3.0

  filter.update(false, t_3, P_M, P_F);  // Update the filter
  double filter_posterior3 = filter.predict(t_3);  // Compute posterior prediction


  //The likelihood p(y_1 = 0, y_2 = 1 | T >= t_2)
  double pY3_t3 = P_M * (1-P_M) * P_M;

  // The evidence probability p(y_1 = 0, y_2 = 1, y_3 = 0) =
  // p(y_1 = 0, y_2 = 1, y_3 = 0 | T > t_3) * p(T > t_3) +
  // p(y_1 = 0, y_2 = 1, y_3 = 0 | t_2 <= T < t_3) * p(t_2 <= T < t_3) +
  // p(y_1 = 0, y_2 = 1, y_3 = 0 | t_1 <= T < t_2) * p(t_1 <= T < t_2) +
  // p(y_1 = 0, y_2 = 1, y_3 = 0 | T < t_1) * p(t < t_1)

  double pY3 = P_M * (1 - P_M) * P_M * S_T(t_3)
    + P_M * (1 - P_M) * (1 - P_F) * (S_T(t_2) - S_T(t_3))
    + P_M * P_F * (1 - P_F) * (S_T(t_1) - S_T(t_2))
    + (1 - P_F) * P_F * (1 - P_F) * (1 - S_T(t_1));

    // Compute the posterior p(X_{t_2} = 2 | y_1 = 0, y_2 = 1) = p(y_1 = 0, y_2 = 1 | T >= t_2) / p(y_1 = 0, y_2 = 1) * p(T >= t_2)

  double posterior3 = (pY3_t3 / pY3) * S_T(t_3);


  cout<<"FILTER STATE AFTER INCORPORATING y_3 = 0 at time t_3 = 3.0"<<endl;
  cout<<"Filter likelihood p(y_1 = 0, y_2 = 1, y_3 = 0 | T > t_3) = "<<filter.likelihood()<<endl;
  cout<<"True likelihood p(y_1 = 0, y_2 = 1, y_3 = 0 | T > t_3) = "<<pY3_t3 << endl;
  cout<<"Filter evidence p(y_1 = 0, y_2 = 1, y_3 = 0) = "<<filter.evidence()<<endl;
  cout<<"True evidence p(y_1 = 0, y_2 = 1, y_3 = 0) = "<<pY3<<endl;
  cout<<"Filter posterior probability p(X_{t_3} = 1 | y_1 = 0, y_2 = 1, y_3 = 0) = "<<filter.predict(t_3)<<endl;
  cout<<"True posterior probability p(X_{t_3} = 1 | y_1 = 0, y_2 = 1, y_3 = 0) = "<<posterior3<<endl<<endl;
}
