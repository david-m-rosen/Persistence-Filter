"""Unit test for the Python implementation of the Persistence Filter."""
import numpy as np

import PersistenceFilter as pf
import persistence_filter_utils as pf_utils


# Lower- and upper-bounds on the admissible rate parameters
lambda_l = .01
lambda_u = 1

# The logarithm of the survival function

logS_T = lambda t: pf_utils.log_general_purpose_survival_function(t, lambda_l, lambda_u)

# The survival function itself
S_T = lambda t: np.exp(logS_T(t))

# Detector error probabilities
P_M = .2
P_F = .01


test_filter = pf.PersistenceFilter(logS_T)


# We compute by hand the persistence filter posterior updates corrresponding
# to integrating a negative measurement at time t_1 = 1.0, a positive
# measurement at time t_2= 2.0, and a negative measurement at time t_3 = 3.0

t_1 = 1.0
t_2 = 2.0
t_3 = 3.0

# INCORPORATE FIRST OBSERVATION y_1 = 0 at t_1 = 1.0

# Update the filter
test_filter.update(False, t_1, P_M, P_F)
# Compute posterior prediction
filter_posterior1 = test_filter.predict(t_1)

# The likelihood p(y_1 = 0 | T >= t_1) = P_M.
pY1_t1 = P_M

# The evidence probability p(y_1 = 0) =
# p(y_1 = 0 | T >= t_1) * p(T >= t_1) + p(y_1 = 0 | T < t_1) * p(T < t_1)
pY1 = P_M * S_T(t_1) + (1 - P_F) * (1 - S_T(t_1))

# Compute the posterior p(X_{t_1} = 1 | y_1 = 0)
# = p(y_1 = 0 | T >= t_1) / p(y_1 = 0) * p(T >= t_1)
posterior1 = (pY1_t1 / pY1) * S_T(t_1)

print "FILTER STATE AFTER INCORPORATING y_1 = 0 at time t_1 = 1.0:"
print "Filter posterior probability p(X_{t_1} = 1 | y_1 = 0) = %f" % test_filter.predict(t_1)
print "True posterior probability p(X_{t_1} = 1 | y_1 = 0) = %f\n" % posterior1


# INCORPORATE SECOND OBSERVATION y_2 = 1 at t_2 = 2.0

# Update the filter
test_filter.update(True, t_2, P_M, P_F)
# Compute posterior prediction
filter_posterior2 = test_filter.predict(t_2)

# The likelihood p(y_1 = 0, y_2 = 1 | T >= t_2)
pY2_t2 = P_M * (1 - P_M)

# The evidence probability p(y_1 = 0, y_2 = 1) =
# p(y_1 = 0, y_2 = 1 | T > t_2) * p(T > t_2) +
# p(y_1 = 0, y_2 = 1 | t_1 <= T < t_2) * p(t_1 <= T < t_2) +
# p(y_1 = 0, y_2 = 1 | T < t_1) * p(t < t_1)

pY2 = P_M * (1 - P_M) * S_T(t_2) + P_M * P_F * (S_T(t_1) - S_T(t_2)) + (1 - P_F) * P_F * (1 - S_T(t_1))

# Compute the posterior p(X_{t_2} = 2 | y_1 = 0, y_2 = 1)
# = p(y_1 = 0, y_2 = 1 | T >= t_2) / p(y_1 = 0, y_2 = 1) * p(T >= t_2)
posterior2 = (pY2_t2 / pY2) * S_T(t_2)


print "FILTER STATE AFTER INCORPORATING y_2 = 1 at time t_2 = 2.0:"
print "Filter posterior probability p(X_{t_2} = 1 | y_1 = 0, y_2 = 1) = %f" % test_filter.predict(t_2)
print "True posterior probability p(X_{t_2} = 1 | y_1 = 0, y_2 = 1) = %f\n" % posterior2


# INCORPORATE THIRD OBSERVATION y_3 = 0 at t_3 = 3.0

# Update the filter
test_filter.update(False, t_3, P_M, P_F)
# Compute posterior prediction
filter_posterior3 = test_filter.predict(t_3)

# The likelihood p(y_1 = 0, y_2 = 1 | T >= t_2)
pY3_t3 = P_M * (1 - P_M) * P_M

# The evidence probability p(y_1 = 0, y_2 = 1, y_3 = 0) =
# p(y_1 = 0, y_2 = 1, y_3 = 0 | T > t_3) * p(T > t_3) +
# p(y_1 = 0, y_2 = 1, y_3 = 0 | t_2 <= T < t_3) * p(t_2 <= T < t_3) +
# p(y_1 = 0, y_2 = 1, y_3 = 0 | t_1 <= T < t_2) * p(t_1 <= T < t_2) +
# p(y_1 = 0, y_2 = 1, y_3 = 0 | T < t_1) * p(t < t_1)

pY3 = P_M * (1 - P_M) * P_M * S_T(t_3) + P_M * (1 - P_M) * (1 - P_F) * (S_T(t_2) - S_T(t_3)) + P_M * P_F * (1 - P_F) * (S_T(t_1) - S_T(t_2)) + (1 - P_F) * P_F * (1 - P_F) * (1 - S_T(t_1))

# Compute the posterior p(X_{t_2} = 2 | y_1 = 0, y_2 = 1)
# = p(y_1 = 0, y_2 = 1 | T >= t_2) / p(y_1 = 0, y_2 = 1) * p(T >= t_2)

posterior3 = (pY3_t3 / pY3) * S_T(t_3)


print "FILTER STATE AFTER INCORPORATING y_2 = 0 at time t_3 = 3.0:"
print "Filter posterior probability p(X_{t_3} = 1 | y_1 = 0, y_2 = 1, y_3 = 0) = %f" % test_filter.predict(t_3)
print "True posterior probability p(X_{t_3} = 1 | y_1 = 0, y_2 = 1, y_3 = 0) = %f\n" % posterior3
