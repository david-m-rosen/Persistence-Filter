"""Python implementation of the persistence filter algorithm."""
from numpy import log, exp, log1p
from persistence_filter_utils import logsum, logdiff

class PersistenceFilter:
  """A Python class implementing the Persistence Filter algorithm.

  This class implements the persistence filter algorithm for computing
  Bayesian beliefs over the temporal persistence of features in semi-static
  environments, as described in the paper "Towards Lifelong Feature-Based
  Mapping in Semi-Static Environments", by D.M. Rosen, J. Mason, and J.J.
  Leonard, from the RSS workshop "The Problem of Mobile Sensors", 2015.
  """
  def __init__(self, log_survival_function, initialization_time=0.0):
    """Constructor for the Persistence Filter class.

    Args:
      log_survival_function:  A function that accepts a single floating-
        point value 't', and returns log S(t), the _logarithm_ of the survival
        function at time t.
      initialization_time:  An optional floating-point value giving the absolute
        time (wall-time) at which the filter was initialized.  Defaults to 0.0.
    """
    #A function that returns the natural logarithm of the survival function S_T()
    self._log_survival_function = log_survival_function
        
    #The timestamp at which this feature was first observed
    self._initialization_time = initialization_time
    
    #The timestamp of the last detector output for this feature
    self._last_observation_time = initialization_time
        
    #The natural logarithm of the likelihood probability p(Y_{1:N} | t_N)
    self._log_likelihood = 0.0
        
    #The natural logarithm of the lower partial sum L(Y_{1:N}).  Note that we initialize this value as 'None', since L(Y_{1:0}) = 0 (i.e. this value is zero at initialization), for which the logarithm is undefined.  We initialize this running sum after the incorporation of the first observation.
    self._log_lower_evidence_sum = None
        
    #The natural logarithm of the marginal (evidence) probability p(Y_{1:N})
    self._log_evidence = 0.0
        
    #A function returning the value of the survival time prior based upon the ELAPSED time since the feature's instantiation
    self._shifted_log_survival_function = lambda t: self._log_survival_function(t - self._initialization_time)
        
    #A function that computes the logarithm of the prior probability assigned to the range [t0, t1) by the shifted survival time prior p_T()
    self._shifted_logdF = lambda t1, t0: logdiff(self._shifted_log_survival_function(t0), self._shifted_log_survival_function(t1))
        
  def update(self, detector_output, observation_time, P_M, P_F):
    """Updates the filter by incorporating a new detector output.

    Args:
      detector_output:  A Boolean value output by the detector
        indicating whether the given feature was detected.
      observation_time:  The timestamp for the detection.
      missed_detection_probability:  A floating-point value
        in the range [0, 1] indicating the missed detection probability
        of the feature detector.
      false_alarm_probability:  A floating-point value in the
        range [0,1] indicating the false alarm probability of the feature
        detector.
    """
    #Update the lower sum LY
    if self._log_lower_evidence_sum is not None:
      #_log_lower_evidence_sum has been previously initialized, so just update it in the usual way
      self._log_lower_evidence_sum = logsum(self._log_lower_evidence_sum, self._log_likelihood + self._shifted_logdF(observation_time, self._last_observation_time)) + \
                         (log(P_F) if detector_output else log(1 - P_F))
    else:
      #This is the first observation we've incorporated; initialize the logarithm of lower running sum here
      self._log_lower_evidence_sum = (log(P_F) if detector_output else log(1 - P_F)) + log1p(-exp(self._shifted_log_survival_function(observation_time)))
        
    #Post-condition: at this point, self._log_lower_evidence_sum is a properly-initialized value

    #Update the measurement likelihood pY_tN
    self._log_likelihood = self._log_likelihood + (log(1.0 - P_M) if detector_output else log(P_M))

    #Update the last observation time
    self._last_observation_time = observation_time

    #Compute the marginal (evidence) probability pY
    self._log_evidence = logsum(self._log_lower_evidence_sum, self._log_likelihood + self._shifted_log_survival_function(self._last_observation_time))

  def predict(self, prediction_time):
    """Compute the posterior persistence probability p(X_t = 1 | Y_{1:N}).

    Args:
      prediction_time:  A floating-point value in the range
        [last_observation_time, infty) indicating the time t
        for which to compute the posterior survival belief p(X_t = 1 | Y_{1:N})

    Returns:
      A floating-point value in the range [0, 1] giving the
      posterior persistence probability p(X_t = 1 | Y_{1:N}).
    """
    return exp(self._log_likelihood - self._log_evidence + self._shifted_log_survival_function(prediction_time))

  @property
  def log_survival_function(self):
    return self._log

  @property
  def shifted_log_survival_function(self):
    return self._shifted_log_survival_function
    
  @property
  def last_observation_time(self):
    return self._last_observation_time

  @property
  def initialization_time(self):
    return self._initialization_time
