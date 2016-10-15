from libpython_persistence_filter import *
from numpy import *
from scipy.stats import bernoulli

'''This function constructs and runs a Persistence Filter on a given sequence of timestamped observations (with P_M and P_F error rates), and returns the persistence probabilities at the 'query_times'''
def run_persistence_filter(Y_arr, t_arr, PM_arr, PF_arr, query_times, logS, init_time=0.0):
    
    if(not isinstance(PM_arr, (list, ndarray))):
        PM_arr = repeat(PM_arr, len(Y_arr))

    if(not isinstance(PF_arr, (list, ndarray))):
        PF_arr = repeat(PF_arr, len(Y_arr))

    #Construct persistence filter
    pf = PersistenceFilter(logS, init_time)

    #PREDICT:

    #Get the indices of all query_times prior to the first observation
    mask = query_times < t_arr[0]
    #Record the predictions for these query times 
    persistence_probs = map(pf.predict, query_times[mask])

    #UPDATE:
    pf.update(asscalar(Y_arr[0]), t_arr[0], PM_arr[0], PF_arr[0])



    for i in range(len(t_arr) - 1):
        #Get the indices of all query_times in [ t[i], t[i+1] )
        
        #PREDICT:
        mask = logical_and( (query_times >= t_arr[i]),  (query_times < t_arr[i+1]) )
        persistence_probs = append(persistence_probs, map(pf.predict, query_times[mask]))
        
        #UPDATE:
        pf.update(asscalar(Y_arr[i+1]), t_arr[i+1], PM_arr[i+1], PF_arr[i+1])

    
    #PREDICT
    mask = query_times >= t_arr[len(t_arr)-1]
    persistence_probs = append(persistence_probs, map(pf.predict, query_times[mask]))

    return persistence_probs

'''This function runs an empirical estimator on the available data, which simply returns the value of the last available observation'''
def run_empirical_estimator(Y_arr, t_arr, query_times):
    
    #PREDICT:

    #Get the indices of all query_times prior to the first observation
    mask = query_times < t_arr[0]
    #Record the predictions for these query times 
    vals = repeat(1.0, count_nonzero(mask))  #We start by assuming that the object is present

    #UPDATE:
    for i in range(len(t_arr) - 1):
        #Get the indices of all query_times in [ t[i], t[i+1] )
        
        #PREDICT:
        mask = logical_and( (query_times >= t_arr[i]),  (query_times < t_arr[i+1]) )
        vals = append(vals, repeat( (1.0 if Y_arr[i] else 0.0), count_nonzero(mask) ) )
        
    #PREDICT
    mask = query_times >= t_arr[len(t_arr)-1]
    vals = append(vals, repeat( (1.0 if Y_arr[i] else 0.0), count_nonzero(mask) ) )

    return vals


'''Sample a set of feature observation times according to a "bursty" Markov switching process that is intended to simulate random revisitations of a patrolling robot.  Here
lambda_r: The rate parameter for the exponentially-distributed inter-visitation time intervals
lambda_o: The rate parameter for the exponentially-distributed time intervals between each reobservation during a single revisitation
p_N: The probability of leaving the area after each observation of the feature; the expected number of feature reobservations per visitation is 1.0 / p_N
'''
def sample_observation_times(lambda_r, lambda_o, p_N, simulation_length):
    current_time = 0

    observation_times = array([])  #Initially empty array

    while current_time < simulation_length:
        #Sample the number of observations we will obtain on this revisit
        N = random.geometric(p_N)

        #Sample the inter_observation_times for this revisit
        inter_observation_times = random.exponential(1.0 / lambda_o, N)

        observation_times = append(observation_times, repeat(current_time, N) + cumsum(inter_observation_times))

        #Sample a revisitation interval
        revisit_interval = random.exponential(1.0 / lambda_r)
        
        #Advance the current time
        current_time = observation_times[len(observation_times) - 1] + revisit_interval


    #Return all of the observation times that fall within the specified interval
    return observation_times[ observation_times <= simulation_length ]
        

#Generate a sequence of observations at a specified set of times and with given error rates
def generate_observations(survival_time, observation_times, P_M, P_F):

    #A function to randomly sample an observation, conditioned upon whether or not the feature is still present
    sample_obs = lambda v : bernoulli.rvs(1-P_M) if v else bernoulli.rvs(P_F)

    Y_binary = array(map(sample_obs, observation_times <= survival_time))

    return Y_binary

'''This function computes and returns the empirical L1 error between the estimator belief state and the ground truth state (considered as continuous-but-discretely sampled functions of time) using the trapezoid rule.'''
def compute_L1_error(ground_truth, belief, query_times):
    return trapz(fabs(ground_truth - belief), query_times)

'''This function computes and returns the mean absolute error between the estimator belief state and the ground truth state'''
def compute_mean_absolute_error(ground_truth, belief, query_times):
    return compute_L1_error(ground_truth, belief, query_times) / (query_times[len(query_times) - 1] - query_times[0])


'''Given a sequence of ground truth feature states (present or absent) and a corresponding sequence of predicted feature states (present or absent), computes and returns the precision/recall curve for the ABSENT classification'''
def compute_feature_absence_precision_and_recall(ground_truth_states, predicted_states):

    #Get the 'absent' predictions by negation
    predicted_absences = logical_not(predicted_states)

    true_absences = logical_not(ground_truth_states)
    

    #PRECISION:
    #Compute total number of removal decisions
    num_predicted_absences = count_nonzero(predicted_absences)

    #Compute the number of removal predictions that are actually correct 
    num_correct_predicted_absences = count_nonzero(true_absences[predicted_absences])

    precision = float64(num_correct_predicted_absences) / num_predicted_absences
    

    #RECALL:
    #Compute the total number of true absences
    num_true_absences = count_nonzero(true_absences)

    #Compute the number of true absences that were correctly detected
    num_correctly_identified_true_absences = count_nonzero(predicted_absences[true_absences])

    recall = float64(num_correctly_identified_true_absences) / num_true_absences

    return (precision, recall)

    
    
    
    
    
