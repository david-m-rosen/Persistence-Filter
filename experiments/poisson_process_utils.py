import numpy
from numpy.random import poisson as Poisson
from numpy.random import uniform as Uniform


'''This function computes and returns a list of events sampled from an inhomogeneous Poisson process
intensity_function:  The (nonnegative-real-valued) intensity function for the Poisson process.  This function accepts a single argument: a list of points from the region T whose representations are the same as those returned by the function uniform_sample_T below, and returns a NumPy array of the intensity function values at those points.
lambda_max:  An upper-bound for the intensity function over the sampling region T
areaT:  The area of the region over which to sample the Poisson process
uniform_sample_T:  This function accepts a single nonnegative integer argument N, and returns a list of N uniformly randomly sampled points from the region T
'''
def sample_Poisson_process(intensity_function, lambda_max, areaT, uniform_sample_T):
    N = Poisson(areaT*lambda_max)  #Sample the initial number of events to generate by drawing from a Poisson distribution w/ rate areaT*IF_upper_bound

    #Uniformly randomly sample a set of locations to associate with these events by drawing from 
    sampled_events = uniform_sample_T(N)

    'sampled_events now contains a set of samples drawn from a homogeneous Poisson process on T with (constant) intensity parameter IF_upper_bound; we now perform rejection sampling on this set using the value of intensity_function() evaluated on these sample points to generate a thinned sample set drawn from the target INhomogeneous Poisson process'

    #Compute vector of intensity function values
    intensity_func_vals = intensity_function(sampled_events)

    #Draw N standard uniform random variables
    U = Uniform(0,lambda_max,N)
    
    thinned_events_indicators =  (U <= intensity_func_vals)
    thinned_events = [event for (event, accepted) in zip(sampled_events, thinned_events_indicators) if accepted]

    return thinned_events
