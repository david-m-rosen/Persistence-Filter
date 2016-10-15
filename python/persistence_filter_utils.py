from numpy import log, exp, log1p
from scipy.special import expn

def logsum(logx, logy):
    #Ensure that logy < logx by swapping them if necessary
    if logx < logy:
        logy, logx = logx, logy

    # Now logy < logx, so we can use the fact that
    # log(x + y) = log(x (1 + y/x) )
    # = logx + log(1 + y/x)
    # = logx + log1p(exp(logy - logx))
    # with y/x <= 1 to compute the logarithm of the sum stably

    return logx + log1p(exp(logy - logx))


def logdiff(logx, logy):
    return logx + log1p(-exp(logy - logx))


def log_general_purpose_survival_function(t, lambda_l, lambda_u):
    if t > 0.0:
        return logdiff(log( expn(1, t*lambda_l)), log(expn(1, t*lambda_u))) - log(log(lambda_u) - log(lambda_l))
    else:
        return 0.0
