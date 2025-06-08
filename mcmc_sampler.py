"""
    mcmc_sampler is the MCMC sampler funciton 

    Use monte-carlo techniques to sample from a probability density P(x)
    that is only known up to a constant. 
        - kernel function f(x) 
        - transition g(x|x')

"""
from scipy.stats import norm
import numpy as np 
import random

def g(x_t, sig=1): 
    """
        transition function. Assumed to be symmetric 
    """
    return norm.rvs(loc=x_t,scale=sig)

class MCMCSampler:
    """
        A MCMC sampler  

        kernel_func - function I am developing a MCMC to sample from 
        scale (float) - relative scale of the transition function 
            how much it explores space 
        proposal_func - transition fuction. Should be symetric 
        seed (float) - starting point
        thinning (int) - number of rounds before selecting. 
        
    """
    eps = 1e-6  # to avoid divide by zero error 

    def __init__(self, kernel_func, scale=1, proposal_func=None, seed=0.0, thinning=10):

        # kernel func
        self.kernel = kernel_func

        self.g = proposal_func
        if not proposal_func:
            self.g = lambda x: g(x, sig=scale)

        self.x_t = seed 
        self.thin = thinning

    def accept_rate(self,x,x_prime):
        """
            calculate the acceptance ratio  
        """
        return self.kernel(x_prime) / self.kernel(x + self.eps) 


    def gen_sample(self):

        for i in range(self.thin):
            x_p = self.g(self.x_t)
            alpha = self.accept_rate(self.x_t, x_p)

            if random.random() <= alpha:
                self.x_t = x_p 

        return self.x_t

    def rvs(self, size=1):
        """
            return size random variates   

            Args:
                size (int) - number of values to return 
        """
        if size < 2:
            return self.gen_sample() 

        return np.array([self.gen_sample() for i in range(size)])



if __name__ == '__main__':
    pass

