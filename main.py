import os.path 
import numpy as np 
from test_sampler import SamplerTester
from mcmc_sampler import MCMCSampler

min_x = -4
max_x = 6

def camel_hump(x):
    """
        Example similar to the wikipedia 
    """
    As = [2,1]
    mus = [-0.5,2.5]
    sigs = [1,1]

    out=0
    for a, mu, sig in zip(As, mus,sigs):
        out += a* np.exp(-(x-mu)**2/(2*sig**2)) 
    return out

def dirac(x): 
    if x > 0: 
        return np.exp(-x)
    else:
        return np.exp(x)

def resonance(x):
    out = 0
    out += 2 * np.exp(-(x)**2/(2*0.1**2))
    out += 0.5 * np.exp(-(x)**2/(2*2**2))
    return out

def step(x):
    if 0 < x and x < 1:
        return 1.
    return 0.

def exp_poly(x):
    if x > 0:
        return (1+3*x+0.4*x**2)* np.exp(-x**2)
    else:
        return 0 


def normalize_func(func, min_x=-6, max_x=6, n=500):
    """
        return a normalized version of the function 
    """
    x = np.linspace(min_x,max_x,n)
    dx = x[1] - x[0]
    const = np.sum([func(xi) for xi in x]) * dx

    return lambda x: func(x) / const



def main():

    my_funcs = { 
        'camel_hump': normalize_func(camel_hump, -4,6),
        'dirac': normalize_func(dirac), 
        'dirac': normalize_func(dirac), 
        'resonance': normalize_func(resonance), 
        'exp_poly': normalize_func(exp_poly), 
    }

    func = my_funcs.get('exp_poly')

    for func_name, func in my_funcs.items():

        mcmc = MCMCSampler(func, scale=2)
        samp_func = mcmc.rvs
        pdf_func = func 

        st = SamplerTester(sampler_func=samp_func, pdf_func=pdf_func, 
                n_avg=20, max_n=1000)
        st.plot(suptitle=func_name, save_loc="mcmc_"+func_name+".png") 

def save_frames():
    func = normalize_func(camel_hump)

    mcmc = MCMCSampler(func, scale=2)
    samp_func = mcmc.rvs
    pdf_func = func 

    st = SamplerTester(sampler_func=samp_func, pdf_func=pdf_func, 
            n_avg=20, max_n=10000, n_frames=100)
    st.plot(save_frames=True, frame_path=os.path.join('temp','img_{:03d}.png')) 

if __name__ == '__main__':
    # main()
    save_frames()
