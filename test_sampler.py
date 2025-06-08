"""
    Functions that are used to test a sampler. 
"""
from scipy.stats import expon, norm
import numpy as np 
import matplotlib.pyplot as plt

# Declarations:
N_DISPLAY = 20 
MAX_N = 500
N_BINS = 100

def sample(func, max_n=MAX_N, n_display=N_DISPLAY, n_avg=1):
    """
        Take a set of samples of average n_avg from pdf function

        Argument:
            func - returns an np array of n number of samples 

        Return  
            n_disp (list) - list of number of samples to that point
            samples (np arr) - np array of the samples taken
    """
    n_disp = [round(max_n/n_display)*i for i in range(1,n_display+1)]
    samp = [ np.mean(func(n_avg)) for i in range(n_disp[-1])  ]
    return n_disp, samp 

def get_bin_edges(samples, n_bins=N_BINS):
    x = np.linspace(min(samples), max(samples), n_bins)
    dx = x[1] - x[0]
    bin_edges = x
    bin_edges = np.append(bin_edges, x[-1] + dx)
    bin_edges -= dx * 0.5 

    return bin_edges

def discretize_space(samples):
    bin_edges = get_bin_edges(samples) 
    x = (bin_edges[1:] + bin_edges[0:-1])/2 

    return bin_edges, x 

def update_dist_plot(ax,n,samples,bin_edges,x,pdf, title):
    """
        update the plot with the new number of samples  
    """
    ax.cla() 
    dist,_,__ = ax.hist(samples[:n],bins=bin_edges, density=True)
    pdf_dist = ax.plot(x, pdf, 'r')

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('P(x)')
    
    # r2 error 
    ss_res = np.sum((dist - pdf)**2)
    ss_tot = np.sum((dist - np.mean(dist))**2)
    r2 = 1.0 - ss_res / ss_tot

    return ax, r2

def update_r2_plot(ax, n_disp,r2_arr):
    """
        update the r2 plot  
    """
    ax.plot(n_disp, r2_arr, color='k')

    ax.set_xlabel('n')
    ax.set_ylabel('$R^2$')

    return ax




if __name__ == '__main__':

    mu = 1.2
    n_avg = 10 

    # from distribution 
    n_disp, samples = sample(lambda x: expon.rvs(scale=mu, size=x), n_avg=1)
    bin_edges, x = discretize_space(samples)
    pdf = expon.pdf(x,scale=mu) 
    r2_arr = [np.nan for i in range(len(n_disp))]

    # for set of samples 
    n_disp, samples_set = sample(lambda x: expon.rvs(scale=mu, size=x), n_avg=n_avg)
    bin_edges_set, x_set = discretize_space(samples_set)
    pdf_set = norm.pdf(x_set,loc=mu, scale=mu/np.sqrt(n_avg)) 
    r2_arr_set = [np.nan for i in range(len(n_disp))]


    plt.ion()
    fig, ax = plt.subplots(2,2)

    for i,n in enumerate(n_disp):

        ax[0,0], r2_arr[i] = update_dist_plot(ax[0,0],n,samples,bin_edges,x,pdf, 'Samples')
        ax[1,0] = update_r2_plot(ax[1,0], n_disp, r2_arr)

        ax[0,1], r2_arr_set[i] = update_dist_plot(ax[0,1],n,samples_set,bin_edges_set,x_set,pdf_set, f'Sample Means (n={n_avg})')
        ax[1,1] = update_r2_plot(ax[1,1], n_disp, r2_arr_set)


        plt.tight_layout() 
        plt.pause(0.05)

    plt.ioff() 
    plt.show() 




