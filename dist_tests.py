
from scipy.stats import norm, binom, beta, expon, poisson 
import numpy as np 
import matplotlib.pyplot as plt 


#================================================== 
# show convergence of binomial to normal 
"""

# parameters:
mu = 15
sig = 1
p = 0.1
max_val = 20 

n_sample = 300

# plotting variables  
x = np.linspace(0,max_val,n_sample)
k = [i for i in range(max_val+1)]


plt.ion() 
fig, ax = plt.subplots()

# continuous 
norm_dist, = ax.plot(x, norm.pdf(x,loc=mu, scale=sig), label='norm')

# discrete
binom_dist, = ax.plot(k, binom.pmf(k,max_val,p), label='binom')

for max_n in range(10,200): 
    x = np.linspace(0,max_n,n_sample)
    k = [i for i in range(max_n+1)]

    mu_loc = p * max_n
    sig_loc = np.sqrt( p * (1-p) * max_n ) 

    # continuous 
    norm_dist.set_data( x, norm.pdf(x,loc=mu_loc, scale=sig_loc) )
    binom_dist.set_data(k, binom.pmf(k,max_n,p))

    ax.relim()
    ax.autoscale_view() 
    ax.legend() 
    plt.pause(0.1)   

plt.ioff() 
plt.show() 
"""

"""
#================================================== 
# Show sampling convergence to the distribution 
# the norm stuff is misleading as it ignores negative values..

min_n = 0
max_n = 1000
n_step = 10

mu = 15 


plt.ion() 
fig, ax = plt.subplots(2,1)

# samples 
samples = poisson.rvs(mu, size=max_n)
k = [i for i in range(min(samples), max(samples))]
#x = np.linspace(min(k), max(k), 200)
pmf = poisson.pmf(k,mu)
norm_pdf = norm.pdf(k,loc=mu, scale=np.sqrt(mu))

n_arr = list(range(min_n, max_n+1, n_step)) 
TVD = [np.nan for i in range(len(n_arr))]
RMSE = [np.nan for i in range(len(n_arr))]
r2 = [np.nan for i in range(len(n_arr))]
r2_norm = [np.nan for i in range(len(n_arr))]

for i,n in enumerate(n_arr): 

    # distribution plot 
    ax[0].cla() 
    sample_dist, bin_edges, _ = ax[0].hist(samples[:n], bins=np.arange(min(samples),max(samples)+1)-0.5, density=True)
    poiss_dist, = ax[0].plot(k, pmf, 'r') 
    norm_dist, = ax[0].plot(k,norm_pdf, 'k--') 

    ax[0].set_title(f"n={n}")
    ax[0].set_xlim([0,max(samples)+1])

    # TVD[i] = 0.5 * abs(sample_dist - pmf ).sum()
    # RMSE[i] = np.sqrt( np.mean( (sample_dist - pmf)**2 ) ) 

    # r2 for poisson
    ss_res = np.sum((sample_dist - pmf)**2)
    ss_tot = np.sum((sample_dist - np.mean(sample_dist))**2)
    r2[i] = 1.0 - ss_res / ss_tot

    # r2 for norm
    ss_res = np.sum((sample_dist - norm_pdf)**2)
    ss_tot = np.sum((sample_dist - np.mean(sample_dist))**2)
    r2_norm[i] = 1.0 - ss_res / ss_tot

   

    ax[1].plot(n_arr, r2, 'r', label='poisson R2')
    ax[1].plot(n_arr, r2_norm, 'k--', label='norm R2')

    # ax[1].plot(n_arr, TVD, 'b', label='TVD')
    # ax[1].plot(n_arr, RMSE, 'r', label='RMSE')

    if i==0:
        ax[1].legend()
    ax[1].set_xlim([0,max_n])
    ax[1].set_xlabel('n')
    ax[1].set_ylabel('TVD')
   
    plt.pause(0.05)   

plt.ioff() 
plt.show() 
"""

#================================================== 
# Show sampling convergence to the distribution 
# compare exponential with norm 

# sampling...
n_step = 20  # steps for display 
n_avg = 20 # sets to average 
min_n = n_avg  # not zero 
max_n = 10000

# bounds of the values
mu = 1.3 
nx = 100 
nx_set = 50

plt.ion() 
fig, ax = plt.subplots(2,2, figsize=(8,6))

# samples 
samples = expon.rvs(scale=mu, size=max_n)

# From distribution 
x = np.linspace(min(samples), max(samples), nx)
dx = x[1] - x[0]
bin_edges = x
bin_edges = np.append(bin_edges, x[-1] + dx)
bin_edges -= dx * 0.5 

pdf = expon.pdf(x,scale=mu)
norm_pdf = norm.pdf(x,loc=mu, scale=mu)

# argument of samples 
n_arr = list(range(min_n, max_n+1, n_avg)) 

# central limit
sample_set = [ np.mean(samples[n_arr[i-1]:n_arr[i]]) for i in range(len(n_arr)) ]
sample_set[0] = np.mean(samples[0:n_arr[0]])

x_set = np.linspace(min(sample_set),max(sample_set),nx_set)
dx_set = x_set[1] - x_set[0]
bin_edges_set = x_set
bin_edges_set = np.append(bin_edges_set, x_set[-1] + dx_set)
bin_edges_set -= dx_set * 0.5 

norm_pdf_set = norm.pdf(x_set, loc=mu, scale=mu/np.sqrt(n_avg))

# Loop
n_disp = range(0,len(n_arr),n_step)  # inds of n_arr to display

ns = [n_arr[i] for i in n_disp]
r2 = [np.nan for i in range(len(n_disp))]
r2_norm = [np.nan for i in range(len(n_disp))]

ns_set = [int(n/n_avg) for n in ns]
r2_norm_set = [np.nan for i in range(len(n_disp))]

for i,ind in enumerate(n_disp):
    n = n_arr[ind]

    # --- sample ---

    # distribution plot 
    ax[0,0].cla() 

    sample_dist, _, __ = ax[0,0].hist(samples[:n], bins=bin_edges, density=True)
    expon_dist, = ax[0,0].plot(x, pdf, 'r') 
    norm_dist, = ax[0,0].plot(x, norm_pdf, 'k--') 

    ax[0,0].set_title(f"From distribution (n={n})")
    ax[0,0].set_xlim([0,max(samples)+1])

    # r2 for poisson
    ss_res = np.sum((sample_dist - pdf)**2)
    ss_tot = np.sum((sample_dist - np.mean(sample_dist))**2)
    r2[i] = 1.0 - ss_res / ss_tot

    # r2 for norm
    ss_res = np.sum((sample_dist - norm_pdf)**2)
    ss_tot = np.sum((sample_dist - np.mean(sample_dist))**2)
    r2_norm[i] = 1.0 - ss_res / ss_tot

    ax[1,0].plot(ns, r2, 'r', label='expon R2')
    ax[1,0].plot(ns, r2_norm, 'k--', label='norm R2')
    
    if i==0:
        ax[1,0].legend()
    ax[1,0].set_xlim([0,max_n])
    ax[1,0].set_xlabel('n')
    ax[1,0].set_ylabel('$R^2$')


    # --- sample averaging ---
    ax[0,1].cla() 
    sample_dist_set, _, __ = ax[0,1].hist(sample_set[:ind], bins=bin_edges_set, density=True)
    norm_dist_set, = ax[0,1].plot(x_set, norm_pdf_set, 'k-') 

    ax[0,1].set_title(f"Sample Avg (n_avg={n_avg}, n_sam={int(n/n_avg)})")

    # r2 for norm set
    ss_res = np.sum((sample_dist_set - norm_pdf_set)**2)
    ss_tot = np.sum((sample_dist_set - np.mean(sample_dist_set))**2)
    r2_norm_set[i] = 1.0 - ss_res / ss_tot

    ax[1,1].plot(ns_set, r2_norm_set, 'k-', label='Sample Avg norm R2')
    ax[1,1].set_xlabel('sets of n')

    plt.pause(0.05)   

plt.ioff() 
plt.show() 

