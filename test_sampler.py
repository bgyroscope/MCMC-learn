""" Functions that are used to test a sampler. 

    _2 indicates rather than the sample from distribution itself, 
    the average of a set n_avg samples 
    
"""
from scipy.stats import expon, norm
import numpy as np 
import matplotlib.pyplot as plt

# Declarations:
N_SAMPLE = 10
N_FRAMES = 20 
MAX_N = 1000
N_BINS = 100

class SamplerTester:

    def __init__(self, 
            sampler_func,
            pdf_func, 
            n_avg=N_SAMPLE,
            max_n = MAX_N, 
            n_bins = N_BINS,
            n_frames = N_FRAMES
            ):
        """
            Class to test the sampler and pdf functions

            Args:
                sampler_func (function): returns a sample from the distribution   
                pdf_func (function): returns P(x), expecting pdf_func(x)   
                n_avg (int): sample size 
                max_n (int): max number of samples 
                n_bins (int): number of bins to display data in 
                n_frames (int): number of animation frames 

            Returns:
                None 
        """
        self.sampler = sampler_func
        self.pdf = pdf_func
        self.n_avg = n_avg
        self.n_bins = n_bins

        # get the display numbers 
        self.n_disp = [round(max_n/n_frames)*i for i in range(1,n_frames+1)]
        self.max_n = self.n_disp[-1]

        # get the samples 
        self.samp = [ self.sampler() for i in range(self.max_n)]
        self.samp_2 = [ np.mean([self.sampler() for i in range(self.n_avg)]) for i in range(self.max_n) ]

        # get the bins
        self.bins, self.x = self.discretize_space(self.samp) 
        self.bins_2, self.x_2 = self.discretize_space(self.samp_2) 

        # pdfs
        self.pdf = [pdf_func(x_i) for x_i in self.x]

        self.mu = np.mean(self.samp)
        self.sig = np.std(self.samp)
        self.pdf_2 = norm.pdf(self.x_2,loc=self.mu,scale=self.sig/np.sqrt(n_avg))

        # set up r2 arrays 
        self.r2_arr = [np.nan for i in range(len(self.n_disp))]
        self.r2_arr_2 = [np.nan for i in range(len(self.n_disp))]

    def __str__(self):
        return f"Testing distribution with mu={self.mu}, sig={self.sig}, for n={self.n_avg} "

    # get the bin edges and x 
    def get_bin_edges(self,samples):
        x = np.linspace(min(samples), max(samples), self.n_bins)
        dx = x[1] - x[0]
        bin_edges = x
        bin_edges = np.append(bin_edges, x[-1] + dx)
        bin_edges -= dx * 0.5 

        return bin_edges

    def discretize_space(self,samples):
        bin_edges = self.get_bin_edges(samples) 
        x = (bin_edges[1:] + bin_edges[0:-1])/2 

        return bin_edges, x 

    def init_plot(self):
        self.fig, self.ax = plt.subplots(2,2, figsize=(8,6))  

        # draw dummy axes for all 
        # Distribution 
        dist,edges,self.patches = self.ax[0,0].hist([],bins=self.bins,density=True)
        self.pdf_line, = self.ax[0,0].plot(self.x,self.pdf,'r') 

        self.ax[0,0].set_title('Distribution')
        self.ax[0,0].set_xlabel('x')
        self.ax[0,0].set_ylabel('P(x)')

        # Distribution r2
        self.r2_line, = self.ax[1,0].plot(self.n_disp, self.r2_arr, 'k')

        self.ax[1,0].set_xlabel('n')
        self.ax[1,0].set_ylabel('$R^2$')
        self.ax[1,0].set_xlim(0,self.max_n)
        self.ax[1,0].set_ylim(0,1)

        # Sample Average Distribution 
        dist,edges,self.patches_2 = self.ax[0,1].hist([],bins=self.bins_2,density=True)
        self.pdf_line_2, = self.ax[0,1].plot(self.x_2,self.pdf_2,'r') 

        self.ax[0,1].set_title(f'Distribution of Sample Averages ({self.n_avg})')
        self.ax[0,1].set_xlabel('x')

        # Sample Average Distribution r2
        self.r2_line_2, = self.ax[1,1].plot(self.n_disp, self.r2_arr_2, 'k')

        self.ax[1,1].set_xlabel('n sample averages')
        self.ax[1,1].set_xlim(0,self.max_n)
        self.ax[1,1].set_ylim(0,1)

        plt.tight_layout() 

    def update_plot(self,i,n):

        # Distribution 
        dist, _ = np.histogram(self.samp[:n],bins=self.bins,density=True)
        for h, patch in zip(dist, self.patches):
            patch.set_height(h)
        self.ax[0,0].set_ylim(0,max(dist)*1.1)

        # Distribution r2
        ss_res = np.sum((dist - self.pdf)**2)
        ss_tot = np.sum((dist - np.mean(dist))**2)
        self.r2_arr[i] = 1.0 - ss_res / ss_tot
        self.r2_line.set_data(self.n_disp, self.r2_arr)

        # Sample Average Distribution 
        dist, _ = np.histogram(self.samp_2[:n],bins=self.bins_2,density=True)
        for h, patch in zip(dist, self.patches_2):
            patch.set_height(h)
        self.ax[0,1].set_ylim(0,max(dist)*1.1)

        # Distribution r2
        ss_res = np.sum((dist - self.pdf_2)**2)
        ss_tot = np.sum((dist - np.mean(dist))**2)
        self.r2_arr_2[i] = 1.0 - ss_res / ss_tot
        self.r2_line_2.set_data(self.n_disp, self.r2_arr_2)

        self.fig.canvas.draw() 
        self.fig.canvas.flush_events() 


    def plot(self):
        plt.ion() 
        self.init_plot() 

        for i,n in enumerate(self.n_disp):
            self.update_plot(i,n) 
            plt.pause(0.05)

        plt.ioff() 
        plt.show() 

if __name__ == '__main__':
    
    # --- exponential ---
    mu = 2
    samp_func = lambda: expon.rvs(scale=mu,size=1)
    pdf_func = lambda x: expon.pdf(x,scale=mu) 


    # # # --- normal ---
    # mu = 20
    # sig = 2
    # samp_func = lambda: norm.rvs(loc=mu, scale=sig,size=1)
    # pdf_func = lambda x: norm.pdf(x,loc=mu, scale=sig) 

    st = SamplerTester(samp_func,pdf_func)

    print('Plotting the results.')
    st.plot() 


