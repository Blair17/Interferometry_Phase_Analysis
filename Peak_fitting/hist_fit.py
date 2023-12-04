import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

root = os.getcwd()

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

datafilename = 'reaction_time_exp.csv'
datafilepath = os.path.join(
    root,
    datafilename)
data = np.genfromtxt(
    fname=datafilepath,
    delimiter=",",
    skip_header=1,
    unpack=True)

hist, bin_edges = np.histogram(data, bins=40)

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

params, covariance = curve_fit(gaussian, bin_centers, hist, p0=[1, 117, 10])

A, mu, sigma = params

# Print the results
print(f'Mean (μ): {mu} ± {covariance[1,1]}')
print(f'Standard Deviation (σ): {sigma} ± {covariance[2,2]}')

# Plot the histogram and the fitted Gaussian curve
plt.hist(data, bins=50,  alpha=0.5, label='Histogram')
x = np.linspace(min(bin_centers), max(bin_centers), 1000)
plt.plot(x, gaussian(x, *params), 'r-', label='Fitted Gaussian')
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Counts', fontsize=12)

plt.legend()
plt.savefig('hist.png')