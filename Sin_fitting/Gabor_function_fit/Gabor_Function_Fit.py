import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import pandas as pd
import re
from Functions import *
import scipy.optimize as spo
from sklearn.metrics import r2_score

def gaborfunc(t, A, w, p, sigma, mu):  
        return (A * np.sin(w*t + p)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def fit_gabor(x, y):
    '''Return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    x = np.array(x) # ensures data are numpy arrays
    y = np.array(y)
    
    ff = np.fft.fftfreq(len(x), (x[1]-x[0])) # finds dominant frequency of data - provides initial guess of frequency
    Fy = abs(np.fft.fft(y)) # finds magnitude of FFT
    
    guess_freq = abs(ff[np.argmax(Fy[1:])+1]) # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(y) * 2.**0.5
    # guess_amp = 0.5 * (np.max(y) - np.min(y))
#     guess_amp = 1.4826 * np.median(np.abs(y - np.median(y))) # MAD
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., np.std(y), np.mean(y)])

    popt, pcov = spo.curve_fit(gaborfunc, x, y, p0=guess, maxfev = 30000)
    A, w, p, sigma, mu = popt
    
    f = w/(2.*np.pi)
    
    fitfunc = lambda t: ( A * np.sin(w*t + p)) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    
    return {"amp": A, "omega": w, "phase": p, 
            "freq": f, "period": 1./f, "fitfunc": fitfunc, 
            "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

filepath = 'video/'
imagePaths = [f for f in glob.glob(filepath+'*.mp4')]
files = sorted(imagePaths, key=lambda x: int(re.search("(-?\d+)", x).group()))
# Returns mp4 file paths in designated folder and sortes them numerically by voltage

voltage_array = [0]
coords = frame_extraction('video/0v.mp4', '0') # Returns coords for ROI

averages25_array = []
averages50_array = []
averages75_array = []

first = True

for j, v in zip(files, voltage_array):
    frames = main(j)

    amplThresh = 14
    
    data_array = []
    slice_array = []
    fit_array = []
    phase_array = []
    means = []
    r_sqd_array = []
    
    fig, ax = plt.subplots(3, 1, figsize=[10,7])
    for index, k in enumerate(frames):
        cropped = k[int(coords[1]):int(coords[1]+coords[3]), 
                    int(coords[0]):int(coords[0]+coords[2])]
        
        grey_image = greyscale(cropped)
        ft_image = calculate_ft(grey_image)
        amplImage = np.log(np.abs(ft_image))
        brights = amplImage < amplThresh
        ft_image[brights] = 0
        filtered_image = calculate_ift(ft_image)
        
        # figs, axs = plt.subplots(3, 1, figsize=(10,7))
        # axs[0].imshow(grey_image)
        # axs[1].imshow(amplImage)
        # axs[2].imshow(filtered_image)
        # figs.savefig('Image_processing.png')
        # plt.close(figs)
        
        # filtered_array = [ ( int(filtered_image.shape[0] * 0.25) ),
                        #   ( int(filtered_image.shape[0] * 0.50) ),
                        #   ( int(filtered_image.shape[0] * 0.75))]
        
        for i in range(1, filtered_image.shape[0]):
            print(f'{v} V, {i} Slice, {index}/{len(frames)-1} Frames')
            slice = filtered_image[i, 0:]
            ax[0].plot(slice)
            
            x = np.arange(1, len(slice)+1)
            res = fit_gabor(x, slice)
            fit = res["fitfunc"](x)
            period = res["period"]
            phase = res["phase"]
            amp = res["amp"]
            freq = res["freq"]
            
            amp_low = np.mean(slice) - np.std(slice)
            amp_high = np.mean(slice) + np.std(slice)
            
            freq_low = np.mean(freq) - np.std(freq)
            freq_high = np.mean(freq) + np.std(freq)
            
            r_squared = r2_score(slice, fit)
            
            if 0 <= r_squared <= 1:
                r_sqd_array.append(r_squared)
            
            ax[1].plot(fit)
            
            if first:
                extract = phase
                
                first_phase = phase
                first_amp = amp
                first_period = period
                first_freq = freq
                
                fit_array.append(fit)
                phase_array.append(np.abs(extract))
                slice_array.append(slice)
                
                first = False
                
            else:
                extract = phase
                
                if np.abs(extract - first_phase) >= first_phase/2 and np.abs(amp - first_amp) >= first_amp/2 and np.abs(period - first_period) >= first_period/2:
                    extract = first_phase
                    
                elif amp_low < first_amp < amp_high and freq_low < first_freq < freq_high and 0 <= r_squared <= 1:
                    
                    fit_array.append(fit)
                    phase_array.append(np.abs(extract))
                    slice_array.append(slice)

    df = pd.DataFrame(r_sqd_array)
    df.to_csv(f'RSQD_TEST_{v}V.csv', index=False, header=False)
    
    figs, axs = plt.subplots(figsize=(10,7))
    axs.hist(r_sqd_array, bins=100)
    figs.savefig('R2_hist.png')
    plt.close(figs)

    sorted = np.sort(phase_array)
    x = np.cumsum(sorted)
    x1 = x / max(x)

    array = [0.25, 0.50, 0.75]
    avg = np.interp(array, x1, sorted)
    print(avg)

    ax[2].plot(sorted, x1, 'o')
    ax[0].set_ylabel('Intentsity (a.u.)')
    ax[0].set_xlabel('Pixels')
    ax[0].set_title('Frame Slices')
    ax[1].set_ylabel('Intentsity (a.u.)')
    ax[1].set_xlabel('Pixels')
    ax[1].set_title('Sin fits')
    ax[2].text(0.5, 0.1, f'25%: {np.round(avg[0],2)}, 50%: {np.round(avg[1],2)}, 75%: {np.round(avg[2],2)}', 
            transform=ax[2].transAxes, fontsize=16, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    ax[2].set_title('CDF')
    ax[2].set_ylabel('Probability Density')
    ax[2].set_xlabel('Phase (a.u.)')
     
    plt.axvline(avg[0], ymin=0, ymax=0.25, color='gray', linestyle='--')
    plt.axvline(avg[1], ymin=0, ymax=0.50, color='gray', linestyle='--')
    plt.axvline(avg[2], ymin=0, ymax=0.75, color='gray', linestyle='--')

    plt.axhline(0.25, xmin=0, xmax=avg[0], color='gray', linestyle='--')
    plt.axhline(0.50, xmin=0, xmax=avg[1], color='gray', linestyle='--')
    plt.axhline(0.75, xmin=0, xmax=avg[2], color='gray', linestyle='--')

    plt.tight_layout()
    plt.savefig(f'TEST_{v}V_Cumulative_prob_dist.png')
    
    averages25_array.append(avg[0])
    averages50_array.append(avg[1])
    averages75_array.append(avg[2])
    
    fig, ax = plt.subplots()
    ax.plot(means)
    ax.set_xlabel('Frame number', fontweight='bold')
    ax.set_ylabel('Mean intensity (a.u.)', fontweight='bold')
    plt.savefig(f'TEST_{v}V_means.png')