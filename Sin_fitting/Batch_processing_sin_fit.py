import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import pandas as pd
import re
from Functions import *
import datetime

date = datetime.datetime.now()

filepath = 'Video_filepath/'
imagePaths = [f for f in glob.glob(filepath+'*.mp4')]
files = sorted(imagePaths, key=lambda x: int(re.search("(-?\d+)", x).group()))
# Returns mp4 file paths in designated folder and sortes them numerically by voltage

voltage_array = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 
                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
coords = frame_extraction('0v.mp4', '0') # Returns coords for ROI

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
    
    fig, ax = plt.subplots(3, 1, figsize=[10,7])
    for k in frames:
        cropped = k[int(coords[1]):int(coords[1]+coords[3]), 
                    int(coords[0]):int(coords[0]+coords[2])]
        
        grey_image = greyscale(cropped)
        ft_image = calculate_ft(grey_image)
        amplImage = np.log(np.abs(ft_image))
        brights = amplImage < amplThresh
        ft_image[brights] = 0
        filtered_image = calculate_ift(ft_image)
        
        figs, axs = plt.subplots()
        axs.imshow(filtered_image)
        figs.savefig('filtered_image.png')
        plt.close(figs)
        
        for i in range(0, filtered_image.shape[0]): 
            print(v, i)
            slice = filtered_image[i, 0:]
            ax[0].plot(slice)

            x = np.arange(1, len(slice)+1)
            res = fit_sin(x, slice)
            fit = res["fitfunc"](x)
            period = res["period"]
            phase = res["phase"]
            # phase = np.abs(phase)
            ax[1].plot(fit)
            
            if first:
                extract = phase
                old_phase = phase
                first = False
            else:
                extract = phase
                
                if np.abs(extract - old_phase) >= period/4:
                    extract = old_phase
                else:
                    old_phase = phase
        
            slice_array.append(slice)
            fit_array.append(fit)
            phase_array.append(extract)
        
    sorted = np.sort(phase_array)
    x = np.cumsum(sorted)
    x1 = x / max(x)

    array = [0.25, 0.50, 0.75]
    avg = np.interp(array, x1, sorted)
    print(avg)

    ax[2].plot(sorted, x1, 'o')
    ax[0].set_ylabel('Intentsity (a.u.)')
    ax[0].set_xlabel('Pixels')
    ax[1].set_ylabel('Probability Density')
    ax[1].set_xlabel('Pixels')
    ax[2].text(0.5, 0.1, f'25%: {np.round(avg[0],2)}, 50%: {np.round(avg[1],2)}, 75%: {np.round(avg[2],2)}', 
            transform=ax[2].transAxes, fontsize=16, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

    plt.axvline(avg[0], ymin=0, ymax=0.25, color='gray', linestyle='--')
    plt.axvline(avg[1], ymin=0, ymax=0.50, color='gray', linestyle='--')
    plt.axvline(avg[2], ymin=0, ymax=0.75, color='gray', linestyle='--')

    plt.axhline(0.25, xmin=0, xmax=avg[0], color='gray', linestyle='--')
    plt.axhline(0.50, xmin=0, xmax=avg[1], color='gray', linestyle='--')
    plt.axhline(0.75, xmin=0, xmax=avg[2], color='gray', linestyle='--')

    plt.tight_layout()
    plt.savefig(f'{v}V_Cumulative_prob_dist_{date}.png')
    
    averages25_array.append(avg[0])
    averages50_array.append(avg[1])
    averages75_array.append(avg[2])
    
    # histogram(data_array, 20, avg[1], v)

dict_of_arrs = {"V": voltage_array, 
                "25%": averages25_array, 
                "Median": averages50_array,
                "75%": averages75_array}
                
df = pd.DataFrame(dict_of_arrs)
df.to_csv(f'Averages_data_{date}.csv', index=False)

diff = (np.diff(averages50_array)) / (period/2)
phase_diff = [((averages50_array[i] - averages50_array[10]) / (period/2) )
              for i in range(0, len(averages50_array))]

fig, ax = plt.subplots()
ax.plot(voltage_array, averages50_array, 'o', color='k')
ax.set_xlabel('Voltage (V)', fontweight='bold')
ax.set_ylabel('Sin fit phase shift', fontweight='bold')
# ax.fill_between(voltage_array, averages25_array, averages75_array, alpha=0.5, color='mediumspringgreen')

ax2 = ax.twinx()
ax2.plot(voltage_array, phase_diff, '--', color='mediumspringgreen')
ax2.set_ylabel('Phase shift (rad/π)', fontweight='bold')

plt.grid()
plt.tight_layout()
plt.savefig(f'Phase_plot_{date}.png')