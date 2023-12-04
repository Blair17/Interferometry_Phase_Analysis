import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import pandas as pd
import re
import datetime
from Batch_frame_functions import *

date = datetime.datetime.now()

filepath = 'Video_filepath/'
imagePaths = [f for f in glob.glob(filepath+'*.mp4')]

files = sorted(imagePaths, key=lambda x: int(re.search("(-?\d+)", x).group()))

voltage_array = [-10, -9, -8, -7, -6, -5, -4, -3, -2, 
                 -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

coords = frame_extraction('0v.mp4', '0')

averages25_array = []
averages50_array = []
averages75_array = []

first = True

for i, v in zip(files, voltage_array):
    print(i, v)
    frames = main(i)

    amplThresh = 9
    data_array = []

    fig, ax = plt.subplots(2, 1, figsize=[10,7])
    for i in frames:
        cropped = i[int(coords[1]):int(coords[1]+coords[3]), 
                    int(coords[0]):int(coords[0]+coords[2])]
        
        ft_image = calculate_ft(cropped)
        amplImage = np.log(np.abs(ft_image))
        brights = amplImage < amplThresh
        ft_image[brights] = 0
        filtered_image = calculate_ift(ft_image)
        
        height = int(filtered_image.shape[0])
        slice = filtered_image[(int(height/2)), 0:]
        ax[0].plot(slice)   
        peaks, _ = find_peaks(slice[:,0])
        
        central_peak = len(peaks)
        extraction = peaks[int(central_peak / 2)]
        
        period = np.mean(np.diff(peaks))
        
        data_array.append(extraction)
        
    sorted = np.sort(data_array)
    x = np.cumsum(sorted)
    x1 = x / max(x)

    array = [0.25, 0.50, 0.75]
    avg = np.interp(array, x1, sorted)
    print(avg)

    ax[1].plot(sorted, x1, 'o')
    ax[0].set_ylabel('Intentsity (a.u.)')
    ax[0].set_xlabel('Pixels')
    ax[1].set_ylabel('Probability Density')
    ax[1].set_xlabel('Pixels')
    ax[1].text(0.5, 0.1, f'25%: {np.round(avg[0],2)}, 50%: {np.round(avg[1],2)}, 75%: {np.round(avg[2],2)}', 
            transform=ax[1].transAxes, fontsize=16, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

    plt.axvline(avg[0], ymin=0, ymax=0.25, color='gray', linestyle='--')
    plt.axvline(avg[1], ymin=0, ymax=0.50, color='gray', linestyle='--')
    plt.axvline(avg[2], ymin=0, ymax=0.75, color='gray', linestyle='--')

    plt.axhline(0.25, xmin=0, xmax=avg[0], color='gray', linestyle='--')
    plt.axhline(0.50, xmin=0, xmax=avg[1], color='gray', linestyle='--')
    plt.axhline(0.75, xmin=0, xmax=avg[2], color='gray', linestyle='--')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{v}V_Cumulative_prob_dist_{date}.png')
    
    averages25_array.append(avg[0])
    averages50_array.append(avg[1])
    averages75_array.append(avg[2])
    
dict_of_arrs = {"V": voltage_array, 
                "25%": averages25_array, 
                "Median": averages50_array,
                "75%": averages75_array}
                
df = pd.DataFrame(dict_of_arrs)
df.to_csv(f'Peak_averages_data_{date}.csv', index=False)

diff = (np.diff(averages50_array)) / (period/2)
diff1 = np.insert(diff, 10, 0)

# yerr = np.abs(averages50_array - [averages25_array, averages75_array])

fig, ax = plt.subplots(2, 1, figsize=[10,7])
ax[0].plot(voltage_array, averages50_array, 'o')
ax[0].set_xlabel('Voltage (V)', fontweight='bold')
ax[0].set_ylabel('Median Peak Position (pixels)', fontweight='bold')
ax[0].fill_between(voltage_array, averages25_array, averages75_array, alpha=0.5, color='mediumspringgreen')
# ax.errorbar(voltage_array, averages50_array, yerr=yerr, color='mediumspringgreen', fmt='o')
ax[0].grid()

ax[1].plot(voltage_array, diff1, 'o')
ax[1].plot(voltage_array, diff1, '--', color='mediumspringgreen')
ax[1].set_xlabel('Voltage (V)', fontweight='bold')
ax[1].set_ylabel('Phase shift (rad/π)', fontweight='bold')
ax[1].grid()
plt.tight_layout()
plt.savefig(f'Phase_plot_{date}.png')