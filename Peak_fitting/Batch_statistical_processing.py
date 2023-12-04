import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import pandas as pd
import datetime
from Batch_frame_functions import *

date = datetime.datetime.now()

filepath = 'Video_filepath/'
imagePaths = [f for f in glob.glob(filepath+'*.mp4')]

files = sorted(imagePaths, key=custom_sort)

voltage_array = [-7, -6, -3, -2, -1, 0, 1, 2, 4, 6, 7]
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
        
        for i in range(0, filtered_image.shape[0]): 
            print(v, i)
            slice = filtered_image[i, 0:]
            # height = filtered_image.shape[0]
            # slice = filtered_image[(int(height/2)), 0:]

            ax[0].plot(slice)
   
            peaks, _ = find_peaks(slice[:,0])
        
            period = np.mean(np.diff(peaks))
        
            if first:
                central_peak = len(peaks)
                extraction = peaks[int(central_peak / 2)]
                old_extraction = extraction
                first = False
                ax[0].axvline(extraction, color='red', linestyle='--')
            else:
                old_extraction = extraction
                extract = np.argmin(np.abs(peaks - extraction))
                extraction = peaks[extract]
            
                if np.abs(extraction - old_extraction) >= period/2:
                    extraction = old_extraction
            
            ax[0].axvline(extraction, color='red', linestyle='--')

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
    
    # histogram(data_array, 20, avg[0], v)


dict_of_arrs = {"V": voltage_array, 
                "25%": averages25_array, 
                "Median": averages50_array,
                "75%": averages75_array}
                
df = pd.DataFrame(dict_of_arrs)
df.to_csv(f'Peak_averages_data_{date}.csv', index=False)

diff = (np.diff(averages50_array)) / (period/2)
phase_diff = [((averages50_array[i] - averages50_array[5]) / (period/2) )
              for i in range(0, len(averages50_array))]

yerr = np.abs(averages50_array - [averages25_array, averages75_array])

fig, ax = plt.subplots()
ax.plot(voltage_array, averages50_array, 'o', color='k')
ax.set_xlabel('Voltage (V)', fontweight='bold')
ax.set_ylabel('Median Peak Position (Pixels)', fontweight='bold')
# ax.fill_between(voltage_array, averages25_array, averages75_array, alpha=0.5, color='mediumspringgreen')
ax.plot(voltage_array, averages50_array, '--', color='mediumspringgreen')
ax.errorbar(voltage_array, averages50_array, yerr=yerr, fmt='o', color='k', ecolor='k', capsize=3, capthick=1)

ax2 = ax.twinx()
ax2.set_ylabel('Phase shift (rad/π)', fontweight='bold')

plt.grid()
plt.tight_layout()
plt.savefig(f'Phase_plot_{date}.png')