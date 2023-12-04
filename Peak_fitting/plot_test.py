import numpy as np
import matplotlib.pyplot as plt
import os

root = os.getcwd()

datafilename = 'Averages_data_full.csv'
datafilepath = os.path.join(
    root,
    datafilename)
voltage_array, averages25_array, averages50_array, averages75_array = np.genfromtxt(
    fname=datafilepath,
    delimiter=",",
    skip_header=1,
    unpack=True)

# averages_slice1 = [x + 280 for x in averages50_array[0:9]]
# averages_slice2 = averages50_array[9:21]
# new_avg_array = np.concatenate((averages_slice1, averages_slice2))

period = 39.6
diff = (np.diff(averages50_array)) / (period/2)
phase_diff = [((averages50_array[i] - averages50_array[10]) / (period/2) ) 
              for i in range(0, len(averages50_array))]

n = 10

yerr = np.abs(averages50_array - [averages25_array, averages75_array])

fig, ax = plt.subplots()
ax.plot(voltage_array, averages50_array, 'o', color='k')
ax.set_xlabel('Voltage (V)', fontweight='bold')
ax.set_ylabel('Median Peak Position (Pixels)', fontweight='bold')
ax.plot(voltage_array, averages50_array, '--', color='mediumspringgreen')
# ax.fill_between(voltage_array, averages25_array, averages75_array, alpha=0.5, color='m')
ax.errorbar(voltage_array, averages50_array, yerr=yerr, fmt='o', color='k', ecolor='k', capsize=3, capthick=1)

plt.grid()
plt.tight_layout()
plt.savefig('Modulation/test.png')