import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob as glob
from Batch_frame_functions import *
from scipy.signal import find_peaks

filepath = 'Video_filepath/'
imagePaths = [f for f in glob.glob(filepath+'*.png')]

image = cv2.imread(imagePaths[0])
grey_image = greyscale(image)
r = cv2.selectROI("select the area", image)
cropped_image_ref_1 = image[int(r[1]):int(r[1]+r[3]), 
                      int(r[0]):int(r[0]+r[2])]
cv2.waitKey(0)

amplThresh = 9

data_array = []

# fig, ax = plt.subplots(2, 1, figsize=[10,7])
for i in imagePaths:
    img = cv2.imread(i)
    grey_image = greyscale(img)
    cropped = img[int(r[1]):int(r[1]+r[3]), 
                  int(r[0]):int(r[0]+r[2])]
    
    ft_image = calculate_ft(cropped)
    amplImage = np.log(np.abs(ft_image))
    brights = amplImage < amplThresh
    ft_image[brights] = 0
    filtered_image = calculate_ift(ft_image)

    height = int(filtered_image.shape[0])
    slice = filtered_image[(int(height/2)), 0:]
    # ax[0].plot(slice)   
    peaks, _ = find_peaks(slice[:,0])
    
    central_peak = len(peaks)
    extraction = peaks[int(central_peak / 2)]
    
    data_array.append(extraction)
    
sorted = np.sort(data_array)
x = np.cumsum(sorted)
x1 = x / max(x)

array = [0.25, 0.50, 0.75]
avg = np.interp(array, x1, sorted)
print(avg)

fig, ax = plt.subplots()
ax.hist(data_array, bins=10)
plt.show()

# ax[1].plot(sorted, x1, 'o')
# ax[0].set_ylabel('Intentsity (a.u.)')
# ax[0].set_xlabel('Pixels')
# ax[1].set_ylabel('Probability Density')
# ax[1].set_xlabel('Pixels')
# ax[1].text(0.5, 0.1, f'25%: {np.round(avg[0],2)}, 50%: {np.round(avg[1],2)}, 75%: {np.round(avg[2],2)}', 
#            transform=ax[1].transAxes, fontsize=16, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

# plt.axvline(avg[0], ymin=0, ymax=0.25, color='gray', linestyle='--')
# plt.axvline(avg[1], ymin=0, ymax=0.50, color='gray', linestyle='--')
# plt.axvline(avg[2], ymin=0, ymax=0.75, color='gray', linestyle='--')

# plt.axhline(0.25, xmin=0, xmax=avg[0], color='gray', linestyle='--')
# plt.axhline(0.50, xmin=0, xmax=avg[1], color='gray', linestyle='--')
# plt.axhline(0.75, xmin=0, xmax=avg[2], color='gray', linestyle='--')

# plt.tight_layout()
# # plt.show()
# plt.savefig('0v_Cumulative_prob_dist.png')
    
