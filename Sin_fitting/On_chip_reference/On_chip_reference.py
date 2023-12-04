import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.signal
import glob as glob
import scipy.optimize as spo
import re
from scipy.signal import find_peaks

def numerical_sort_key(filename):
    # Extract numbers and non-numeric parts using regular expression
    parts = re.findall(r'([-+]?\d+|\D+)', filename)
    
    # Convert numeric parts to integers, leave non-numeric parts as strings
    return [int(part) if part[0].isdigit() or (len(part) > 1 and part[1:].isdigit()) else part for part in parts]

def sort_filenames_numerically(filenames):
    return sorted(filenames, key=numerical_sort_key)

def gaborfunc(t, A, w, p, c, sigma, mu):  
        return ((A * np.sin(w*t + p)) + c) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def fit_gabor(x, y):
    '''Return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    x = np.array(x) # ensures data are numpy arrays
    y = np.array(y)
    
    ff = np.fft.fftfreq(len(x), (x[1]-x[0])) # finds dominant frequency of data - provides initial guess of frequency
    Fy = abs(np.fft.fft(y)) # finds magnitude of FFT
    
    guess_freq = abs(ff[np.argmax(Fy[1:])+1]) # excluding the zero frequency "peak", which is related to offset
    # guess_amp = np.std(y) * 2.**0.5
    guess_amp = 0.5 * (np.max(y) - np.min(y))
#     guess_amp = 1.4826 * np.median(np.abs(y - np.median(y))) # MAD
    guess_offset = np.mean(y)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset, np.std(y), np.mean(y)])

    popt, pcov = spo.curve_fit(gaborfunc, x, y, p0=guess, maxfev = 30000)
    A, w, p, c, sigma, mu = popt
    
    f = w/(2.*np.pi)
    
    fitfunc = lambda t: ( A * np.sin(w*t + p)) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    
    return {"amp": A, "omega": w, "phase": p, 
            "freq": f, "period": 1./f,"offset": c, "fitfunc": fitfunc, 
            "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def calculate_ft(input):
    return  np.fft.fftshift(np.fft.fft2(input))

def calculate_ift(input):
    return np.abs(np.fft.ifft2(input))

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def greyscale(input):
    return np.sum(input.astype('float'), axis=2)

def cross_image(im1, im2):
    return scipy.signal.fftconvolve(im1, im2[::-1,::-1], mode='same')

def pixel_centre(input):
    return np.unravel_index(np.argmax(input), input.shape)

def coords_extraction(image, angle):
    image = rotate_image(image, angle)
    r = cv2.selectROI("select the area", image)
    # cropped_image_ref_1 = image[int(r[1]):int(r[1]+r[3]), 
    #                     int(r[0]):int(r[0]+r[2])]
    cv2.waitKey(0)
    
    return r

def image_crop(image, angle, r):
    image = rotate_image(image, angle)
    # r = cv2.selectROI("select the area", image)
    cropped_image = image[int(r[1]):int(r[1]+r[3]), 
                        int(r[0]):int(r[0]+r[2])]
    # cv2.waitKey(0)
    
    return cropped_image

def image_cleaning(image):
    grey_image = greyscale(image)
    ft_image = calculate_ft(grey_image)
    amplImage = np.log(np.abs(ft_image))
    brights = amplImage < amplThresh
    ft_image[brights] = 0
    amplImage2 = np.abs(ft_image)
    I_fft_abs = calculate_ift(ft_image)
    
    return I_fft_abs

def slice_clean_image(image):
    half_array = int(image.shape[0]/2)
    slice = image[half_array, 0:]
    
    return slice

filepath = 'Sin_fitting/On_chip_reference/BI2_chip/BI2_data'
imagePaths = [f for f in glob.glob(filepath+'*.png')]
files = sort_filenames_numerically(imagePaths)

voltage_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                 -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]

angle = 0
amplThresh = 13
slice_array = []
peak_shift_array = []

filepath = 'Sin_fitting/On_chip_reference/BI2_chip/BI2_data/0v.png'
image = cv2.imread(filepath)

ref_coords = coords_extraction(image, angle)
act_coords = coords_extraction(image, angle)

for i, v in zip(files, voltage_array):
    print(v)
    
    i = cv2.imread(i)

    reference = image_crop(i, angle, ref_coords)
    active = image_crop(i, angle, act_coords)

    clean_reference = image_cleaning(reference)
    active_reference = image_cleaning(active)

    reference_slice = slice_clean_image(clean_reference)
    active_slice = slice_clean_image(active_reference)

    peaks_ref, _ = find_peaks(reference_slice)
    peaks_act, _ = find_peaks(active_slice)
    
    avg_peak_shift = np.mean(np.abs(peaks_ref[0] - peaks_act[0]))
    
    peak_shift_array.append(avg_peak_shift)

    fig, ax = plt.subplots(figsize=[10,7])
    ax.plot(reference_slice, label='Reference', lw=3, color='k')
    ax.plot(active_slice, label='Active', lw=3, color='red')
    ax.legend(loc='upper right', fontsize=16)
    ax.text(0.2, 0.1, f'Peak shift: {avg_peak_shift}', fontsize=20, transform=ax.transAxes, bbox=dict(facecolor='white'))
    plt.savefig(f'Reference_and_Active_Phase_Profiles_{v}.png')
    plt.close()

fig, ax = plt.subplots()
ax.plot(voltage_array, peak_shift_array, 'o', color='k', markersize=10)
ax.set_xlabel('Voltage (V)', fontsize=16)
ax.set_ylabel('Peak shift Difference', fontsize=16)
plt.savefig('Reference_Active_Point_plot.png')

# # x = np.arange(1, len(slice)+1)
# # res = fit_gabor(x, slice)
# # fit = res["fitfunc"](x)
# # period = res["period"]
# # phase = res["phase"]
# # amp = res["amp"]
# # freq = res["freq"]

# plt.subplots(figsize=(10,7))
# ax1 = plt.subplot2grid(shape=(4, 4), loc=(0, 0), rowspan=2, colspan=2)
# ax2 = plt.subplot2grid(shape=(4, 4), loc=(0, 2), rowspan=2, colspan=2)
# ax3 = plt.subplot2grid(shape=(4, 4), loc=(2, 0), colspan=4, rowspan=2)

# ax1.imshow(image)
# ax2.imshow(I_fft_abs)
# ax3.plot(slice - 401, label='Slice', lw=3)
# # ax3.plot(fit, linestyle='--', color='red', label='Fit', lw=3, alpha=0.6)
# ax3.legend(loc='upper right', fontsize=16)
# ax3.set_xlabel('X Pixel', fontsize=16)
# ax3.set_ylabel('Z Pixel', fontsize=16)
# # ax3.tick_params(labelsize=16)

# plt.tight_layout()
# plt.savefig(f'211123/BI2/BI2_Plots/Cleaned_image_{v}V')
# plt.close()
    
# slice_array.append(slice)
    
# fig, ax = plt.subplots()
# for s, v in zip(slice_array,voltage_array):
#     ax.plot(s, label=f'{v}V')
#     ax.legend(loc='upper right', fontsize=12)
#     plt.savefig(f'211123/BI2/BI2_Plots/All_slices.png')
    
# df = pd.DataFrame(slice_array)
# df.to_csv('211123/BI2/BI2_Plots/All_slices.csv')