import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.signal
import glob as glob
import scipy.optimize as spo

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

filename = f'image.png'
image = cv2.imread(filename)
image = rotate_image(image, 104)
r = cv2.selectROI("select the area", image)
cropped_image = image[int(r[1]):int(r[1]+r[3]), 
                      int(r[0]):int(r[0]+r[2])]
cv2.waitKey(0)

### Image greyscale ###
grey_image = greyscale(cropped_image)

### Image FT ###
ft_image = calculate_ft(grey_image)
amplImage = np.log(np.abs(ft_image))

# ### Thesholding ###
amplThresh = 13
brights = amplImage < amplThresh
ft_image[brights] = 0
amplImage2 = np.abs(ft_image)

### Inverse FT ###
I_fft_abs = calculate_ift(ft_image)
half_array = int(I_fft_abs.shape[0]/2)
slice = I_fft_abs[half_array, 0:]

x = np.arange(1, len(slice)+1)
res = fit_gabor(x, slice)
fit = res["fitfunc"](x)
period = res["period"]
phase = res["phase"]
amp = res["amp"]
freq = res["freq"]

plt.subplots(figsize=(10,7))

ax1 = plt.subplot2grid(shape=(4, 4), loc=(0, 0), rowspan=2, colspan=2)
ax2 = plt.subplot2grid(shape=(4, 4), loc=(0, 2), rowspan=2, colspan=2)
ax3 = plt.subplot2grid(shape=(4, 4), loc=(2, 0), colspan=4, rowspan=2)
# ax4 = plt.subplot2grid((5, 4), (3, 0), colspan = 2)
# ax5 = plt.subplot2grid((5, 4), (4, 0), colspan = 2)
# ax6 = plt.subplot2grid((5, 4), (2, 2), colspan = 2, rowspan=3)

# fig, ax = plt.subplots(3,1, figsize=(10,9))
ax1.imshow(image)
ax2.imshow(I_fft_abs)
ax3.plot(slice - 401, label='Slice', lw=3)
ax3.plot(fit, linestyle='--', color='red', label='Fit', lw=3, alpha=0.6)
ax3.legend(loc='upper right', fontsize=16)
ax3.set_xlabel('X Pixel', fontsize=16)
ax3.set_ylabel('Z Pixel', fontsize=16)
# ax3.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig(f'test.png')