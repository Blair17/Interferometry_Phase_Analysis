import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob as glob
from scipy.optimize import curve_fit
import datetime
import scipy.optimize as spo
import os

date = datetime.datetime.now()

SAVING_FRAMES_PER_SECOND = 200

def greyscale(input):
    return np.sum(input.astype('float'), axis=2)

def calculate_ft(input):
    return  np.fft.fftshift(np.fft.fft2(input))

def calculate_ift(input):
    return np.abs(np.fft.ifft2(input))

def frame_extraction(video_file, voltage):
    frames = main(video_file)

    image = greyscale(frames[0])
    r = cv2.selectROI("select the area", frames[0])
    cropped_image_ref_1 = image[int(r[1]):int(r[1]+r[3]), 
                        int(r[0]):int(r[0]+r[2])]
    cv2.waitKey(0)
    
    return r

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def histogram(data, bins, mean, v):
    "Pass data array, number of bins, mean value of array (avg[0]) and voltage (v)"
    
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    params, covariance = curve_fit(gaussian, bin_centers, hist, p0=[160, mean, 5])
    A, mu, sigma = params
    x = np.linspace(min(bin_centers), max(bin_centers), 1000)

    residuals = hist - gaussian(bin_centers, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((hist-np.mean(hist))**2)
    r_squared = 1 - (ss_res / ss_tot)

    my_string = r'$r^2$'

    fig, ax = plt.subplots()
    ax.hist(data, bins=20, alpha=0.5, label='Histogram')
    ax.plot(x, gaussian(x, *params), 'r-', label='Fitted Gaussian')
    ax.legend(loc='center left')
    ax.text(0.035, 0.9, f'μ: {np.round(mu,2)}, σ: {np.round(sigma,2)}, {my_string}: {np.round(r_squared,2)}', 
                transform=ax.transAxes, fontsize=16, bbox=dict(facecolor='white', 
                edgecolor='gray', alpha=0.8))
    plt.axvline(mu, color='gray', linestyle='--', lw=2)
    plt.axvline(mu-sigma, color='gray', linestyle='--')
    plt.axvline(mu+sigma, color='gray', linestyle='--')

    plt.savefig(f'hist_{v}_{date}.png')

def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def main(video_file):
    frames_array = []
    filename, _ = os.path.splitext(video_file)
    
    # read the video file    
    cap = cv2.VideoCapture(video_file)
    
    # get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'{fps}=')
    
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    
    # start the loop
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        # get the duration by dividing the frame count by the FPS
        frame_duration = count / fps
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            # if closest duration is less than or equals the frame duration, 
            # then save the frame
            # frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            # cv2.imwrite(os.path.join(filename, f"frame{frame_duration_formatted}.jpg"), frame) 
            
            img = rotate_image(frame, -34)
            
            frames_array.append(img)
            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # increment the frame count
        count += 1
        
    return frames_array

def sinfunc(t, A, w, p, c):  
        return A * np.sin(w*t + p) + c

def fit_sin(x, y):
    '''Return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    x = np.array(x) # ensures data are numpy arrays
    y = np.array(y)
    
    ff = np.fft.fftfreq(len(x), (x[1]-x[0])) # finds dominant frequency of data - provides initial guess of frequency
    Fy = abs(np.fft.fft(y)) # finds magnitude of FFT
    
    guess_freq = abs(ff[np.argmax(Fy[1:])+1]) # excluding the zero frequency "peak", which is related to offset
    # guess_amp = np.std(y) * 2.**0.5
    # guess_amp = 0.5 * (np.max(y) - np.min(y))
    guess_amp = 1.4826 * np.median(np.abs(y - np.median(y))) # MAD
    guess_offset = np.mean(y)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    popt, pcov = spo.curve_fit(sinfunc, x, y, p0=guess)
    A, w, p, c = popt
    
    f = w/(2.*np.pi) # calculates frequency in Hertz
    
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    
    return {"amp": A, "omega": w, "phase": p, "offset": c, 
            "freq": f, "period": 1./f, "fitfunc": fitfunc, 
            "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}
    
def coords_extraction(image, angle):
    image = rotate_image(image, angle)
    r = cv2.selectROI("select the area", image)
    cropped_image_ref_1 = image[int(r[1]):int(r[1]+r[3]), 
                        int(r[0]):int(r[0]+r[2])]
    cv2.waitKey(0)
    
    return r