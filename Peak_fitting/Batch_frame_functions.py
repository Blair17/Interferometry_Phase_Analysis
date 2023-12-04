import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import cv2
import glob as glob
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os
import re

SAVING_FRAMES_PER_SECOND = 2

def greyscale(input):
    return np.sum(input.astype('float'), axis=2)

def calculate_ft(input):
    return  np.fft.fftshift(np.fft.fft2(input))

def calculate_ift(input):
    return np.abs(np.fft.ifft2(input))

def frame_extraction(video_file, voltage):
    frames = main(video_file)

    # image = cv2.imread(frames[0])
    image = greyscale(frames[0])
    r = cv2.selectROI("select the area", frames[0])
    image_ROI = image[int(r[1]):int(r[1]+r[3]), 
                        int(r[0]):int(r[0]+r[2])]
    cv2.waitKey(0)

    # amplThresh = 9

    # data_array = []

    # fig, ax = plt.subplots(2, 1, figsize=[10,7])
    # for i in frames:
    #     # img = cv2.imread(i)
    #     # grey_image = greyscale(img)
    #     cropped = i[int(r[1]):int(r[1]+r[3]), 
    #                 int(r[0]):int(r[0]+r[2])]
        
    #     ft_image = calculate_ft(cropped)
    #     amplImage = np.log(np.abs(ft_image))
    #     brights = amplImage < amplThresh
    #     ft_image[brights] = 0
    #     filtered_image = calculate_ift(ft_image)

    #     height = int(filtered_image.shape[0])
    #     slice = filtered_image[(int(height/2)), 0:]
    #     ax[0].plot(slice)   
    #     peaks, _ = find_peaks(slice[:,0])
        
    #     central_peak = len(peaks)
    #     extraction = peaks[int(central_peak / 2)]
        
    #     data_array.append(extraction)
        
    # sorted = np.sort(data_array)
    # x = np.cumsum(sorted)
    # x1 = x / max(x)

    # array = [0.25, 0.50, 0.75]
    # avg = np.interp(array, x1, sorted)
    # print(avg)

    # ax[1].plot(sorted, x1, 'o')
    # ax[0].set_ylabel('Intentsity (a.u.)')
    # ax[0].set_xlabel('Pixels')
    # ax[1].set_ylabel('Probability Density')
    # ax[1].set_xlabel('Pixels')
    # ax[1].text(0.5, 0.1, f'25%: {np.round(avg[0],2)}, 50%: {np.round(avg[1],2)}, 75%: {np.round(avg[2],2)}', 
    #         transform=ax[1].transAxes, fontsize=16, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

    # plt.axvline(avg[0], ymin=0, ymax=0.25, color='gray', linestyle='--')
    # plt.axvline(avg[1], ymin=0, ymax=0.50, color='gray', linestyle='--')
    # plt.axvline(avg[2], ymin=0, ymax=0.75, color='gray', linestyle='--')

    # plt.axhline(0.25, xmin=0, xmax=avg[0], color='gray', linestyle='--')
    # plt.axhline(0.50, xmin=0, xmax=avg[1], color='gray', linestyle='--')
    # plt.axhline(0.75, xmin=0, xmax=avg[2], color='gray', linestyle='--')

    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(f'{voltage}v_Cumulative_prob_dist.png')
    
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

    plt.savefig(f'Modulation/Sin_fitting/hist_{v}.png')

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
    
    # make a folder by the name of the video file
    # if not os.path.isdir(filename):
        # os.mkdir(filename)
    
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
            
            img = rotate_image(frame, -4)
            
            frames_array.append(img)
            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # increment the frame count
        count += 1
        
    return frames_array

def custom_sort(filename):
    match = re.search(r'(-?\d+)v\.mp4', filename)
    if match:
        return int(match.group(1))
    return 0  # For filenames that don't match the pattern