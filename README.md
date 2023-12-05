# Interferogram phase analysis
Python analysis scripts for data produced by a Michelson interferometer. Images or videos of interference fringes are analysed using several different methods, with the ultimate goal of
detecting the phase shift introduced by a phase-modulating device.

## Features
The code is sectioned into two different methods of analysis; one method focuses on tracking a central peak, whilst the other fits a sin (and Gabor) function. In general, the procedure for both areas is as follows:

* Data read in
    * Read in video or image data for each applied voltage
    * If video data is used, the video is divded into frames using the corresponding FPS
* Fourier filtering
    * A region of interest (ROI) is manually selected and then applied to all other frames/voltages
    * The ROI is next fourier transformed and a threshold is set. All pixels with a brightness below this threshold are omitted from the image, which is then inverse fourier transformed to produce a clean image
* Data extraction
    * Each ROI from each frame/image is next sliced by each pixel row, with either a peak fit or sin/Gabor fit applied
    * This allows the phase or position of this fringe pattern to be retrieved
    * This phase value is thus compared to the '0 V' condition, so a phase shift can be determined

### Peak Fitting 
The position of the central peak of the slice is recorded, with the central peak positions being plotted and a Cumulative Distribution Function (CDF) being applied. The CDF allows the median 
peak position to be found and possesses some benefits to a simple Gaussian/normal distribution and histogram: 

    - When taking a mean, outliers at either extremity of the fit significantly affect the mean value; with a median, this is not the case
    - The resolution is not determined by the bin size

### Sin & Gabor fitting 
To improve the accuracy, in this code a sin curve is fitted to the data, as opposed to tracking just the central peak. However, the sin fit allows for oscillations in the signal, but not 
localisation. To improve, a Gabor function is next applied. This function is essentially just a sin equation multiplied by an exponential.

### On-chip reference 

As a differing technique, this code analyses fringe images where the beam covers an active and reference area in a single image. This step allows an on chip reference to normalise the 
phase shifts to.

## Notes
This repository is essentially a filestore for all current code that is in development for this type of analysis. 