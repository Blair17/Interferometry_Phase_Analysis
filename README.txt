Python analysis scripts for data produced by a Michelson interferometer. Images or videos of interference fringes are analysed using several different methods, with the ultimate goal of
detecting the phase shift introduced by a phase-modulating device.

### Peak Fitting ###

This code breaks fringe videos down to individual frames for each measurement (voltage) and selects a Region of Interest (ROI) from a sample image. This ROI is applied to every further 
frame for each voltage. From here, the images are cleaned using a Fourier filtering and thresholding technique. Next, slices are taken for each pixel row in each ROI and plotted. The 
position of the central peak of the slice is recorded, with the central peak positions being plotted and a Cumulative Distribution Function (CDF) being applied. The CDF allows the median 
peak position to be found and possesses some benefits to a simple Gaussian/normal distribution and histogram: 

    - When taking a mean, outliers at either extremity of the fit significantly affect the mean value; with a median, this is not the case
    - The resolution is not determined by the bin size

### Sin & Gabor fitting ###

To improve the accuracy, in this code a sin curve is fitted to the data, as opposed to tracking just the central peak. However, the sin fit allows for oscillations in the signal, but not 
localisation. To improve, a Gabor function is next applied. This function is essentially just a sin equation multiplied by an exponential.

### On-chip reference ###

As a differing technique, this code analyses fringe images where the beam covers an active and reference area in a single image. This step allows an on chip reference to normalise the 
phase shifts to.
