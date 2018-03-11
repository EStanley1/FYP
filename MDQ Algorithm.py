import struct
import wave

import numpy as np
import scipy as sc
from scipy import signal
import matplotlib.pyplot as plt
from decimal import *

getcontext().prec = 30

input_val = input("Enter fs value:")    # Ask for fs range
f_s = int(input_val)              # Read in the input from user and make sure it's a number
f_n = f_s/2                       # The frequency range that can be detected:

# Time as a list
time = range(0, 600)

# i - different scaled versions of the wavelet function, g, used in the convolution
# Lower scales correspond to higher frequencies and vice versa
# i = [0,1,2,3,4,5,6] results in an octave band zero-phase filter bank
i = [0, 1, 2, 3, 4, 5, 6]

# constant which determines the size of the search region relative to the glottal period.
c = 0.2

# Create Blackman window funciton
#bwf = signal.blackman(100)
#window = signal.blackman(51)
#plt.plot(window)
# plt.show()

# Create an sample waves to test with
# need to increase the number of sampling points
# work out how many needed for 16kHz
impulse = signal.unit_impulse(600, 'mid')
square = signal.square(time)


# Code to plot a wave
def plot_impulse():
    plt.plot(np.arange(0, 600), impulse)
    plt.margins(0.0, 0.01)
    plt.xlabel('Time [samples]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


# Read in wave file - in wav format
def input_wave_file(file_name):
    wave_file = wave.open(file_name, 'r')
    number_frames = wave_file.getnframes()
    for i in range(0, number_frames):
        wave_data = wave_file.readframes(1)
        data = struct.unpack("<h", wave_data)
        print(int(data[0]))


# (1) Create Function for cosine-modulated Gaussian Pulse
# - not -1/+1 but have stepped through it and can't see that there is something wrong
# 11/3/18
def g_of_t(time_i, scale):
    s_i_val = np.power(2, scale)
    time_val = np.true_divide(time_i, s_i_val)
    c_val = np.pi * f_s * time_val
    cos_val = -(np.cos(np.radians(c_val)))

    numerator = np.power(time_val,2)
    tau = np.true_divide(f_n, 2)
    denominator = 2 * (np.power(tau, 2))
    frac = -(np.true_divide(numerator,denominator))
    exp_val = np.exp(frac)

    g_t = cos_val*exp_val
    return g_t


# get cosine-modulated gaussian pulse for the whole time period for a given scale
# return as an array, with which the input signal can be convoluted.
# Last worked on: 11/3/18
def gt_array(i):
    gaussian_pulse = []
    for t in time:
        gt_val = g_of_t(t, i)
        gaussian_pulse.append(gt_val)
    gt_as_array = np.asarray(gaussian_pulse)
    return gt_as_array


# (2) Create Function for the wavelet transform
# tends to zero as t tends to infinity
# x[t] is only non zero for one value of t - how is there meant to be diff vals anywhere else?
# Last worked on: 11/03/18
def yi_of_t(x, g_pulse):
    yi_t = np.convolve(x, g_pulse)
    return yi_t


# (3) Create Function to calculate the mean based signal
def mbs():
    return None


# (4) Create Function to calculate a search interval for each GCI location
def search_int():
    return None


# (5) Create Function to determine the location of maximum amplitude
def max_amp():
    return None


# (6) Create Function to measure distance from maxmima locations to the GCI
def dist():
    return None


# (7) Create Function to calculate maxima dispersion quotient
def mdq():
    return None


# Plot the wavelet transform for the different scales on a graph
# Last worked on: 11/03/28
def plot_wave(wave_array):
    plt.title("Wavelet Transform")
    plt.ylabel("Y")
    plt.xlabel("Time")


    level_0 = wave_array[0:1]
    level_0 = level_0[0]

    level_1 = wave_array[1:2]
    level_1 = level_1[0]
    level_2 = wave_array[2:3]
    level_2 = level_2[0]
    level_3 = wave_array[3:4]
    level_3 = level_3[0]
    level_4 = wave_array[4:5]
    level_4 = level_4[0]
    level_5 = wave_array[5:6]
    level_5 = level_5[0]
    level_6 = wave_array[6:]
    level_6 = level_6[0]

    plt.plot(time, level_0, color="blue", linewidth=1.0, linestyle="-")
    plt.plot(time, level_1, color="green", linewidth=1.0, linestyle="-")
    plt.plot(time, level_2, color="red", linewidth=1.0, linestyle="-")
    plt.plot(time, level_3, color="cyan", linewidth=1.0, linestyle="-")
    plt.plot(time, level_4, color="magenta", linewidth=1.0, linestyle="-")
    plt.plot(time, level_5, color="yellow", linewidth=1.0, linestyle="-")
    plt.plot(time, level_6, color="black", linewidth=1.0, linestyle="-")

    plt.margins(0.0, 0.1)
    plt.xlabel('Time [samples]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    # result = np.arange((len(i)*len(time)), dtype=np.float64).reshape(len(i), len(time))
    result = []
    for j in i:
        temp = gt_array(j)                  # get the gaussian pulse for current scale
        temp = temp[0]
        y_t = np.convolve(impulse, temp)    # use this to convolve with the input wave (impulse)
        result.append(y_t)                  # append result

    result = np.asarray(result)             # cast results as np array

    print(result[0,300])
    print(result[1, 300])
    print(result[2, 300])
    print(result[3, 300])
    print(result[4, 300])
    print(result[5, 300])
    print(result[6, 300])

    plot_wave(result)                       # plot results on graph
