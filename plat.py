'''
plat.py
A script to compare the effect of platinization on array noise.
(C) 2014 bnaecker@stanford.edu
'''

import numpy as np
import os
from pyret import binary
import matplotlib.pyplot as plt

def check_args(args):
    '''
    Validate command-line inputs.
    '''
    # Check that the first two inputs are existing binary files
    if not (os.path.exists(args[0]) and os.path.exists(args[1])):
        raise FileNotFoundError('The inputs must be *.bin files')

def read_bin_files(files):
    '''
    Read two binary files, and return the data sequences. If the
    sequences are of different lengths (i.e., the before and after
    recordings were of diffferent lengths), the longer is truncated
    to the size of the shorter
    '''
    before, after = binary.readbin(files[0]), binary.readbin(files[1])
    if before.shape == after.shape:
        return before, after

    # Truncate lengths
    if before.shape[0] < after.shape[0]:
        after = after[:before.shape[0], :]
    elif before.shape[0] > after.shape[0]:
        before = before[:after.shape[0], :]

    # Truncate number of channels
    if before.shape[1] < after.shape[1]:
        after = after[:, :before.shape[1]]
    elif before.shape[1] > after.shape[1]:
        before = before[:, :after.shape[1]]
    
    return before, after

def next_pow2(num):
    '''
    Compute the next power of 2, useful for FFTs.
    '''
    return int(2 ** np.ceil(np.log2(num)))

def power_spectrum(data, L=None):
    '''
    Compute the power-spectrum of the data

    Input
    -----

    data (ndarray):
        The data to compute the power spectrum of. The first
        dimension should be the time axis, FFTs are done along
        this axis.

    L (int):
        The length of the FFT to compute. If None (the default),
        the next power of 2 of the size of the data is used.

    Output
    ------

    spec (ndarray):
        The *un-normalized* power spectrum of the data.
    '''

    # Length
    if L is None:
        L = next_pow2(data.shape[0])
    else:
        L = int(L)

    # Compute the FFT
    fft = np.fft.fft(data, n=L, axis=0)

    # Compute the power spectrum
    return np.square(np.abs(fft))

def pad_data(data, N, axis=0):
    '''
    Pad a data array with zeros so that the given axis has size N
    '''

    # Get the size of the data
    n = data.shape[axis]
    if n > N:
        raise ValueError('Requested padded length is less than the data size')

    # Pad
    return np.concatenate((data, np.zeros((N - n, data.shape[-1]))), axis)

def avg_power_spectrum(data, sample_rate=10000, n_segments=64):
    '''
    Estimate the average power spectrum of the given data, using the Welch method. 
    The data is divided into an appropriate number of overlapping segments, whose 
    length and overlap is determined by the number of sample points. Windowed FFTs 
    are computed for each segment, using a Hamming window function, and the power 
    spectrum of each segment is computed and normalized by the total energy of the 
    window. These power spectra are then averaged together and normalized by the 
    number of segments, resulting in an estimate of the full power spectrum of 
    the data.

    Input
    -----

    data (ndarray):
        The data of which to estimate the power spectrum. The first dimension
        should be samples, but the second dimension may be of any size.

    sample_rate (int) [10000]:
        The sample rate of the data. The default is 10K, which is probably
        what you want to use.

    n_segments (int) [64]:
        Number of segments to use.

    Output
    ------

    spec (ndarray):
        The estimated power spectrum of the data. Note that the spectra are *one-sided*

    spec_sem (ndarray):
        The standard error of the estimated power spectrum

    fax (ndarray):
        The frequency axis.
    '''

    # Check dimensions
    if np.ndim(data) > 2:
        raise ValueError('data cannot have more than two dimensions')

    # Compute segment size and amount of overlap
    total_length = data.shape[0]
    spacing = int(total_length / n_segments)    # Sliding FFTs spacing
    segment_length = 3 * spacing                # Size of each segment

    # Pad the data array
    padded_data = pad_data(data, total_length)

    # Get a Hamming window of the appropriate size
    window = np.tile(np.hamming(segment_length), (padded_data.shape[-1], 1)).T
    window_energy = np.sum(np.square(window))

    # Compute power spectrum for each segment
    power = np.zeros((n_segments, segment_length, padded_data.shape[-1]))
    for segment in range(n_segments):

        # Window the data
        begin, end = segment * spacing, min(segment * spacing + segment_length, padded_data.shape[0])
        d = padded_data[begin : end, :] * window[:(end - begin), :]

        # Compute the normalized power spectrum
        power[segment, :, :] = power_spectrum(d, segment_length) / window_energy

    # Compute the average spectrum and standard error
    avg_spectrum = np.mean(power, axis=0) / n_segments
    sem_spectrum = np.std(power, axis=0) / np.sqrt(n_segments)

    # Return *one-sided* spectra
    return np.fft.fftshift(avg_spectrum)[segment_length / 2:], \
            np.fft.fftshift(sem_spectrum)[segment_length / 2:], \
            np.fft.fftshift(np.fft.fftfreq(segment_length, 1 / sample_rate))[segment_length / 2:]

def plot_channels(data, channel=None, sample_rate=10000, max_samples=1000):
    '''
    Plot the actual data
    '''
    if channel is None:
        channel = np.arange(data.shape[1])
    time = np.arange(0, max_samples) * (1 / sample_rate)
    plt.plot(time, np.take(data[:max_samples, :], channel, axis=1))

def plot_spectra(freq, spectra, channel=None, max_freq=100):
    '''
    Plot power spectra.
    '''
    if channel is None:
        channel = np.arange(spectra.shape[-1])
    idx = freq < max_freq
    plt.semilogy(freq[idx], np.take(spectra[idx, :], channel, axis=1))

def comp_spectra(freq, spec1, spec2, channel=None, max_freq=1000):
    '''
    Compare before/after power spectra.
    '''
    if channel is None:
        channel = np.arange(spec1.shape[-1])
    idx = freq < max_freq
    plt.semilogy(freq[idx], np.take(spec1[idx, :], channel, axis=1))
    plt.semilogy(freq[idx], np.take(spec2[idx, :], channel, axis=1))

def run_comparison(beforefile, afterfile, comp_type='full'):
    '''
    Compare the power spectra from the two given files. The first file
    should be the data before platinization and the second is data after.

    `comp_type` is a string identifying which type of comparison to run.
        'full'  - Compare both power spectra and signal RMS (the default)
        'spec'  - Compares power spectra
        'rms'   - Compares RMS of signals before and after
    '''
    # Load the data
    before, after = read_bin_files((beforefile, afterfile))

    # Do the requested comparison
    if comp_type == 'full':

        # Estimate the spectra
        bmean, bstd, freq = avg_power_spectrum(before)
        amean, astd, _ = avg_power_spectrum(after)

        # Compute RMS
        brms = np.std(before, axis=0)
        arms = np.std(after, axis=0)

        # Returns
        return freq, (bmean, bstd), (amean, astd), brms, arms

    elif comp_type == 'spec':

        # Estimate the spectra
        bmean, bstd, freq = avg_power_spectrum(before)
        amean, astd, _ = avg_power_spectrum(after)

        return freq, (bmean, bstd), (amean, astd)

    elif comp_type == 'rms':

        # Compute rms
        brms = np.std(before, axis=0)
        arms = np.std(after, axis=0)

        return brms, arms

if __name__ == '__main__':
    
    # Check input arguments
    filenames = os.sys.argv[1:3]
    check_args(filenames)

    # Run the comparison
    run_comparison(filenames[0], filenames[1], 'full')
