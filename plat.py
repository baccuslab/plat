'''
plat.py
A script to compare the effect of platinization on array noise.
(C) 2014 bnaecker@stanford.edu
'''

import numpy as np
import os
import matplotlib.pyplot as plt
from pyret import binary

def _check_args(args):
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

    Input
    -----

    files (iterable of strings):
        The filenames to read in. Only first 2 are used.

    Output
    ------
    
    before, after (ndarray):
        The recorded data from the two files.
    '''
    before, after = binary.readbin(files[0]), binary.readbin(files[1])
    if before.shape == after.shape:
        return before, after

    # Truncate arrays to the size of the smallest
    mins = np.min(np.vstack((before.shape, after.shape)), axis=0)
    before[mins[0]:, mins[1]:] = []
    after[mins[0]:, mins[1]:] = []
    
    return before, after

def _next_pow2(num):
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
    if not L:
        L = _next_pow2(data.shape[0])
    else:
        L = int(L)

    # Compute the FFT
    fft = np.fft.fft(data, n=L, axis=0)

    # Compute the power spectrum
    return np.square(np.abs(fft))

def _pad_data(data, N, axis=0):
    '''
    Pad a data array with zeros so that the given axis has size N
    '''

    # Get the size of the data
    n = data.shape[axis]
    if n > N:
        raise ValueError('Requested padded length is less than the data size')

    # Pad
    return np.concatenate((data, np.zeros((N - n, data.shape[-1]))), axis)

def avg_power_spectrum(data, do_sem=False, sample_rate=10000, n_segments=64):
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

    do_sem (Boolean) [False]:
        Compute the SEM of the average power spectrum. This defaults to False,
        as this computation can be *very* expensive for long recordings.

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
    padded_data = _pad_data(data, np.ceil(total_length / segment_length) * segment_length)

    # Get a Hamming window of the appropriate size
    window = np.tile(np.hamming(segment_length), (padded_data.shape[-1], 1)).T
    window_energy = np.sum(np.square(window))

    # Compute power spectrum for each segment
    power = np.zeros((n_segments, segment_length, padded_data.shape[-1]))
    for segment in range(n_segments):

        # Window the data
        print('segment {0:d} of {1:d}'.format(segment + 1, n_segments))
        begin, end = segment * spacing, min(segment * spacing + segment_length, padded_data.shape[0])
        d = padded_data[begin : end, :] * window[:(end - begin), :]

        # Compute the normalized power spectrum
        power[segment, :, :] = power_spectrum(d, segment_length) / window_energy

    # Compute the average spectrum and standard error, one-sided
    avg_spectrum = np.fft.fftshift(np.mean(power, axis=0) / n_segments)[segment_length / 2:]
    if do_sem:
        sem_spectrum = np.fft.fftshift(np.std(power, axis=0) / np.sqrt(n_segments))[segment_length / 2:]
    else:
        sem_spectrum = None

    # Compute return frequency values
    freq = np.fft.fftshift(np.fft.fftfreq(segment_length, 1 / sample_rate))[segment_length / 2:]

    # Return 
    return avg_spectrum, sem_spectrum, freq

def plot_channels(data, channel=None, sample_rate=10000, max_samples=1000):
    '''
    Plot the actual data.

    Input
    -----

    data (ndarray):
        True binary recording data.

    channel (iterable of ints) [None]:
        Channels to plot. If None (the default) plot all channels.

    sample_rate (float) [10000]:
        Sample rate of the data.

    max_samples (int) [1000]:
        Number of samples to plot.

    Output
    ------

    fig (Matplotlib.Figure):
        The figure into which the data are plotted.
    '''

    # Find channels to plot
    if channel is None:
        channel = np.arange(data.shape[1])

    # Get the time axis
    time = np.arange(0, max_samples) * (1 / sample_rate)

    # Plot data
    fig = plt.figure()
    plt.plot(time, np.take(data[:max_samples, :], channel, axis=1))

    return fig

def plot_spectra(freq, spectra, channel=None, bw=(0, 2000)):
    '''
    Plot power spectra.

    Input
    -----

    freq (ndarray):
        The frequency axis of the power spectra.

    spectra (ndarray):
        The spectra to plot.

    channel (iterable of ints) [None]:
        Channels to plot. If None (the default) plot all channels.

    bw (tuple of floats) [(0, 2000)]:
        The bandwith to plot, low freq to high freq.

    Output
    ------

    fig (Matplotlib.Figure):
        The figure into which the power spectra are plotted
    '''

    # Find channels to plot
    if channel is None:
        channel = np.arange(spectra.shape[-1])

    # Get frequencies to plot
    idx = (freq >= bw[0]) & (freq <= bw[1])

    # Plot spectra
    fig = plt.figure()
    plt.semilogy(freq[idx], np.take(spectra[idx, :], channel, axis=1))

    return fig

def comp_spectra(freq, spec1, spec2, channel=None, bw=(0, 2000)):
    '''
    Compare before/after power spectra.

    Input
    -----

    freq (ndarray):
        Frequency axis of the power spectra.

    spec1, spec2 (ndarray):
        The two power spectra to compare.

    channel (iterable of ints) [None]:
        The channels to plot. If None (the default) plot all channels.

    bw (tuple of floats) [(0, 2000)]:
        The bandwidth to plot, low freq to high freq.

    Output
    ------

    fig (Matplotlib.Figure):
        THe figure into which the power spectra are plotted
    '''

    # Find channels to plot
    if channel is None:
        channel = np.arange(spec1.shape[-1])

    # Get frequencies to plot
    idx = (freq >= bw[0]) & (freq <= bw[1])

    # Plot two power spectra
    fig = plt.figure()
    plt.semilogy(freq[idx], np.take(spec1[idx, :], channel, axis=1))
    plt.semilogy(freq[idx], np.take(spec2[idx, :], channel, axis=1))

    return fig

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
    print('loading data ... ', end='', flush=True)
    before, after = read_bin_files((beforefile, afterfile))
    print('done.')

    # Do the requested comparison
    print('running spectrum comparison ... ', flush=True)
    if comp_type == 'full':

        # Estimate the spectra
        print('\nbefore platinization\n--------------------')
        bmean, bstd, freq = avg_power_spectrum(before)
        print('\nafter platinization\n-------------------')
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

def print_all(fnames, freq, spec1, spec2, rms1, rms2, bw=(0,1600)):
    '''
    Print all power spectra to PDF files
    '''
    # Construct path for saving figures and notify
    base_dir = os.path.join(os.getcwd(), 'figures')
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    print('\nsaving figures to {s} ... '.format(s=base_dir), flush=True)

    # Plot and save figures for each channel's spectrum
    on = plt.isinteractive()
    plt.ioff()
    for chan in range(spec1.shape[-1]):
        fig = comp_spectra(freq, spec1, spec2, channel=chan, bw=bw)
        plt.title('Channel {0:d}'.format(chan))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.legend(('Before', 'After'))
        print('figure {:02d}'.format(chan), flush=True)
        plt.savefig(os.path.join(base_dir, 'channel{0:02d}.png'.format(chan)), format='png')
        plt.close(fig)

    # Plot and save figure showing RMS ratio
    fig = plt.figure()
    plt.plot(rms2 / rms1, 'o')
    plt.title('RMS ratio (after / before)')
    plt.xlabel('Channel')
    plt.savefig(os.path.join(base_dir, 'rms_ratio.png'), format='png')
    plt.close(fig)

    # Notify
    plt.interactive(on)
    print('done.')

if __name__ == '__main__':
    
    # Check input arguments
    filenames = os.sys.argv[1:3]
    _check_args(filenames)

    # Run the comparison
    freq, before, after, brms, arms = run_comparison(filenames[0], filenames[1], 'full')

    # Print all comparisons to PDF
    print_all(filenames, freq, before[0], after[0], brms, arms, bw=(0, 1600))
    os.sys.exit(0)
