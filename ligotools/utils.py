import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, windows
from scipy.io import wavfile
from matplotlib import mlab
from scipy.interpolate import interp1d

def write_wavfile(filename, fs, data):
    d = np.int16(data / np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename, int(fs), d)

def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    norm = 1./np.sqrt(1./(dt*2))
    hf = np.fft.rfft(strain)
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

def reqshift(data, fshift=100, sample_rate=4096):
    x = np.fft.rfft(data)
    T = len(data)/float(sample_rate)
    df = 1.0/T
    nbins = int(fshift/df)
    y = np.roll(x.real, nbins) + 1j*np.roll(x.imag, nbins)
    y[0:nbins] = 0.
    z = np.fft.irfft(y)
    return z