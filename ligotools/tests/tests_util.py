import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
import tempfile
import os

def test_write_wavfile_basic():
    fs = 4096
    data = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        filename = tmp.name
    
    try:
        write_wavfile(filename, fs, data)
        read_fs, read_data = wavfile.read(filename)
        
        assert read_fs == fs
        assert len(read_data) == len(data)
        assert read_data.dtype == np.int16
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_write_wavfile_scaling():
    fs = 8000
    data = np.array([1.0, -1.0])
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        filename = tmp.name
    
    try:
        write_wavfile(filename, fs, data)
        _, read_data = wavfile.read(filename)
        
        assert np.max(np.abs(read_data)) > 25000
        assert np.max(np.abs(read_data)) < 32767
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_whiten_basic():
    dt = 1.0/4096
    Nt = 4096
    strain = np.random.randn(Nt)
    
    freqs = np.fft.rfftfreq(Nt, dt)
    interp_psd = interp1d(freqs, np.ones_like(freqs))
    
    white_ht = whiten(strain, interp_psd, dt)
    
    assert len(white_ht) == Nt
    assert not np.any(np.isnan(white_ht))


def test_whiten_zeros():
    dt = 1.0/4096
    Nt = 1024
    strain = np.zeros(Nt)
    
    freqs = np.fft.rfftfreq(Nt, dt)
    interp_psd = interp1d(freqs, np.ones_like(freqs))
    
    white_ht = whiten(strain, interp_psd, dt)
    
    assert len(white_ht) == Nt
    assert np.allclose(white_ht, 0)


def test_reqshift_basic():
    sample_rate = 4096
    data = np.random.randn(4096)
    
    shifted = reqshift(data, fshift=100, sample_rate=sample_rate)
    
    assert len(shifted) == len(data)
    assert not np.any(np.isnan(shifted))


def test_reqshift_sine():
    sample_rate = 4096
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    data = np.sin(2 * np.pi * 100 * t)
    
    shifted = reqshift(data, fshift=50, sample_rate=sample_rate)
    
    assert len(shifted) == len(data)
    assert np.max(np.abs(shifted)) > 0.5