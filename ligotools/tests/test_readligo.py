import numpy as np
from ligotools import readligo
import os


#i kept getting path errors so this made it easier"
def get_data_path():
    test_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(test_dir))
    return os.path.join(project_root, "data", "H-H1_LOSC_4_V2-1126259446-32.hdf5")

def test_loaddata_output_structure():
    fn = get_data_path()
    strain, time, chan_dict = readligo.loaddata(fn, 'H1')
    assert isinstance(strain, np.ndarray)
    assert isinstance(time, np.ndarray)
    assert isinstance(chan_dict, dict)

def test_loaddata_strain_not_empty():
    fn = get_data_path()
    strain, time, chan_dict = readligo.loaddata(fn, 'H1')
    assert len(strain) > 0
    assert len(time) == len(strain)

def test_loaddata_time_array_monotonic():
    fn = get_data_path()
    strain, time, chan_dict = readligo.loaddata(fn, 'H1')
    assert np.all(np.diff(time) > 0)
    assert time[0] < time[-1]

def test_loaddata_channel_dict_not_empty():
    fn = get_data_path()
    strain, time, chan_dict = readligo.loaddata(fn, 'H1')
    assert len(chan_dict) > 0
    assert all(isinstance(key, str) for key in chan_dict.keys())