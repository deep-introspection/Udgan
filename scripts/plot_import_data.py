import mne
from udgan.io import convert_mat_structure

data_path = 'data/2010-02-16_0009'

# Events management
raw, events, event_id = convert_mat_structure(data_path)
mne.viz.plot_events(events=events, event_id=event_id,
                    sfreq=raw.info['sfreq'])
event_id = {'Boop': 1, 'Beep': 2, 'Motor response': 6}

# Check raw signals
raw.plot()

# Generate topographic layout
from mne.channels import make_eeg_layout
layout = make_eeg_layout(raw.info, exclude=['MOh','MOb'])

# Generate evoked potentials
tmin, tmax = -0.2, 0.5
include = []
picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=True,
                       include=include, exclude='bads')
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
epochs.plot(trellis=False, title='Auditory odd ball')

# Visualization of neural repsonses for differ
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
import numpy as np
res=16
times = np.arange(-0.2,0.5,0.1)

for event in event_id.keys():
    evoked = epochs[event].average()
    evoked.plot(titles='Evoked potential for ' + event)
    evoked.plot_topomap(times, ch_type='eeg', title='Evoked scalp reponses for ' + event)
