import mne
from udgan.io import convert_mat_structure

data_path = 'data/2010-02-16_0009'

raw, events, event_id = convert_mat_structure(data_path)

mne.viz.plot_events(events=events, event_id=event_id,
                    sfreq=raw.info['sfreq'])

raw.plot()
