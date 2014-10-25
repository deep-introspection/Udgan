from scipy.io import loadmat
import os
import os.path as op
import mne
import numpy as np


def convert_mat_structure(data_path):
    """Read and convert matfiles from Udgan
    Parameters
    ----------
    data_path : str
        The absolute path to the directory harboring the data.

    Returns
    -------
    raw : instance of mne.io.Raw
        The raw data
    events : np.ndarray of int, shape(n_events, 3) | None
        The events that are parsed from Triggers.mat if it is present.
        Else None.
    event_id : dict | None
        The event ids hat are parsed from Triggers.mat if it is present.
        Else None.
    """
    fnames = os.listdir(data_path)
    if 'EEG.mat' in fnames:
        eeg_fname = fnames[fnames.index('EEG.mat')]
    else:
        raise ValueError('Data path does not include EEG data')

    fname = op.join(data_path, eeg_fname)
    mat = loadmat(fname)
    ch_names = mat['H']['sensors'][0][0][0]['name'][0]
    ch_names = [c[0][0] for c in ch_names]
    ch_types = mat['H']['sensors'][0][0][0]['category'][0]
    ch_types = [c[0][0].replace('occular', 'eog') for c in ch_types]
    sfreq = float(mat['H']['sampleRate'][0][0][0][0])

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq,
                           ch_types=ch_types)

    pos = np.array([x[0][0][0] for x in
                   mat['H']['sensors'][0][0][0]['coils'][0][2:]])
    eeg_idx = [i for i, e in enumerate(ch_types) if e == 'eeg']
    assert len(eeg_idx) == len(pos)

    chs = info['chs']
    for ii, (ch, pos_) in enumerate(zip(chs, pos)):
        if ii not in eeg_idx:
            continue
        ch['loc'] = np.array([0, 0, 0, 1] * 3, dtype='f4')
        ch['loc'][:3] = pos_[:]

    dtype = [('channel', 'O'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
    pos_dig = np.loadtxt('data/Udgan_190210.pol', dtype=dtype)

    fiducials = np.concatenate([pos_dig[:3], pos_dig[-7:]])
    fiducials = np.array([fiducials['x'], fiducials['y'],
                          fiducials['z']]).T
    fiducials = fiducials[:-1]

    fid = [{'coord_frame': 4, 'r': fiducials[i::3].mean(0),
            'kind': 1, 'ident': i + 1} for i in [0, 1, 2]]

    info['dig'] = fid

    events = None
    event_ids = None
    data = mat['F']
    if 'Triggers.mat' in fnames:
        trig_fname = fnames[fnames.index('Triggers.mat')]
        triggers = loadmat(op.join(data_path, trig_fname))
        event_names = [c[0][0] for c in
                       triggers['Events'][0]['data'][0][0]['name']]
        event_ids = [ord(c.lower()) - 96 for c in event_names]
        times = [c[0][0] for c in
                 triggers['Events'][0]['data'][0][0]['time']]
        events = np.zeros((len(times), 3), dtype=np.int)
        events[:, 0] = np.array(times) * sfreq
        events[:, 2] = event_ids
        event_ids = {k: ord(k.lower()) - 96 for k in set(event_names)}
    else:
        print('Data path does not include Triggers data')
    raw = mne.io.RawArray(data, info)
    return raw, events, event_ids
