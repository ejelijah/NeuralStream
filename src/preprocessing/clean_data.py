import mne

def preprocess_eeg(raw):
    # 1. Bandpass Filter: Remove low-frequency drift and high-frequency noise
    # We focus on 7Hz - 30Hz where motor imagery signals live
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    
    # 2. Re-referencing: Subtract the average of all electrodes to remove common noise
    raw.set_eeg_reference('average', projection=True)
    
    # 3. Artifact Rejection: (In the notebook, we will use ICA to remove eye blinks)
    return raw

def create_epochs(raw, event_id, tmin=-0.2, tmax=4.0):
    # Segment continuous data into 4-second chunks based on "events" (cues)
    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=(None, 0), preload=True, reject=None)
    return epochs