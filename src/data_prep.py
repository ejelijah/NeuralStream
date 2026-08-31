import mne
import numpy as np
from moabb.datasets import BNCI2014_001
from sklearn.preprocessing import StandardScaler

def scale_data(epochs):
    """
    Helper function to scale EEG data. 
    Scaling ensures the model learns patterns, not just signal volume.
    """
    data = epochs.get_data() # [Trials, Channels, Samples]
    for i in range(data.shape[0]):
        scaler = StandardScaler()
        # Scale across the time dimension for each channel
        data[i] = scaler.fit_transform(data[i].T).T
    epochs._data = data
    return epochs

def get_cleaned_epochs(subject_id=1, training=True):
    """
    Fetches and cleans EEG data for a single subject.
    """
    dataset = BNCI2014_001()
    sessions = dataset.get_data(subjects=[subject_id])
    session_key = '0train' if training else '1test'
    
    raw = sessions[subject_id][session_key]['0']
    raw.load_data()
    
    raw.filter(7., 30., fir_design='firwin', verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=4.0, 
                        baseline=(None, 0), preload=True, verbose=False)
    
    # Apply Scaling
    epochs = scale_data(epochs)
    return epochs, event_id

def get_multi_subject_data(subject_list=[1, 2, 3], training=True):
    """
    Fetches and combines cleaned, scaled data from multiple subjects.
    """
    all_epochs = []
    event_id = None
    
    for sub_id in subject_list:
        print(f"Loading and Scaling Subject {sub_id}...")
        epochs, eid = get_cleaned_epochs(subject_id=sub_id, training=training)
        all_epochs.append(epochs)
        event_id = eid # Keep track of the last event_id
    
    combined_epochs = mne.concatenate_epochs(all_epochs)
    print(f"Successfully combined {len(subject_list)} subjects.")
    return combined_epochs, event_id






