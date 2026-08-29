import mne
from moabb.datasets import BNCI2014_001

def get_cleaned_epochs(subject_id=1):
    # All the logic you wrote in Notebook 1
    dataset = BNCI2014_001()
    sessions = dataset.get_data(subjects=[subject_id])
    raw = sessions[subject_id]['0train']['0']
    raw.load_data()
    
    # Preprocessing
    raw.filter(7., 30., fir_design='firwin', verbose=False)
    
    # Epoching
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=4.0, 
                        baseline=(None, 0), preload=True, verbose=False)
    return epochs, event_id
