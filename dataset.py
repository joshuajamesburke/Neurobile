import mne
import numpy as np
import os
import torch
from torch.utils.data import Dataset    
import matplotlib.pyplot as plt

class EEGDataSet(Dataset):
    """
    EEG Dataset that returns [[tensor, label]] where tensor shape is [1, 3, 384], 1 conv channel. 3 eeg channels, 256 samples
    Data description: https://www.bbci.de/competition/iv/desc_2b.pdf
    250 Hz, +/-50 uV, 0.5 - 100Hz with notch at 50 Hz. Downsampled/decimated to 128 Hz

    Events:
        276 0x0114 Idling EEG (eyes open)
        277 0x0115 Idling EEG (eyes closed)
        768 0x0300 Start of a trial (9)
        769 0x0301 Cue onset left (class 1) (label 0)
        770 0x0302 Cue onset right (class 2) (label 1)
        781 0x030D BCI feedback (continuous)
        783 0x030F Cue unknown
        1023 0x03FF Rejected trial
        1077 0x0435 Horizontal eye movement
        1078 0x0436 Vertical eye movement
        1079 0x0437 Eye rotation
        1081 0x0439 Eye blinks
        32766 0x7FFE Start of a new run    
    """

    PREPROCESSED_DATA_FILE = "data/preprocessed_data.pt"

    def __init__(self):
        if not os.path.exists(self.PREPROCESSED_DATA_FILE) :
            self._preprocess()
        self.data = torch.load(self.PREPROCESSED_DATA_FILE, weights_only=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def _preprocess(self):    
        self.data = []
        for i in range(1, 10): # subjects 10
            for j in range (1, 4): # sessions 4
                raw_data = mne.io.read_raw_gdf(f'data/B0{i}0{j}T.gdf', preload=True)
                raw_data = raw_data.filter(l_freq=4, h_freq=50) # 4 hz to reduce EOG contamination and drift
                raw_data = raw_data.resample(128)  # jitters slightly but neural net will learn anyways

                # Get left, right event times and label as 0, 1 respectively
                events, events_dict = mne.events_from_annotations(raw_data) # events = [[time, 1, type]]
                mask = (events[:, 2] == events_dict['769']) | (events[:, 2] == events_dict['770'])
                lr_events = [[row[0], (0 if row[2] == events_dict['769'] else 1)] for row in events[mask]]

                # For each event, append EEG data for 3 channels at the event start time + 3s in addition to label
                # Create additional synthetic data by sliding the window forward
                for lr_event in lr_events:
                    T = int(3 * raw_data.info['sfreq'])
                    for offset in range(0, T//3, 2):
                        start = lr_event[0] + offset
                        stop = start + T
                        selected_channel_names = ['EEG:C3', 'EEG:C4', 'EEG:Cz']  
                        eeg_data = torch.tensor(raw_data[selected_channel_names, start:stop][0]).float().unsqueeze(0)
                        self.data = self.data + [[eeg_data, lr_event[1]]]
            
        torch.save(self.data, self.PREPROCESSED_DATA_FILE)

def main():
    # Debug - load dataset, print EEG channels for first entry
    data = EEGDataSet()
    lines = plt.plot(data[0][0].squeeze().numpy().T)
    plt.legend(lines, ['EEG:C3', 'EEG:C4', 'EEG:Cz'])
    plt.show()

if __name__ == "__main__":
    main()


