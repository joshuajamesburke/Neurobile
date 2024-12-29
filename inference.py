import torch
from model import EEGModel

class EEGInferenceApp:
    """ 
    Class to determine if user imagined a left or right movement.
    Expects EEG data for C3, C4, Cz, sampled 128 Hz, 50-100 uV, bandpass 4-50 Hz
    Requires trained CHECKPOINT_FILE
    """
    LEFT = 0
    RIGHT = 1
    _CHECKPOINT_FILE = "data/checkpoint.pt"    

    def __init__(self):
        self.device = (torch.device('mps') if torch.backends.mps.is_available() else 
            torch.device('cuda') if torch.cuda.is_available() else 
            torch.device('cpu'))
        self.model = EEGModel().to(device=self.device)
        self._load_checkpoint()
    
    def predict_imagined_movement(self, data):
        """ 
        Args:
            data: np array [3, 256] where first dim is channel (C3, C4, Cz), second dim is time

        Returns:
            int: 0/1 for left/right imagined movement
        """
        #self.model.eval()
        eeg = torch.tensor(data).float().unsqueeze(0).unsqueeze(0)  # [batch, conv, eeg_channels, time_series]

        eeg = eeg.to(device=self.device, non_blocking=True)
        _, probabilities = self.model(eeg)
        _, idx = torch.max(probabilities, dim=1)

        return idx
    
    def _load_checkpoint(self):
        checkpoint = torch.load(self._CHECKPOINT_FILE, weights_only=True, map_location='cpu')
        self.model.load_state_dict(checkpoint["MODEL_STATE"])
        epochs_run = checkpoint["EPOCHS_RUN"]
        print(f"Loading checkpoint at epoch {epochs_run}")
