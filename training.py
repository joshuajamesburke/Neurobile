import torch
import torch.nn as nn
import numpy as np
import os
import torch.optim as optim

from model import EEGModel
from dataset import EEGDataSet

METRICS_SIZE = 3
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2

class EEGTrainingApp:
    CHECKPOINT_FILE = "data/checkpoint.pt"

    def __init__(self):
        self.device = (torch.device('mps') if torch.backends.mps.is_available() else 
                       torch.device('cuda') if torch.backends.cuda.is_available() else 
                       torch.device('cpu'))

        self.epochs = 1000
        self.batch_size = 64

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

        self.epochs_run = 0
        self.save_every = 100

        if os.path.exists(self.CHECKPOINT_FILE):
            self._load_checkpoint()

    def init_model(self):
        return EEGModel().to(device=self.device)
    
    def init_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=0.0001)

    def init_dataloaders(self):
        dataset = EEGDataSet()

        validation_size = int(0.2 * len(dataset))
        shuffled_indices = np.random.permutation(np.array(range(0, len(dataset))))
        train_indices = shuffled_indices[:-validation_size]
        val_indices = shuffled_indices[-validation_size:]
        data_train = [dataset[index] for index in train_indices]
        data_val = [dataset[index] for index in val_indices]

        train_loader = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(data_val, batch_size=self.batch_size, shuffle=True)

        return train_loader, val_loader
    
    def do_training(self, train_dl):
        self.model.train()
        
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device = self.device
        )

        for batch_n, batch_t in enumerate(train_dl):
            self.optimizer.zero_grad()

            loss = self.compute_batch_loss(
                batch_n,
                batch_t,
                train_dl.batch_size,
                trnMetrics_g
            )
            loss.backward()
            self.optimizer.step()
            

        return trnMetrics_g.to('cpu')
    
    def do_validation(self, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device
            )

            for batch_n, batch_t in enumerate(val_dl):
                self.compute_batch_loss(
                        batch_n,
                        batch_t,
                        val_dl.batch_size,
                        valMetrics_g
                )

        return valMetrics_g.to('cpu')
    
    def compute_batch_loss(self, batch_n, batch_t, batch_size, metrics_t):
        eeg, labels = batch_t

        eeg = eeg.to(device=self.device, non_blocking=True) 
        labels = labels.to(device=self.device, non_blocking=True)

        logits, probabilities = self.model(eeg)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        
        start_ndx = batch_n * batch_size
        end_ndx = start_ndx + len(labels)

        # Record metrics for the batch (metrics_t is for the epoch, i.e. all batches)
        metrics_t[METRICS_LABEL_NDX, start_ndx:end_ndx] = labels
        _, metrics_t[METRICS_PRED_NDX, start_ndx:end_ndx] = torch.max(probabilities, dim=1)
        metrics_t[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss

        return loss.mean()

    def main(self):
        train_dl, val_dl = self.init_dataloaders()

        for epoch in range(self.epochs_run, self.epochs + 1):
            trnMetrics_t = self.do_training(train_dl)
            valMetrics_t = self.do_validation(val_dl)
            
            if epoch == self.epochs_run or epoch % 10 == 0:
                print(f'\nEpoch {epoch}: ')
                self.log_metrics('trn', trnMetrics_t)
                self.log_metrics('val', valMetrics_t)

            if epoch > 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

    def log_metrics(self, mode_str, metrics_t):
        loss = metrics_t[METRICS_LOSS_NDX].mean()
        correct = (metrics_t[METRICS_PRED_NDX].int() == metrics_t[METRICS_LABEL_NDX].int()).sum()
        accuracy = 100 * correct / metrics_t.shape[1]

        print(f'\t{mode_str} accuracy {accuracy:.2f}%, loss: {loss:.2f}')

    def _load_checkpoint(self):
        checkpoint = torch.load(self.CHECKPOINT_FILE, weights_only = True)
        self.model.load_state_dict(checkpoint["MODEL_STATE"])
        self.epochs_run = checkpoint["EPOCHS_RUN"]
        print(f"Resuming training from checkpoint at epoch {self.epochs_run}")
    
    def _save_checkpoint(self, epoch):
        checkpoint = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(checkpoint, self.CHECKPOINT_FILE)
        print(f"Saved checkpoint at {self.CHECKPOINT_FILE}")

if __name__ == "__main__":
    EEGTrainingApp().main()
