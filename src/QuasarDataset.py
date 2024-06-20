import os
import h5py
import torch
from BaseDataset import BaseHDF5Dataset

class QuasarDataset(BaseHDF5Dataset):
    def add_specific_attributes(self, f, file, data):
        
        redshift = f[file]['redshift'][()]
        data['redshift'] = torch.tensor(redshift, dtype=torch.float32)