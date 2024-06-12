import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

class BaseHDF5Dataset(Dataset):
    def __init__(self, hdf5_file, max_files=None):
        self.hdf5_file = hdf5_file
        self.f = h5py.File(hdf5_file, 'r')
        self.files = list(self.f.keys())
        if max_files:
            self.files = self.files[:max_files]

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            file = self.files[idx]
            # Load common attributes
            flux = f[file]['flux'][:]
            wavelength = f[file]['wavelength'][:]
            mask = f[file]['mask'][:]
            ivar = f[file]['ivar'][:]
            spectrum_index = f[file]['spectrum_index'][()]  # Read index
            
            data = {
                'spectrum_index': torch.tensor(spectrum_index, dtype=torch.long),
                'flux': torch.tensor(flux, dtype=torch.float32),
                'wavelength': torch.tensor(wavelength, dtype=torch.float32),
                'mask': torch.tensor(mask, dtype=torch.float32),
                'ivar': torch.tensor(ivar, dtype=torch.float32)
            }
            self.add_specific_attributes(f, file, data)
            return data

    def add_specific_attributes(self, f, file, data):
        # This method will be overridden by subclasses to add specific attributes
        pass

    def __len__(self):
        return len(self.files)

    def __del__(self):
        self.f.close()
