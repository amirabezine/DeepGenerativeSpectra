from BaseDataset import BaseHDF5Dataset

class StellarDataset(BaseHDF5Dataset):
    def add_specific_attributes(self, f, file, data):
        """
        Adds specific attributes for the StellarDataset. The 'radial_velocity' is nullable
        and will be set to None if not present.
        """
        if 'radial_velocity' in f[file]:
            radial_velocity = f[file]['radial_velocity'][()]
            data['radial_velocity'] = torch.tensor(radial_velocity, dtype=torch.float32)
        else:
            data['radial_velocity'] = None  # Indicates the value can be null
