import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class Camelyon17Dataset(Dataset):
    def __init__(self, root_dir, metadata_file, center=None, split=None, transform=None, limit=None):
        """
        Args:
            root_dir (str): Directory with all the images (e.g., path to 'patches/').
            metadata_file (str): Path to the metadata.csv file.
            center (int or list, optional): Filter by center(s) (0 to 4).
            split (int or list, optional): Filter by split (e.g., 0 for train, 1 for val, 2 for test).
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): Restrict the dataset to the first N samples.
        """
        self.root_dir = root_dir
        # Ensure patient and node are read as strings to preserve padding like '096'
        self.metadata = pd.read_csv(metadata_file, dtype={'patient': str, 'node': str})
        
        # Filter by center
        if center is not None:
            if isinstance(center, int):
                center = [center]
            self.metadata = self.metadata[self.metadata['center'].isin(center)]
            
        # Filter by split
        if split is not None:
            if isinstance(split, int):
                split = [split]
            self.metadata = self.metadata[self.metadata['split'].isin(split)]
            
        if limit is not None:
            if len(self.metadata) > limit:
                self.metadata = self.metadata.sample(n=limit, random_state=42)
            
        self.metadata = self.metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata.iloc[idx]
        patient = row['patient']
        
        # Ensure patient id has 3 digits
        if len(patient) < 3:
            patient = patient.zfill(3)

        node = str(row['node'])
        x = str(row['x_coord'])
        y = str(row['y_coord'])
        
        folder_name = f"patient_{patient}_node_{node}"
        file_name = f"patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png"
        img_path = os.path.join(self.root_dir, folder_name, file_name)
        
        if not os.path.exists(img_path):
             raise FileNotFoundError(f"Image not found at {img_path}")
             
        image = Image.open(img_path).convert('RGB')
        label = int(row['tumor'])

        if self.transform:
            image = self.transform(image)

        return image, label
