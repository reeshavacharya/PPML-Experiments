import os
import glob
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms

# The 14 finding labels in the NIH Dataset
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

class NIHChestDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 1. Map all images across the images_001 to images_012 folders
        image_paths = glob.glob(os.path.join(data_dir, 'images_*', 'images', '*.png'))
        if not image_paths:
            # Fallback in case they aren't nested in an extra 'images' subfolder
            image_paths = glob.glob(os.path.join(data_dir, 'images_*', '*.png'))
            
        self.image_map = {os.path.basename(p): p for p in image_paths}
        
        # 2. Load CSV and list files
        csv_path = os.path.join(data_dir, 'Data_Entry_2017.csv')
        self.df = pd.read_csv(csv_path)
        
        train_val_list_path = os.path.join(data_dir, 'train_val_list.txt')
        test_list_path = os.path.join(data_dir, 'test_list.txt')
        
        with open(train_val_list_path, 'r') as f:
            train_val_images = [line.strip() for line in f.readlines()]
            
        with open(test_list_path, 'r') as f:
            test_images = [line.strip() for line in f.readlines()]

        # 3. Apply Split Logic
        if split in ['train', 'val']:
            # Deterministic split: 80% train, 20% val using seed 42
            train_images, val_images = train_test_split(
                train_val_images, test_size=0.20, random_state=42
            )
            target_images = train_images if split == 'train' else val_images
        elif split == 'test':
            target_images = test_images
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        # Filter dataframe to only include images in our specific split
        self.df = self.df[self.df['Image Index'].isin(target_images)].reset_index(drop=True)
        
        # Filter out images that might be listed but missing from the extracted folders
        self.df = self.df[self.df['Image Index'].isin(self.image_map.keys())].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        img_path = self.image_map[img_name]
        
        # Load image and convert to RGB (ViT expects 3 channels)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Parse multi-hot labels
        labels_str = row['Finding Labels']
        labels = torch.zeros(len(DISEASES), dtype=torch.float32)
        
        if 'No Finding' not in labels_str:
            for i, disease in enumerate(DISEASES):
                if disease in labels_str:
                    labels[i] = 1.0
                    
        return image, labels

def get_transforms():
    # ViT-Base standard input is 224x224
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform