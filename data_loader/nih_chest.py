import kagglehub
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


def download_nih_chest_dataset(target_dir="data/NIH-Chest"):
    """
    Downloads the NIH Chest X-ray dataset from Kaggle using kagglehub
    and saves it directly to the specified target directory using dest_dir.
    """
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"Dataset already downloaded at: {target_dir}")
        return target_dir

    print(f"Downloading NIH Chest X-ray dataset using kagglehub directly to {target_dir}...")
    
    # Download directly to target directory
    path = kagglehub.dataset_download("nih-chest-xrays/data", dest_dir=target_dir)
    
    print(f"Dataset successfully stored in: {path}")
    return path


class NIHChestDataset(Dataset):
    def __init__(self, data_dir, image_list, df, transform=None):
        self.data_dir = data_dir
        self.image_list = image_list
        self.transform = transform
        
        # Build image path dictionary for fast lookup
        self.image_paths = {}
        for i in range(1, 13):
            folder_path = os.path.join(data_dir, f"images_{i:03d}", "images")
            if os.path.exists(folder_path):
                for img_name in os.listdir(folder_path):
                    self.image_paths[img_name] = os.path.join(folder_path, img_name)
                    
        # Filter dataframe for the images in our split
        self.df = df[df['Image Index'].isin(image_list)].set_index('Image Index')
        
        # Define the 14 standard finding labels
        self.all_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
            'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        
        # Get image path
        img_path = self.image_paths.get(img_name)
        if img_path is None:
            # Fallback if image not found in expected structure
            img_path = os.path.join(self.data_dir, img_name)
            
        # Load image with PIL and convert to RGB (grayscale -> 3 channels)
        image = Image.open(img_path).convert("RGB")
        
        # Apply torchvision transforms
        if self.transform:
            image = self.transform(image)

        # Get labels
        labels_str = self.df.loc[img_name, 'Finding Labels']
        labels_list = labels_str.split('|')
        
        # Multi-hot encode labels
        labels = torch.zeros(len(self.all_labels), dtype=torch.float32)
        for i, label in enumerate(self.all_labels):
            if label in labels_list:
                labels[i] = 1.0

        return {"image": image, "label": labels}


def create_data_loaders(data_dir="data/NIH-Chest", batch_size=32, num_workers=4, client_id=None, num_clients=1):
    """
    Creates train, validation, and test data loaders.
    Uses 90% for training and 10% for validation from train_val_list.txt
    and test_list.txt for test loader.
    """
    # Use standard torchvision transforms for reliable 2D image handling
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),           # Converts PIL [0,255] -> [0,1] float tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load dataframe
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)

    # Read train_val and test lists
    with open(os.path.join(data_dir, "train_val_list.txt"), "r") as f:
        train_val_list = [line.strip() for line in f.readlines() if line.strip()]
        
    with open(os.path.join(data_dir, "test_list.txt"), "r") as f:
        test_list = [line.strip() for line in f.readlines() if line.strip()]

    # Split train_val into 90% train and 10% val
    train_list, val_list = train_test_split(train_val_list, test_size=0.10, random_state=42)

    # Partition data if running in a federated setting with multiple clients
    if client_id is not None and num_clients > 1:
        print(f"Partitioning data for {client_id} (Total clients: {num_clients})")
        # Ensure lists are sorted for deterministic partitioning
        train_list.sort()
        val_list.sort()
        
        # Determine client index (e.g., 'site-1' -> 0, 'site-2' -> 1)
        try:
            client_idx = int(client_id.split('-')[-1]) - 1
        except Exception as e:
            print(f"Warning: Could not parse client index from {client_id}. Defaulting to index 0.")
            client_idx = 0
            
        client_idx = max(0, min(client_idx, num_clients - 1)) # bound check
        
        # Calculate chunk boundaries for train
        train_chunk_size = len(train_list) // num_clients
        train_start = client_idx * train_chunk_size
        train_end = train_start + train_chunk_size if client_idx < num_clients - 1 else len(train_list)
        train_list = train_list[train_start:train_end]
        
        # Calculate chunk boundaries for val
        val_chunk_size = len(val_list) // num_clients
        val_start = client_idx * val_chunk_size
        val_end = val_start + val_chunk_size if client_idx < num_clients - 1 else len(val_list)
        val_list = val_list[val_start:val_end]
        
        print(f"{client_id} partition size -> Train: {len(train_list)}, Val: {len(val_list)}")

    # Create datasets
    print("Creating datasets...")
    train_dataset = NIHChestDataset(data_dir, train_list, df, transform=train_transform)
    val_dataset = NIHChestDataset(data_dir, val_list, df, transform=val_test_transform)
    test_dataset = NIHChestDataset(data_dir, test_list, df, transform=val_test_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Data loaders created: Train ({len(train_loader)} batches), Val ({len(val_loader)} batches), Test ({len(test_loader)} batches)")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Calculate target directory relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    target_data_dir = os.path.join(project_root, "data", "NIH-Chest")
    
    download_nih_chest_dataset(target_dir=target_data_dir)
    
    # You can instantiate dataloaders here:
    # train_loader, val_loader, test_loader = create_data_loaders(data_dir=target_data_dir, batch_size=32)