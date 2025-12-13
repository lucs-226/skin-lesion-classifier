import glob
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms, models

# Global Weights Access for Transforms
WEIGHTS = models.EfficientNet_B3_Weights.DEFAULT

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row['label_idx']

class ExternalDataset(Dataset):
    def __init__(self, root_dir, class_map, transform=None):
        self.files = glob.glob(str(root_dir / "**" / "*.jpg"), recursive=True)
        self.transform = transform
        self.class_map = class_map
        
    def __len__(self): 
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        label_str = Path(path).parent.name
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Returns -1 if label is not in our map
        return img, self.class_map.get(label_str, -1)

def get_transforms(img_size):
    # Dynamic Transform Generation from Weights
    auto_transform = WEIGHTS.transforms()
    mean = auto_transform.mean
    std = auto_transform.std
    
    norm_layer = transforms.Normalize(mean=mean, std=std)
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        norm_layer,
        transforms.RandomErasing(p=0.2)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm_layer
    ])
    
    return train_transform, val_transform