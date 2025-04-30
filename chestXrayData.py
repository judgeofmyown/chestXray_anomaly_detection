import os
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

# load_dotenv()
# data_path = os.getenv("DATA_PATH")

# train_df = pd.read_csv(data_path)

def preprocess_data(df):
    df = df.fillna(0)

    df["x_min"] = df.apply(lambda row: (row.x_min)/row.width, axis=1)
    df["x_max"] = df.apply(lambda row: (row.x_max)/row.width, axis=1)
    df["y_min"] = df.apply(lambda row: (row.y_min)/row.height, axis=1)
    df["y_max"] = df.apply(lambda row: (row.y_max)/row.height, axis=1)

    dist_images = np.array(df["image_id"].value_counts().index)

    return dist_images, df

# dataset = 'train' for training 'test' for testing
class chestXDataset(Dataset):
    def __init__(self, df, dist_images, dataset):
        self.df = df
        self.dist_images = dist_images
        self.dataset=dataset

    def __len__(self):
        return len(self.dist_images)
    
    def __getitem__(self, idx):
        img_id = self.dist_images[idx]
        data = self.df[self.df["image_id"] == img_id]
        boxes = data.values[:, 4:8]
        labels = data.values[:, 2]

        boxes = np.vstack(boxes).astype(np.float32)
        labels = np.vstack(labels).astype(np.float32)
        
        img_path = f"C:\Users\tanbi\Downloads\xray_dataset\vinbigdata\{self.dataset}"+ img_id +".jpg"
        # img = Image.open(img_path)
        # call preprocess_img....
        image = preprocess_img(img_path)
        targets = {}
        if np.all(boxes == 0):
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:  
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64).flatten()
        
        targets["boxes"] = boxes
        targets["labels"] = labels
        return image, targets

def preprocess_img(img_path):
    """
        preprocess the image data  
        --- resize image shape 
        --- convert to tensor
    """
    img = Image.open(img_path)
    img = img.resize((500, 500))
    
    # transforms ----------
    transform = T.Compose([
        T.ToTensor()
    ])
    
    img_tensor = transform(img)
    return img_tensor

def createLoaders(df, train_ratio, val_ratio, batch_size):
    test_ratio = 1 - (train_ratio + val_ratio)

    train_size = int(train_ratio * len(df))
    val_size = int(val_ratio * len(df))
    test_size = len(df) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(df, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

def test():
    print("working..")
