import os, torch, torch.nn as nn, torch.optim as optim, json
from torch.utils.data import DataLoader, random_split
from torchvision import models 
from dataset import GOOSEDataset
from tqdm import tqdm 
from src.model import DeepLabHead

def train_config(config_path="src/config/config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
        return config

def train(config):
    rgb_dir = config["data"]["rgb_dir"]
    mask_dir = config["data"]["mask_dir"]
    mapping_csv = config["data"]["mapping_csv"]
    traversable_labels = config["data"]["traversable_labels"]
    val_split = config["training"]["val_split"]
    batch_size = config["training"]["batch_size"]
    checkpoint_path = config["training"]["checkpoint_path"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]

    dataset = GOOSEDataset(rgb_dir, mask_dir, mapping_csv, traversable_labels)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
        
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights_backbone="DEFAULT", weights=None)
    model.classifier = DeepLabHead(960, 1)
    model = model.cuda() if torch.cuda.is_available() else model

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            imgs = imgs.cuda() if torch.cuda.is_available() else imgs
            masks = masks.cuda() if torch.cuda.is_available() else masks
            optimizer.zero_grad()
            outputs = model(imgs)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc="[Val]"):
                imgs = imgs.cuda() if torch.cuda.is_available() else imgs
                masks = masks.cuda() if torch.cuda.is_available() else masks

                outputs = model(imgs)["out"]
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model updated: Val Loss = {best_val_loss:.4f}")

if __name__ == '__main__':
    config = train_config()
    train(config)