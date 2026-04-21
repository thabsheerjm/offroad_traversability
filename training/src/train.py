import os, torch, torch.nn as nn, torch.optim as optim, json
from torch.utils.data import DataLoader, random_split
from torchvision import models 
from dataset import GOOSEDataset
from tqdm import tqdm 
from model import DeepLabHead
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_config(config_path="src/config/config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
        return config
    
def compute_metrics(preds, masks):
    preds = (torch.sigmoid(preds)>0.5).int()
    masks = masks.int()

    intersection = (preds & masks).sum().item()
    union = (preds | masks).sum().item()
    correct = (preds == masks).sum().item()
    total = preds.numel()

    iou = intersection /union if union>0 else 1.0
    accuracy = correct/total
    return accuracy, iou 

def train(config):
    # Get configuration
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
    best_val_iou = -float('inf')

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stop_patience = 10
    epochs_since_improvement = 0 

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
        val_accuracy = 0 
        val_iou = 0 
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc="[Val]"):
                imgs = imgs.cuda() if torch.cuda.is_available() else imgs
                masks = masks.cuda() if torch.cuda.is_available() else masks

                outputs = model(imgs)["out"]
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                acc, iou =  compute_metrics(outputs, masks)
                val_accuracy += acc
                val_iou += iou 

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_accuracy/ len(val_loader)
        avg_val_iou =  val_iou/ len(val_loader)


        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, "
                f"Val Acc = {avg_val_acc:.4f}, Val IoU = {avg_val_iou:.4f}")
        
        scheduler.step(avg_val_loss)

        # Checkpointing
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model updated: Val IoU = {best_val_iou:.4f}")
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epoch(s)")

        # Early stopping
        if epochs_since_improvement >= early_stop_patience:
            print("Early stopping triggered! Exiting.")
            break

if __name__ == '__main__':
    config = train_config()
    train(config)