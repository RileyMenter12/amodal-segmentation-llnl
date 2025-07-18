#Model and trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import shutil
from dataloader import ModalAmodalDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec1 = self.dec1(torch.cat([self.up1(bottleneck), enc4], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec1), enc3], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec2), enc2], dim=1))
        dec4 = self.dec4(torch.cat([self.up4(dec3), enc1], dim=1))
        output = torch.sigmoid(self.final_conv(dec4))
        return output

class ModalAmodalTrainer:
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4, output_dir="training_outputs"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.val_images_dir = os.path.join(output_dir, "validation_images")
        os.makedirs(self.val_images_dir, exist_ok=True)

        self.criterion = nn.MSELoss()  # For RGB regression
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(self.train_loader):
            rgb = batch['rgb'].to(self.device)
            modal_mask = batch['modal_mask'].to(self.device)
            amodal_mask = batch['amodal_mask'].to(self.device)  # Now 3-channel RGB
            input_tensor = torch.cat([rgb, modal_mask], dim=1) # Concatenate along channel dimension
            self.optimizer.zero_grad()
            predictions = self.model(input_tensor)
            loss = self.criterion(predictions, amodal_mask)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Train Batch [{batch_idx}/{len(self.train_loader)}], Loss: {loss.item():.6f}')
        return total_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        if os.path.exists(self.val_images_dir):
            shutil.rmtree(self.val_images_dir)
        os.makedirs(self.val_images_dir, exist_ok=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                rgb = batch['rgb'].to(self.device)
                modal_mask = batch['modal_mask'].to(self.device)
                amodal_mask = batch['amodal_mask'].to(self.device)
                input_tensor = torch.cat([rgb, modal_mask], dim=1) # Concatenate along channel dimension
                predictions = self.model(input_tensor)
                loss = self.criterion(predictions, amodal_mask)
                total_loss += loss.item()
                if batch_idx < 10:
                    self.save_validation_images(rgb[0], modal_mask[0], amodal_mask[0], predictions[0], epoch, batch_idx)
        return total_loss / len(self.val_loader)

    def save_validation_images(self, rgb, modal_mask, gt_amodal, pred_amodal, epoch, batch_idx):
        rgb_img = transforms.ToPILImage()(rgb.cpu())
        modal_img = transforms.ToPILImage()(modal_mask.cpu().repeat(3, 1, 1))
        gt_img = transforms.ToPILImage()(gt_amodal.cpu())
        pred_img = transforms.ToPILImage()(pred_amodal.cpu().clamp(0, 1))
        width, height = rgb_img.size
        combined_img = Image.new('RGB', (width * 4, height))
        combined_img.paste(rgb_img, (0, 0))
        combined_img.paste(modal_img, (width, 0))
        combined_img.paste(gt_img, (width * 2, 0))
        combined_img.paste(pred_img, (width * 3, 0))
        filename = f"epoch_{epoch+1}_batch_{batch_idx+1}.png"
        filepath = os.path.join(self.val_images_dir, filename)
        combined_img.save(filepath)
        print(f"Saved validation image: {filename}")

    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            val_loss = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            print(f"Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.6f}")
            checkpoint_path = os.path.join(self.output_dir, f"model_rgb.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
        self.plot_training_history()
        print(f"\nTraining completed! Results saved in {self.output_dir}")

    def plot_training_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'training_history_rgb.png'))
        plt.close()


def create_model_and_trainer(train_loader, val_loader, device, lr=1e-4, output_dir="training_outputs"):
    model = UNet(in_channels=6, out_channels=3)  # RGB output
    trainer = ModalAmodalTrainer(model, train_loader, val_loader, device, lr, output_dir)
    return model, trainer


def load_model_for_inference(model_path, device):
    model = UNet(in_channels=4, out_channels=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    full_train_dataset = ModalAmodalDataset(root_dir="/content/drive/MyDrive/Data_DSC2025", split="train")
    train_size = int(0.75 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Load previous model except final layer (for RGB output training)
    # Initialize new model for RGB amodal mask prediction
    model = UNet(in_channels=4, out_channels=3).to(device)

    # Load previous checkpoint (trained on binary mask)
    checkpoint_path = "/content/drive/MyDrive/model.pth"  # Update if needed
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Remove final_conv weights from checkpoint (size mismatch)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("final_conv.")}
    model.load_state_dict(filtered_state_dict, strict=False)
    print("Loaded pretrained weights (excluding final layer).")


    # Continue training with new RGB output
    trainer = ModalAmodalTrainer(model, train_loader, val_loader, device, lr=1e-4, output_dir="training_outputs")
    trainer.train(num_epochs=10)