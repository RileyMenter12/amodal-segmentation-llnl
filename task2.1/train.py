import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import wandb
import cv2

# Initialize wandb
wandb.init(
    project="amodal-mask-training",
    name="unet3d-video-mask-run",
    config={
        "epochs": 30,
        "batch_size": 5,
        "lr": 1e-4,
        "input_frames": 16,
        "architecture": "3D UNet"
    }
)

def log_wandb_samples(rgb_video, modal_video, gt_amodal_video, pred_amodal_video, step=0):
    def prepare(video):
      if video.dim() == 5:  # [1, C, T, H, W]
          video = video.squeeze(0)
      elif video.dim() != 4:
          raise ValueError(f"Expected 4D or 5D tensor, got {video.shape}")

      # Now video is [C, T, H, W]
      video = video.permute(1, 2, 3, 0)  # [T, H, W, C]
      return (video.cpu().numpy() * 255).astype(np.uint8)

    wandb.log({
        "RGB Input Video": wandb.Video(prepare(rgb_video), fps=2, format="mp4"),
        "Modal Mask Video": wandb.Video(prepare(modal_video), fps=2, format="mp4"),
        "GT Amodal Mask Video": wandb.Video(prepare(gt_amodal_video), fps=2, format="mp4"),
        "Predicted Amodal Mask Video": wandb.Video(prepare(pred_amodal_video), fps=2, format="mp4"),
    }, step=step)

def save_video_from_tensor(video_tensor, output_path, fps=10):
    video_np = video_tensor.numpy()  # [T, C, H, W]
    T, C, H, W = video_np.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for frame in video_np:
        if C == 1:
            frame = frame[0]
            frame = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif C == 3:
            frame = np.transpose(frame, (1, 2, 0))
            frame = (frame * 255).astype(np.uint8)
            frame_bgr = frame
        else:
            raise ValueError(f"Unsupported number of channels: {C}")
        out.write(frame_bgr)
    out.release()

class VideoUNetTrainer:
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4, output_dir="training_outputs"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training"):
            rgb = batch['rgb_frames'].to(self.device)
            modal = batch['modal_mask_frames'].to(self.device)
            amodal = batch['amodal_frames'].to(self.device)

            input_tensor = torch.cat([rgb, modal], dim=2).permute(0, 2, 1, 3, 4)
            target = amodal.permute(0, 2, 1, 3, 4)

            self.optimizer.zero_grad()
            output = self.model(input_tensor)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        print(f"Training Loss: {avg_loss:.6f}")
        wandb.log({"train_loss": avg_loss}, step=epoch)
        return avg_loss

    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                rgb = batch['rgb_frames'].to(self.device)
                modal = batch['modal_mask_frames'].to(self.device)
                amodal = batch['amodal_frames'].to(self.device)

                input_tensor = torch.cat([rgb, modal], dim=2).permute(0, 2, 1, 3, 4)
                target = amodal.permute(0, 2, 1, 3, 4)

                output = self.model(input_tensor)
                loss = self.criterion(output, target)
                running_loss += loss.item()

                if i == 0:
                    # Log a few videos for comparison
                    pred_binary = (output > 0.5).float()
                    log_wandb_samples(rgb, modal, amodal, pred_binary, step=epoch)

                # Optionally save video
                scene_id = batch["scene_id"][0]
                camera_id = batch["camera_id"][0]
                object_id = batch["object_id"][0]

                pred_binary = (output[0] > 0.5).float().cpu()
                video_tensor = pred_binary.permute(1, 0, 2, 3)  # [T, C, H, W]

                save_video_from_tensor(
                    video_tensor,
                    os.path.join(self.output_dir, f"{scene_id}_{camera_id}_{object_id}_epoch{epoch+1}.mp4")
                )

        avg_loss = running_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        print(f"Validation Loss: {avg_loss:.6f}")
        wandb.log({"val_loss": avg_loss}, step=epoch)
        return avg_loss

    def save_checkpoint(self, epoch):
        save_path = os.path.join(self.output_dir, f"unet3d_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, save_path)
        print(f"Checkpoint saved to {save_path}")

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
            self.save_checkpoint(epoch)
