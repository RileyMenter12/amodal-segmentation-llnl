import os
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from PIL import Image
import numpy as np

class VideoAmodalDataset(Dataset):
    def __init__(self, root_dir, modal_mask_root_dir, transform=None, num_frames=16):
        self.root_dir = root_dir  # Main data (RGB, segmentation, amodal)
        self.modal_mask_root = modal_mask_root_dir  # Separate modal mask location
        self.transform = transform
        self.num_frames = num_frames
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for scene_id in os.listdir(self.root_dir):
            scene_path = os.path.join(self.root_dir, scene_id)
            if not os.path.isdir(scene_path):
                continue

            for camera_id in os.listdir(scene_path):
                cam_path = os.path.join(scene_path, camera_id)
                if not os.path.isdir(cam_path) or not camera_id.startswith("camera_"):
                    continue

                for object_id in os.listdir(cam_path):
                    obj_path = os.path.join(cam_path, object_id)
                    if not os.path.isdir(obj_path) or not object_id.startswith("obj_"):
                        continue

                    # Get video paths
                    rgb_video_path = os.path.join(cam_path, "rgba_output.mp4")
                    binary_video_path = os.path.join(obj_path, "segmentation_output.mp4")  # global
                    amodal_video_path = os.path.join(obj_path, "rgba_output.mp4")  # RGB for the object

                    # Modal mask from separate root
                    modal_video_path = os.path.join(
                        self.modal_mask_root, scene_id, camera_id, object_id, "modal_mask_video.mp4"
                    )

                    # Only add if all exist
                    if (
                        os.path.exists(rgb_video_path)
                        and os.path.exists(binary_video_path)
                        and os.path.exists(modal_video_path)
                    ):
                        samples.append({
                            "rgb_video": rgb_video_path,
                            "binary_video": binary_video_path,
                            "amodal_video": amodal_video_path,
                            "modal_video": modal_video_path,
                            "scene_id": scene_id,
                            "camera_id": camera_id,
                            "object_id": object_id
                        })
        return samples

    def _read_video_frames(self, video_path, is_mask=False):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if is_mask:
                # For mask videos, convert to grayscale and binarize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # single channel
                _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
                frame = cv2.resize(frame, (224, 224))
                frame = Image.fromarray(frame)  # grayscale PIL Image
            else:
                # For RGB videos
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frame = Image.fromarray(frame)  # RGB PIL Image

            if self.transform:
                frame = self.transform(frame)  # This converts PIL -> Tensor (CxHxW)

            frames.append(frame)
        cap.release()

        # Handle empty or too few frames
        if len(frames) == 0:
            if is_mask:
                black_frame = torch.zeros((1, 224, 224), dtype=torch.float32)
            else:
                black_frame = torch.zeros((3, 224, 224), dtype=torch.float32)
            frames = [black_frame] * self.num_frames

        if len(frames) < self.num_frames:
            black_frame = frames[0].clone()
            padding = [black_frame] * (self.num_frames - len(frames))
            frames += padding

        elif len(frames) > self.num_frames:
            frames = frames[:self.num_frames]

        return torch.stack(frames)  # Shape: [T, C, H, W]

 # Shape: [num_frames, C, H, W]


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rgb_frames = self._read_video_frames(sample["rgb_video"], is_mask=False)
        binary_frames = self._read_video_frames(sample["binary_video"], is_mask=True)  # masks
        amodal_frames = self._read_video_frames(sample["amodal_video"], is_mask=True)  # masks
        modal_frames = self._read_video_frames(sample["modal_video"], is_mask=True)  # masks

        return {
            "rgb_frames": rgb_frames,                # [T, 3, H, W]
            "binary_frames": binary_frames,          # [T, 1, H, W]
            "amodal_frames": amodal_frames,          # [T, 1, H, W]
            "modal_mask_frames": modal_frames,       # [T, 1, H, W]
            "scene_id": sample["scene_id"],
            "camera_id": sample["camera_id"],
            "object_id": sample["object_id"]
        }

if name == "__main__":
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

    dataset = VideoAmodalDataset(
        root_dir="your root",
        modal_mask_root_dir="modal_mask_root",
        transform=transform,
        num_frames=24
    )


    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
