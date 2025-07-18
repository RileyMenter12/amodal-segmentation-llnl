import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import glob

class ModalAmodalDataset(Dataset):
    def __init__(self, root_dir, split, transform=None, img_size=(224, 224)):
        """
        Args:
            root_dir (str): Root directory containing train folder
            transform (callable, optional): Optional transform to be applied on samples
            img_size (tuple): Target image size for resizing
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size

        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])

        # Build dataset index
        self.samples = self._build_sample_index()

    def _build_sample_index(self):
        """Build index of all valid samples"""
        samples = []

        # Navigate through train directory
        split_dir = os.path.join(self.root_dir, self.split)

        # Iterate through all subdirectories (scene folders)
        for scene_folder in os.listdir(split_dir):
            scene_path = os.path.join(split_dir, scene_folder)
            if not os.path.isdir(scene_path):
                continue

            # Look for camera folders
            for camera_folder in os.listdir(scene_path):
                camera_path = os.path.join(scene_path, camera_folder)
                if not os.path.isdir(camera_path) or not camera_folder.startswith('camera_'):
                    continue

                # Get all RGBA images in camera folder
                rgba_files = glob.glob(os.path.join(camera_path, 'rgba_*.png'))
                segmentation_files = glob.glob(os.path.join(camera_path, 'segmentation_*.png'))

                if not rgba_files or not segmentation_files:
                    continue

                # Sort files to ensure proper ordering
                rgba_files.sort()
                segmentation_files.sort()

                # Look for object folders
                for obj_folder in os.listdir(camera_path):
                    obj_path = os.path.join(camera_path, obj_folder)
                    if not os.path.isdir(obj_path) or not obj_folder.startswith('obj_'):
                        continue

                    # Extract object ID from folder name
                    try:
                        obj_id = int(obj_folder.split('_')[1])
                    except:
                        continue

                    # Get amodal segmentation files
                    amodal_files = glob.glob(os.path.join(obj_path, 'segmentation_*.png'))
                    amodal_files.sort()

                    # Match frames between RGB, modal, and amodal
                    for rgba_file in rgba_files:
                        frame_name = os.path.basename(rgba_file).replace('rgba_', '').replace('.png', '')

                        # Find corresponding segmentation file
                        seg_file = os.path.join(camera_path, f'segmentation_{frame_name}.png')
                        amodal_file = os.path.join(obj_path, f'segmentation_{frame_name}.png')

                        if os.path.exists(seg_file) and os.path.exists(amodal_file):
                            samples.append({
                                'rgb_path': rgba_file,
                                'segmentation_path': seg_file,
                                'amodal_path': amodal_file,
                                'object_id': obj_id,
                                'frame_id': frame_name,
                                'scene': scene_folder,
                                'camera': camera_folder
                            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load RGB image (convert RGBA to RGB)
        rgb_image = Image.open(sample['rgb_path']).convert('RGB')

        # Load panoptic segmentation
        panoptic_seg = Image.open(sample['segmentation_path'])
        panoptic_array = np.array(panoptic_seg)

        # Create modal mask for specific object
        modal_mask = (panoptic_array == sample['object_id']).astype(np.uint8)
        modal_mask = Image.fromarray(modal_mask * 255)  # Convert to 0-255 range

        # Load amodal mask
        amodal_mask = Image.open(sample['amodal_path']).convert('L')

        # Apply transforms
        if self.transform:
            rgb_tensor = self.transform(rgb_image)
            modal_tensor = self.transform(modal_mask)
            amodal_tensor = self.transform(amodal_mask)
        else:
            # Convert to tensors if no transform provided
            rgb_tensor = transforms.ToTensor()(rgb_image)
            modal_tensor = transforms.ToTensor()(modal_mask)
            amodal_tensor = transforms.ToTensor()(amodal_mask)

        # Ensure masks are single channel
        if modal_tensor.shape[0] > 1:
            modal_tensor = modal_tensor[0:1]  # Take first channel
        if amodal_tensor.shape[0] > 1:
            amodal_tensor = amodal_tensor[0:1]  # Take first channel

        return {
            'rgb': rgb_tensor,
            'modal_mask': modal_tensor,
            'amodal_mask': amodal_tensor,
            'object_id': sample['object_id'],
            'frame_id': sample['frame_id'],
            'scene': sample['scene'],
            'camera': sample['camera']
        }

def create_dataloader(root_dir, split, batch_size=4, shuffle=True, num_workers=4, img_size=(224, 224)):
    """
    Create a DataLoader for the modal-amodal dataset

    Args:
        root_dir (str): Root directory containing the dataset
        batch_size (int): Batch size for training
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of worker processes for data loading
        img_size (tuple): Target image size

    Returns:
        DataLoader: PyTorch DataLoader object
    """

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = ModalAmodalDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        img_size=img_size
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return dataloader

# Example usage
if __name__ == "__main__":
   # Example usage
   root_directory = "/content/DSC_DATA"  # Replace with your actual path

   # Create dataloader
   train_loader = create_dataloader(
       root_dir=root_directory,
       split='train',
       batch_size=1,
       shuffle=True,
       num_workers=0,
       img_size=(224, 224)
   )

   # Test the dataloader
   print(f"Dataset size: {len(train_loader.dataset)}")

   # Get one batch and save visualization
   for batch_idx, batch in enumerate(train_loader):
       if batch_idx >= 5:  # Save first 5 samples
           break

       # Extract tensors (remove batch dimension)
       rgb_tensor = batch['rgb'][0]  # Shape: (3, H, W)
       modal_tensor = batch['modal_mask'][0]  # Shape: (1, H, W)
       amodal_tensor = batch['amodal_mask'][0]  # Shape: (1, H, W)

       # Convert tensors to PIL images
       rgb_img = transforms.ToPILImage()(rgb_tensor)
       modal_img = transforms.ToPILImage()(modal_tensor.squeeze(0))
       amodal_img = transforms.ToPILImage()(amodal_tensor.squeeze(0))

       # Get image dimensions
       width, height = rgb_img.size

       # Create side-by-side image
       combined_img = Image.new('RGB', (width * 3, height))
       combined_img.paste(rgb_img, (0, 0))
       combined_img.paste(modal_img.convert('RGB'), (width, 0))
       combined_img.paste(amodal_img.convert('RGB'), (width * 2, 0))

       # Save the combined image
       filename = f"sample_{batch_idx + 1}_obj{batch['object_id'][0]}_frame{batch['frame_id'][0]}.png"
       combined_img.save(filename)
       print(f"Saved: {filename}")

       print(f"RGB range: [{rgb_tensor.min():.3f}, {rgb_tensor.max():.3f}]")
       print(f"Modal mask range: [{modal_tensor.min():.3f}, {modal_tensor.max():.3f}]")
       print(f"Amodal mask range: [{amodal_tensor.min():.3f}, {amodal_tensor.max():.3f}]")