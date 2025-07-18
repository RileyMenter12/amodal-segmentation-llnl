#3D U-Net

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super().__init__()
        self.enc1 = DoubleConv3D(in_channels, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.enc2 = DoubleConv3D(64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.enc3 = DoubleConv3D(128, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.enc4 = DoubleConv3D(256, 512)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.bottleneck = DoubleConv3D(512, 1024)

        self.up1 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = DoubleConv3D(1024, 512)
        self.up2 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = DoubleConv3D(512, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = DoubleConv3D(256, 128)
        self.up4 = nn.ConvTranspose3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec4 = DoubleConv3D(128, 64)

        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)           # (B, 64, T, H, W)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        dec1 = self.dec1(torch.cat([self.up1(bottleneck), enc4], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec1), enc3], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec2), enc2], dim=1))
        dec4 = self.dec4(torch.cat([self.up4(dec3), enc1], dim=1))

        output = torch.sigmoid(self.final_conv(dec4))  # (B, out_channels, T, H, W)
        return output
