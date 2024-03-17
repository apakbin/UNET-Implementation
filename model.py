import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_k_size, conv_k_size, conv_stride):
        """
        Creates a DownsampleBlock.

        in_channels:  Number of input channels
        out_channels: Number of output channels
        pool_k_size:  Max pool kernel size (same used for stride in max pool)
        conv_k_size:  Conv2d kernel size
        conv_stride:  Conv2d stride
        """
        super().__init__()

        # Pooling layer. Setting stride equal to kernel size
        self.pool = nn.MaxPool2d(kernel_size = pool_k_size, stride = pool_k_size)


        # Conv layers
        self.conv = nn.Sequential(
            # Conv2d layer. Number of channels does not change. Padding = same to ensure input size is same as output
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size = conv_k_size, 
                      stride      = conv_stride, 
                      padding     = 'same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 
                      kernel_size = conv_k_size, 
                      stride      = conv_stride, 
                      padding     = 'same'),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.pool(x))