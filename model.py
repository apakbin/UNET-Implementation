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
    

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, upsample_s_factor, conv_k_size, conv_stride):
        """
        Creates an UpsampleBlock.

        in_channels:      Number of input channels
        out_channels:     Number of output channels
        skip_channels:    Number of channels in the skip connection
        upsample_s_factor:  Upsample kernel sizes.
        conv_k_size:      Conv2d kernel size
        conv_stride:      Conv2d stride
        """
        super().__init__()

        # Upsample input
        self.up_sample = nn.Upsample(scale_factor = upsample_s_factor)

        # Conv layers. First conv input channels will be number of input channels + skip conn 
        # channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 
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
    
    def forward(self, x, skip_conn):
        """
        Forward method.

        x:         Input tensor.
        skip_conn: The skip connection.
        """

        # x and skip_conn should have the same number of dimensions
        assert x.ndim == skip_conn.ndim

        # upsample the input
        upsampled = self.up_sample(x)

        # the last two dimensions should match between upsampled and skip conn
        assert upsampled.shape[-2:] == skip_conn.shape[-2:]

        concat_dim = 0 if x.ndim == 3 else 1

        return self.conv(torch.cat([upsampled, skip_conn], dim = concat_dim))