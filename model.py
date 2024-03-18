import math
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class downsample_block(nn.Module):
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
    

class upsample_block(nn.Module):
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
    

class u_net(nn.Module):
    def __init__(self, 
        in_out_channels       = 1, 
        fbins                 = 256, 
        intermediate_channels = [4, 4, 4, 8, 8, 16],
        pool_k_size           = (2, 1),
        conv_k_size           = (3, 3),
        conv_stride           = (1, 1)):
        """
        Creates a UNet.

        in_out_channels       : Number of channels in the output as well as the output.
        fbins                 : Number of frequency bins. This will be the same as the first dimension in each channel (256 in original design).
        intermediate_channels : List of intermediate number of channels in encoder. With the exception of the last one, the decoder will have the same in reverse order.
        pool_k_size           : Maxpool kernel size.
        conv_k_size           : Conv2d kernel size.
        conv_stride           : Conv2d stride.
        """
        super().__init__()

        # assert fbins is a power of two since the current architecture does not support other scenarios
        assert (fbins & (fbins-1) == 0) and fbins > 0, "ERROR: fbins should be a positive power of two!"

        # The dimension along fbins gets divided by two len(intermediate_channels) - 1 times. So
        # we have to make sure it stays larger than 1, otherwise encoder is too deep.
        assert fbins >= 2**(len(intermediate_channels) - 1), "ERROR: Encoder/Decoder too deep with the current choice of fbins."
        
        # the first two Conv2d layers
        # having the first intermediate channel (first element in intermediate_channels) as h,
        # we will have the following transition in channel numbers:
        # in_out_channels -> floor(sqrt(first intermediate channel)) -> first intermediate channel.
        # We will use pointwise convolution layers to only increase the number of channels.
        first_intermediate = int(math.sqrt(intermediate_channels[0]))

        self.first_conv = nn.Sequential(
            # Conv2d layer. Number of channels does not change. Padding = same to ensure input size is same as output
            nn.Conv2d(in_out_channels, first_intermediate, 
                      kernel_size = 1, 
                      stride      = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(first_intermediate, intermediate_channels[0], 
                      kernel_size = 1, 
                      stride      = 1),
            nn.ReLU(inplace=True),
        )

        # the last Conv2d layer
        # Unlike first_conv, this will only have one Conv2d, not followed by a ReLU
        self.last_conv = nn.Conv2d(intermediate_channels[0], in_out_channels, 
            kernel_size = 1, 
            stride      = 1)
        
        # Encoder: blocks of encoder
        self.enc = nn.ModuleList()

        # Create a list of blocks to implement the encoder, append to the intermediate_channels list
        for i in range(len(intermediate_channels) - 1):
            self.enc.append(downsample_block(
                in_channels  = intermediate_channels[i],
                out_channels = intermediate_channels[i + 1], 
                pool_k_size  = pool_k_size,
                conv_k_size  = conv_k_size,
                conv_stride  = conv_stride))
        
        # Decoder: blocks of decoder
        self.dec = nn.ModuleList()

        # Create a lisf of blocks to implement decoder, go through the intermediate_channels list in reverse
        # If upsample_s_factor is different from pool_k_size, the current architecture will not work, so
        # setting it as such.
        for i in range(len(intermediate_channels) - 1, 0, -1):
            self.dec.append(upsample_block(
                in_channels       = intermediate_channels[i], 
                skip_channels     = intermediate_channels[i-1], 
                out_channels      = intermediate_channels[i-1], 
                upsample_s_factor = pool_k_size, 
                conv_k_size       = conv_k_size, 
                conv_stride       = conv_stride))
            
    def forward(self, x):
        # pass x through the first conv layers 'first_conv'
        x = self.first_conv(x)

        # a list for keeping skip connections
        skip_connections = []

        # apply encoder blocks in self.enc in a for loop
        for enc_block in self.enc:
            skip_connections.append(x)
            x = enc_block(x)
        
        # go through decoder blocks and the reversed skip connections list, and apply decoder block
        for dec_block, s_conn in zip(self.dec, reversed(skip_connections)):
            x = dec_block(x, s_conn)
        
        # pass through the final conv layer
        return self.last_conv(x)