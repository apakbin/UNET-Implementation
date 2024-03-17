import torch
from model import DownsampleBlock

if __name__ == "__main__":


    in_channels  = 4
    out_channels = 8
    fbins        = 256
    T            = 100
    pool_k_size  = (2, 1)
    conv_k_size  = (3, 3)
    conv_stride  = (1, 1)
    db = DownsampleBlock(in_channels, out_channels, pool_k_size, conv_k_size, conv_stride)

    input = torch.rand(in_channels, fbins, T)
    output = db(input)
    assert (tuple(output.shape) == (out_channels, fbins//2, T)), "ERROR: Shape mismatch in DownsampleBlock!"