import torch
from model import DownsampleBlock, UpsampleBlock

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
    assert (tuple(output.shape) == (out_channels, fbins//2, T)), f"ERROR: Shape mismatch in DownsampleBlock! Expected {(out_channels, fbins//2, T)}, got {tuple(output.shape)}!"


    nbatch            = 1
    in_channels       = 16
    out_channels      = 8
    skip_channels     = 8
    fbins             = 8
    T                 = 100
    pool_k_size       = (2, 1)
    upsample_s_factor = (2, 1)
    conv_k_size       = (3, 3)
    conv_stride       = (1, 1)
    ub = UpsampleBlock(in_channels, skip_channels, out_channels, upsample_s_factor, conv_k_size, conv_stride)

    input = torch.rand(nbatch, in_channels, fbins, T)
    skip_conn = torch.rand(nbatch, skip_channels, fbins * 2, T)
    output = ub(input, skip_conn)
    assert (tuple(output.shape) == (nbatch, out_channels, fbins*2, T)), f"ERROR: Shape mismatch in UownsampleBlock! Expected {(nbatch, out_channels, fbins*2, T)}, got {tuple(output.shape)}!"