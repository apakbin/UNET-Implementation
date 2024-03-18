import torch
from model import downsample_block, upsample_block, u_net, shift_mask

if __name__ == "__main__":

    C = 10
    fbins = 32
    T     = 15
    test  = torch.randint(0, 100, (C, fbins, T))

    mask = shift_mask(C, T, [1, 3, 4], [6, 7], test.dtype)

    shifted = test @ mask

    for c in range(C):
        print (c)
        print (test[c,:,:])
        print (shifted[c, :, :])
        print ('- - - ')

    in_channels  = 4
    out_channels = 8
    fbins        = 256
    T            = 100
    pool_k_size  = (2, 1)
    conv_k_size  = (3, 3)
    conv_stride  = (1, 1)
    db = downsample_block(in_channels, out_channels, pool_k_size, conv_k_size, conv_stride, ts_conv=True)

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
    ub = upsample_block(in_channels, skip_channels, out_channels, upsample_s_factor, conv_k_size, conv_stride)

    input = torch.rand(nbatch, in_channels, fbins, T)
    skip_conn = torch.rand(nbatch, skip_channels, fbins * 2, T)
    output = ub(input, skip_conn)
    assert (tuple(output.shape) == (nbatch, out_channels, fbins*2, T)), f"ERROR: Shape mismatch in UownsampleBlock! Expected {(nbatch, out_channels, fbins*2, T)}, got {tuple(output.shape)}!"

    print ('- - - - - - - - - - - ')
    # TEST ALL:
    in_out_channels       = 1
    fbins                 = 256
    T                     = 100
    intermediate_channels = [4, 4, 4, 8, 8, 16]
    pool_k_size           = (2, 1)
    conv_k_size           = (3, 3)
    conv_stride           = (1, 1)
    ts_conv               = False
    u_net_ = u_net(
        in_out_channels, 
        fbins, 
        intermediate_channels,
        pool_k_size,
        conv_k_size,
        conv_stride,
        ts_conv)
    pytorch_total_params = sum(p.numel() for p in u_net_.parameters() if p.requires_grad)
    nbatch = 10
    input_data = torch.rand(nbatch, in_out_channels, fbins, T)
    print (u_net_(input_data).shape)
    
    print (pytorch_total_params)