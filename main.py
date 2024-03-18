import torch
from model import downsample_block, upsample_block, u_net, shift_mask

def test_downsample_block(in_channels, out_channels, fbins, T, pool_k_size, conv_k_size, conv_stride, ts_conv):

    db = downsample_block(in_channels, out_channels, pool_k_size, conv_k_size, conv_stride, ts_conv)

    input = torch.rand(in_channels, fbins, T)
    output = db(input)
    assert (tuple(output.shape) == (out_channels, fbins//2, T)), f"ERROR: Shape mismatch in downsample_block! Expected {(out_channels, fbins//2, T)}, got {tuple(output.shape)}!"


def test_upsample_block(nbatch, in_channels, out_channels, skip_channels, fbins, T, upsample_s_factor, conv_k_size, conv_stride, ts_conv):
    ub = upsample_block(in_channels, skip_channels, out_channels, upsample_s_factor, conv_k_size, conv_stride, ts_conv)

    input     = torch.rand(nbatch, in_channels, fbins, T)
    skip_conn = torch.rand(nbatch, skip_channels, fbins * 2, T)
    output    = ub(input, skip_conn)
    assert (tuple(output.shape) == (nbatch, out_channels, fbins*2, T)), f"ERROR: Shape mismatch in upsample_block! Expected {(nbatch, out_channels, fbins*2, T)}, got {tuple(output.shape)}!"

def test_time_shift():

    input = torch.Tensor([
        [[1, 2, 3], 
         [4, 5, 6], 
         [7, 8, 9]],
        [[10, 11, 12], 
         [13, 14, 15], 
         [16, 17, 18]],
        [[19, 20, 21], 
         [22, 23, 24], 
         [25, 26, 27]],
        [[28, 29, 30], 
         [31, 32, 33], 
         [34, 35, 36]]])

    expected_output = torch.Tensor([
        [[2, 3, 0], 
         [5, 6, 0], 
         [8, 9, 0]],
        [[0, 10, 11], 
         [0, 13, 14], 
         [0, 16, 17]],
        [[19, 20, 21], 
         [22, 23, 24], 
         [25, 26, 27]],
        [[28, 29, 30], 
         [31, 32, 33], 
         [34, 35, 36]]])

    C, _, T = input.shape[-3:]
    mask = shift_mask(n_channels = C, T = T, shift_left_idxs = range(0, C//4), shift_right_idxs = range(C//4, C//2), dtype = input.dtype)

    shifted = input @ mask
    assert torch.equal(expected_output, shifted), f"ERROR: Shift did not work properly! Expected\n{str(expected_output)}\ngot\n{str(shifted)}!"
    


def test_u_net(nbatch, in_out_channels, fbins, T, intermediate_channels, pool_k_size, conv_k_size, conv_stride, ts_conv):

    u_net_ = u_net(
        in_out_channels, 
        fbins, 
        intermediate_channels,
        pool_k_size,
        conv_k_size,
        conv_stride,
        ts_conv)
    pytorch_total_params = sum(p.numel() for p in u_net_.parameters() if p.requires_grad)
    print (f"Created a {('Conv2d' if not ts_conv else 'TSConv')} UNET with **{pytorch_total_params}** trainable parameters.")
    input  = torch.rand(nbatch, in_out_channels, fbins, T)
    output = u_net_(input)
    
    assert output.shape == input.shape, f"ERROR: Shape mismatch between input and output in UNET! Expected {input.shape}, got {output.shape}!"

    


# Tests
tests_dict = {
    'downsample_block_Conv2d': [test_downsample_block, {'in_channels': 4, 'out_channels': 8, 'fbins': 256, 'T':  100, 'pool_k_size': (2, 1), 'conv_k_size': (3, 3), 'conv_stride': (1, 1), 'ts_conv': False}], 
    'downsample_block_TSConv': [test_downsample_block, {'in_channels': 4, 'out_channels': 8, 'fbins': 256, 'T':  100, 'pool_k_size': (2, 1), 'conv_k_size': (3, 3), 'conv_stride': (1, 1), 'ts_conv': True}],
    'upsample_block_Conv2d':   [test_upsample_block,   {'nbatch': 1, 'in_channels': 16, 'out_channels': 8, 'skip_channels': 8, 'fbins': 8, 'T' : 100, 'upsample_s_factor': (2, 1), 'conv_k_size': (3, 3), 'conv_stride': (1, 1), 'ts_conv': False}],
    'upsample_block_TSConv':   [test_upsample_block,   {'nbatch': 1, 'in_channels': 16, 'out_channels': 8, 'skip_channels': 8, 'fbins': 8, 'T' : 100, 'upsample_s_factor': (2, 1), 'conv_k_size': (3, 3), 'conv_stride': (1, 1), 'ts_conv': True}],
    'u_net_Conv2d':            [test_u_net,            {'nbatch': 10, 'in_out_channels': 1, 'fbins': 256, 'T': 100, 'intermediate_channels': [4, 4, 4, 8, 8, 16], 'pool_k_size': (2, 1), 'conv_k_size': (3, 3), 'conv_stride': (1, 1), 'ts_conv': False}],
    'u_net_TSConv':            [test_u_net,            {'nbatch': 10, 'in_out_channels': 1, 'fbins': 256, 'T': 100, 'intermediate_channels': [4, 4, 4, 8, 8, 16], 'pool_k_size': (2, 1), 'conv_k_size': (3, 3), 'conv_stride': (1, 1), 'ts_conv': True}],
    'time_shift':              [test_time_shift,       {}]

}


if __name__ == "__main__":

    for test, (f, kwargs) in tests_dict.items():
        try: 
            f(**kwargs)
            print (f"Test {test} successful!")
        except AssertionError as e:
            print (e)
            print (f"Test {test} Failed!")
        print ('~~ ~~ ~~ '*5 + '\n')