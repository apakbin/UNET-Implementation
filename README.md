## Overview
In this project, we implement a streamlined U-Net architecture using PyTorch 2.2.1. The implementation features Conv2d layers and a custom convolution layer, CustConv, designed to minimize the number of parameters.

The U-Net architecture takes an input tensor of shape [256, T, 1] and outputs a tensor of the same shape. Below is the list of all tensor dimensionalities throughout the network:

[256, T, 1] → [256, T, 4] → [128, T, 4] → [64, T, 4] → [32, T, 8] → [16, T, 8] → [8, T, 16] → [16, T, 8] → [32, T, 8] → [64, T, 4] → [128, T, 4] → [256, T, 4] → [256, T, 1]
