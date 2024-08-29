## Overview
In this project, we implement a streamlined U-Net architecture using PyTorch 2.2.1. The implementation features Conv2d layers and a custom convolution layer, CustConv, designed to minimize the number of parameters.

The U-Net architecture takes an input tensor of shape [256, T, 1] and outputs a tensor of the same shape. Below is the list of all tensor dimensionalities throughout the network:

[256, T, 1] → [256, T, 4] → [128, T, 4] → [64, T, 4] → [32, T, 8] → [16, T, 8] → [8, T, 16] → [16, T, 8] → [32, T, 8] → [64, T, 4] → [128, T, 4] → [256, T, 4] → [256, T, 1]

The architecture includes two types of convolution operations:

* Standard Conv2d - Regular 2D convolution layers.
* CustConv - A custom convolution operator designed to simplify the model by reducing kernel size while maintaining the same receptive field.

## CustConv: Some Implementation Details
We have used matrix multiplication for implementing time shift in a channel. Let us assume we have data for a single channel in the form of $X \in \mathbb{R}^{f \times T}$ which is matrix with $f$ rows and $T$ columns. Denoting the $t$'th column as $x_t$, we can write X as:

$$
X = \begin{bmatrix} x_1 & x_2 & \cdots & x_T \end{bmatrix}
$$

In our use case, $x_t$ is the values for different frequency bins for time step $t$.

Now consider the following mask matrix $M_r \in \mathbb{R}^{T \times T}$, which is a square matrix of size $T$:

$$
M_r = \begin{bmatrix} 
    0      & 1 & 0 & \dots  & 0\\
    0      & 0 & 1 & \dots  & 0\\
    \vdots & \vdots & \vdots & \ddots  & \vdots\\
    0      & 0 & 0 & \dots  & 1 \\
    0      & 0 & 0 & \dots  & 0 
    \end{bmatrix},
$$

where the first column is all $0$'s and the right $T-1$ columns are the left $T-1$ columns of a $T \times T$ identity matrix. 

By right multiplying $X$ with $M_r$, we would get

$$
XM_r = \begin{bmatrix} 0 & x_1 & x_2 & \cdots & x_{T-1}\end{bmatrix}.
$$

So right multiplication by $M_r$ results in a shift to the right. Similarly, consider the matrix $M_l \in \mathbb{R}^{T \times T}$ of the form:

$$
M_l = \begin{bmatrix} 
    0      & 0 & 0 & \dots  & 0 & 0\\
    1      & 0 & 0 & \dots  & 0 & 0\\
    0      & 1 & 0 & \dots  & 0 & 0\\
    \vdots & \vdots & \vdots & \ddots & \vdots & \vdots\\
    0      & 0 & 0 & \dots  & 1 & 0 \\
    \end{bmatrix},
$$

where the rightmost column is all $0$'s and the left $T-1$ columns are the right $T-1$ columns of a $T \times T$ identity matrix.

Right multiplying $X$ by $M_l$ results in:

$$
XM_l = \begin{bmatrix} x_2 & \cdots & x_T & 0\end{bmatrix}.
$$

In this project, we use the same mechanics for implementing time shifts. If a \pytorchb tensor $X$ has the shape $C \times f \times T$, we create a mask $\mathbf{Mask}$ of size $C \times T \times T$ where:
\begin{itemize}
    \item if channel $c$ is static, $\mathbf{Mask}[c, :, :] = I_{T}$, where $I_{T}$ is a $T \times T$ identity matrix,
    \item if channel $c$ needs to be left shifted, $\mathbf{Mask}[c, :, :] = M_l$,
    \item and finally if channel $c$ needs to be right shifted, $\mathbf{Mask}[c, :, :] = M_r$.
\end{itemize}
Afterwards, we get the shifted version of the input by performing the matrix operation $X \mathbf{Mask}$.

To create $M_l$ and $M_r$, I use the following functions:
```
    def shift_right_mask(n, dtype):
        mask  = torch.roll(torch.eye(n, dtype=dtype), shifts=[1], dims=[1])
        mask[:, 0] = 0
        return mask

    def shift_left_mask(n, dtype):
        mask = torch.roll(torch.eye(n, dtype=dtype), shifts=[-1], dims=[1])
        mask[:, -1] = 0
        return mask
```

which shift identity matrices to either left or right and then set the "rolled" column to zero.

Their work is then aggregated to create the final mask:
```
    def shift_mask (n_channels, T, shift_left_idxs, shift_right_idxs, dtype):
        mask = torch.stack([torch.eye(T, dtype = dtype) for _ in range(n_channels)])
        mask[shift_left_idxs]  = shift_left_mask(T, dtype)
        mask[shift_right_idxs] = shift_right_mask(T, dtype)
    
        return mask
```

where a mask of all identity matrices gets created, and at the indeces where we want to shift left or right, we plant $M_l$ or $M_r$ accordingly.
