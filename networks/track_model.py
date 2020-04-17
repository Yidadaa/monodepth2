import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GeometricTnfAffine(nn.Module):
  def __init__(self, h: int, w: int, scale_factor: float = 1.0):
    '''Make transformation with affine matrix.

    Args:
      h, w: height and width of grid
      scale_factor: scale factor
    '''
    self.h, self.w = h, w
    self.grid = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h)) # shape: (2, h, w)
    self.grid = torch.Tensor(self.grid, reuqires_grad=False).transpose(0, -1) # shape: (h, w, 2)
    self.grid /= scale_factor

    # convert to homogeneous coordinates
    self.grid = torch.cat((self.grid, torch.ones(h, w, 1)), -1).unsqueeze(-1) # shape: (h, w, 3, 1)


  def forward(self, images: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    '''Make transition: P = Theta @ [X, Y, 1].T

    Args:
      images: torch.Tensor with shape (n, 3, h, w)
      theta: torch.Tensor with shape (n, 3)

    Returns:
      position: torch.Tensor with shape (n, h, w, 2)
    '''
    # make sampling grid
    theta = theta.view((-1, 1, 1, 2, 3)).contiguous() # shape: (n, 1, 1, 2, 3)
    sampling_grid = theta @ self.grid # shape: (n, h, w, 2, 1)
    sampling_grid = sampling_grid.squeeze(-1) # shape: (n, h, w, 2)

    # sample transformed images
    transformed_images = F.grid_sample(images, sampling_grid)

    return transformed_images