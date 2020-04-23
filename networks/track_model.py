from typing import List

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
        super(GeometricTnfAffine, self).__init__()

        self.h, self.w = h, w
        self.grid = np.meshgrid(np.linspace(-1, 1, w),
                                np.linspace(-1, 1, h))  # shape: (2, h, w)
        self.grid = torch.Tensor(self.grid, reuqires_grad=False).transpose(
            0, -1)  # shape: (h, w, 2)
        self.grid /= scale_factor

        # convert to homogeneous coordinates
        self.grid = torch.cat((self.grid, torch.ones(
            h, w, 1)), -1).unsqueeze(-1)  # shape: (h, w, 3, 1)

    def forward(self, images: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        '''Make transition: P = Theta @ [X, Y, 1].T

        Args:
          images: torch.Tensor with shape (n, 3, h, w)
          theta: torch.Tensor with shape (n, 3)

        Returns:
          position: torch.Tensor with shape (n, h, w, 2)
        '''
        # make sampling grid
        # shape: (n, 1, 1, 2, 3)
        theta = theta.view((-1, 1, 1, 2, 3)).contiguous()
        sampling_grid = theta @ self.grid  # shape: (n, h, w, 2, 1)
        sampling_grid = sampling_grid.squeeze(-1)  # shape: (n, h, w, 2)

        # sample transformed images
        transformed_images = F.grid_sample(images, sampling_grid)

        return transformed_images


class CycleTracking(nn.Module):
    def __init__(self, pretrained: bool = True, temporal_out: int = 4):
        '''Cycle tracking module.

        Args:
          temporal_out: TODO: what's this?
        '''
        super(CycleTracking, self).__init__()

        self.scale_factor = 1 / (512 ** 0.5)
        self.image_feature_size = 30
        self.patch_feature_size = 10
        self.scale = self.image_feature_size // self.patch_feature_size

        self.transition_predictor = nn.Sequential([
            nn.Conv2d(self.image_feature_size ** 2, 128,
                      kernel_size=4, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=4, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Linear(64 * 4 * 4, 3)
        ])  # type: nn.Module

    def forward(self, image_feats: List[torch.Tensor], patch_feats: torch.Tensor, thetas: torch.Tensor):
        '''Do cycle tracking.

        Args:
          image_feats: list (len == t) of tensors with shape (n, c, h, w), default: t = 3, h = w = 512
          patch_feats: the patches from image_feats[0], shape (n, c, h // s, w // s)
          thetas: transofmations from images to patches, shape (n, 2, 3)
        '''
        image_feats = [torch.unsqueeze(feat, 0) for feat in image_feats]
        image_feats = torch.cat(image_feats).transpose(
            0, 1)  # shape: (n, t, c, h, w)

        image_feats = F.relu(image_feats)
        # normalize per image feature map
        image_feats_norm = F.normalize(image_feats, p=2, dim=1)

        patch_feats = F.relu(patch_feats)
        patch_feats_norm = F.normalize(patch_feats, p=2, dim=0)

        correlation_mat = self.compute_correlation_softmax(
            patch_feats_norm, image_feats_norm[1:])  # shape: (n, t * sh *sw, h, w)

        theta_pred = self.transition_predictor()

    def compute_correlation_softmax(self, query: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        '''Compute correlation map between query and base feature maps.

        Args:
          query: patch features, shape: (n, c, h, w)
          base: image features, shape: (n, t, c, h * s, w * s), `s` denotes scale factor

        Returns:
          corr_feat: correlation matrix, shape: (n, t * sh * sw, h, w)
        '''
        h, w = query.shape[-2:]
        n, t, c, sh, sw = base.shape

        base_vector = base.transpose(1, 2).view(
            n, c, -1).transpose(1, 2)  # shape: (n, t * sh * sw, c)
        query_vector = query.transpose(n, c, -1)  # shape: (n, c, h * w)

        # shape: (n, t * sh * sw, h * w)
        corr_feat = torch.matmul(base_vector, query_vector)

        # shape: (n, t, sh * sw, h, w)
        corr_feat = corr_feat.view(n, t, sh * sw, h, w)
        # compute softmax along sh * sw dim
        corr_feat = F.softmax(corr_feat, dim=2)

        return corr_feat.view(n, t * sh * sw, h, w)

    def transform_vector2mat(self, transform_vector: torch.Tensor):
        '''Convert transform vector (3 * 1) to matrix (2 * 3).

        Args:
          transform_vector: shape: (n, 3)

        Returns:
          transform_matrix: shape: (n, 2, 3)
        '''
        n, s = transform_vector.shape[0],  1 / self.scale
        theta = transform_vector[:, 2]
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

        tm = torch.zeros(
            (n, 2, 3), requires_grad=transform_vector.requires_grad)

        tm[:, 0, 0], tm[:, 0, 1], tm[:, 0, 2], \
        tm[:, 1, 0], tm[:, 1, 1], tm[:, 1, 2] = \
            s * cos_theta, -s * sin_theta, transform_vector[:, 0], \
            s * sin_theta, -s * cos_theta, transform_vector[:, 1]

        return tm
