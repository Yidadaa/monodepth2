from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GeometricTnfAffine(nn.Module):
    def __init__(self, h: int, w: int, scale_factor: float = 1.0):
        '''Make transformation with affine matrix.

        Args:
            h, w: height and width of grid
            scale_factor: scale factor, default is `1.0`
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

    def affine_grid(self, theta: torch.Tensor) -> torch.Tensor:
        '''Make sampling grid.

        Args:
            theta: torch.Tensor with shape (n, 2, 3)

        Returns:
            sampling_grid: torch.Tensor with shape (n, h_out, w_out, 2)
        '''
        # shape: (n, 1, 1, 2, 3)
        theta = theta.view((-1, 1, 1, 2, 3)).contiguous()
        sampling_grid = theta @ self.grid  # shape: (n, h, w, 2, 1)
        sampling_grid = sampling_grid.squeeze(-1)  # shape: (n, h, w, 2)
        return sampling_grid

    def forward(self, images: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        '''Do transformation: P = Theta @ [X, Y, 1].T

        Args:
            images: torch.Tensor with shape (n, c, h, w)
            theta: torch.Tensor with shape (n, 2, 3)

        Returns:
            transformed_images: torch.Tensor with shape (n, c, h_out, w_out)
        '''
        # make sampling grid
        sampling_grid = self.affine_grid(theta)

        # sample transformed images
        transformed_images = F.grid_sample(images, sampling_grid)

        return transformed_images


class GeometricGridLoss(nn.Module):
    '''Measure loss on imaginary grid of points.
    Reference: "Convolutional neural network architecture for geometric matching." Section 4.1.
    '''
    def __init__(self, h: int, w: int):
        super(GeometricGridLoss, self).__init__()
        self.TransformPoint = GeometricTnfAffine(h, w)

    def forward(self, theta: torch.Tensor, theta_gt: torch.Tensor):
        '''Compute the loss of L(theta, theta_gt).

        Args:
            theta: shape: (n, 2, 3)
            theta_gt: shape: (n, 2, 3)

        Returns:
            loss: mes loss between two transformed grids.
        '''
        sampled_grid = self.TransformPoint.affine_grid(theta) # shape: (n, h, w, 2)
        sampled_grid_gt = self.TransformPoint.affine_grid(theta_gt) # shape: (n, h, w, 2)

        return F.mse_loss(sampled_grid, sampled_grid_gt)


class WeakInlinerLoss(nn.Module):
    '''Alignment loss for learning semantic correspondences.
    Reference:  End-to-end Weakly supervsised semantic alignment.
    '''
    def __init__(self): 
        super(WeakInlinerLoss, self).__init__()


class CycleTracking(nn.Module):
    def __init__(self, pretrained: bool = True, temporal_out: int = 4):
        '''Cycle tracking module.

        Args:
            temporal_out: the length of temporal sequence
        '''
        super(CycleTracking, self).__init__()

        self.scale_factor = 1 / (512 ** 0.5)
        self.image_feature_size = 30
        self.patch_feature_size = 10
        self.scale = self.image_feature_size // self.patch_feature_size

        self.transform_predictor = nn.Sequential([
            nn.Conv2d(self.image_feature_size ** 2, 128,
                        kernel_size=4, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=4, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 4 * 4, 3)
        ])  # type: nn.Module

        self.GeoSampler = GeometricTnfAffine(self.image_feature_size, self.image_feature_size)
        self.GridLoss = GeometricGridLoss(self.patch_feature_size, self.patch_feature_size)
        self.WeakInlierLoss = WeakInlinerLoss()


    def forward(self, image_feats: List[torch.Tensor], patch_feats: torch.Tensor,
            thetas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Do cycle tracking.

        Args:
            image_feats: list (len == t) of tensors with shape (n, c, h, w), default: t = 3, h = w = 512
            patch_feats: the patches from image_feats[0], shape (n, c, h // s, w // s)
            thetas: transofmations from images to patches, shape (n, 2, 3)

        Returns:
            forward_transform_thetas: shape: (t, n, 2, 3)
            theta_mat_base_img_to_tgt_patch: shape: (n, 2, 3)
            theta_mat_tgt_img_to_base_patch: shape: (n, 2, 3)
            corr_mat_base_img_to_tgt_patch: shape: (n, t * sh *sw, h, w)
        '''
        image_feats = [torch.unsqueeze(feat, 0) for feat in image_feats] # shape: (t, n, c, sh, sw)
        image_feats = torch.cat(image_feats).transpose(0, 1).transpose(1, 2)  # shape: (n, c, t, sh, sw)
        image_feats = F.relu(image_feats)

        # normalize per image feature map along channel dim
        image_feats_norm = F.normalize(image_feats, p=2, dim=1)

        base_img_feats = image_feats[1:] # shape: (n, c, t, sh, sw)
        base_img_feats_norm = image_feats_norm[1:] # shape: (n, c, t, sh, sw)

        target_img_feats = image_feats[0].squeeze(dim=2) # shape: (n, c, sh, sw)
        target_img_feats_norm = image_feats[0].squeeze(dim=2) # shape: (n, c, sh, sw)

        patch_feats = F.relu(patch_feats) # shape: (n, c, h, w)
        patch_feats_norm = F.normalize(patch_feats, p=2, dim=1)

        # predict the transformation from base images to target patches
        corr_mat_base_img_to_tgt_patch = self.compute_correlation_patche2images(
            patch_feats_norm, base_img_feats_norm)  # shape: (n, t * sh *sw, h, w)
        theta_base_img_to_tgt_patch = self.transform_predictor(corr_mat_base_img_to_tgt_patch) # shape: (n * t, 3)
        theta_mat_base_img_to_tgt_patch = self.transform_vector2mat(theta_base_img_to_tgt_patch) # shape: (n * t, 2, 3)

        # sample predicted patches from base image features, got sampled base patches
        n, c, t, sh, sw = base_img_feats.shape
        base_feats_ori = base_img_feats.transpose(1, 2).view(n * t, c, sh, sw).contiguous() # shape: (n * t, c, h, w)
        base_sampled_patch_feats = self.GeoSampler(base_feats_ori, theta_mat_base_img_to_tgt_patch) # shape: (n * t, c, h, w)

        # do skip prediction, predict the transformation from target image to sampled base patches
        corr_mat_tgt_img_to_base_patch = self.compute_correlation_patches2image(
            base_sampled_patch_feats, target_img_feats_norm) # shape: (n, t * sh * sw, h, w)
        theta_tgt_img_to_base_patch = self.transform_predictor(corr_mat_tgt_img_to_base_patch) # shape: (n * t, 3)
        theta_mat_tgt_img_to_base_patch = self.transform_vector2mat(theta_tgt_img_to_base_patch) # shape: (n * t, 2, 3)

        # cycle tracking
        transform_thetas = [] # type: List[torch.Tensor], shape: [(t, n, 2, 3)]
        base_img_transposed = base_img_feats.transpose(1, 2) # shape: (n, t, c, sh, sw)
        
        for i in range(1, t + 1):
            # cycle, forward and backward
            # target -> base img 1 -> base img 2 -> ... -> base img i
            # target <- base img 1 <- base img 2 <- ... <- base img i
            #         |
            #    closure_theta
            forw_transform_thetas, last_forw_patches = \
                self.recurrent_align(patch_feats_norm, base_img_transposed[:, :i], base_img_feats_norm[:, :, :i])
            _, last_back_patches = \
                self.recurrent_align(last_forw_patches, base_img_transposed[:, i - 1:-1:-1], base_img_feats_norm[:, :, i - 1:-1:-1])
            # 
            closure_theta, _ = self.track_and_sample(last_back_patches, target_img_feats, target_img_feats_norm)

            transform_thetas += [forw_transform_thetas, closure_theta.unsqueeze(dim=0)]
            # TODO: assure the usage of forw_transform_thetas

        forward_transform_thetas = torch.cat(transform_thetas, dim=0) # shape: (t, n, 2, 3)

        outputs = (
            forward_transform_thetas, # shape: (t, n, 2, 3)
            thetas, # shape: (n, 2, 3)
            theta_mat_base_img_to_tgt_patch, # shape: (n * t, 2, 3)
            theta_mat_tgt_img_to_base_patch, # shape: (n * t, 2, 3)
            corr_mat_base_img_to_tgt_patch # shape: (n, t * sh * sw, h, w)
        )
        
        return outputs


    def compute_loss(self,
        forward_transform_thetas: torch.Tensor,
        thetas: torch.Tensor,
        theta_mat_base_img_to_tgt_patch: torch.Tensor,
        theta_mat_tgt_img_to_base_patch: torch.Tensor,
        corr_mat_base_img_to_tgt_patch: torch.Tensor) -> torch.Tensor:
        '''Loss function.
        '''
        t, n = forward_transform_thetas.shape[:2]

        # TODO: what's this?
        indexs = list(range(t))
        indexs = [j for j in [sum(indexs[:i]) - 1 for i in indexs][2:] if j < t]
        loss_target_theta = [self.GridLoss(forward_transform_thetas[i], theta) for i in indexs]

        # skip align loss
        theta_expanded = thetas.unsqueeze(1).repeat(1, t, 1, 1).view(-1, 2, 3) # shape: (n * t, 2, 3)
        loss_target_skip = self.GridLoss(theta_mat_base_img_to_tgt_patch, thetas)

        # inliner loss
        loss_inliner = self.WeakInlierLoss(corr_mat_base_img_to_tgt_patch, theta_mat_tgt_img_to_base_patch) # type: torch.Tensor
        loss_inliner = -loss_inliner.mean()

        return loss_target_theta, loss_target_skip, loss_inliner


    def recurrent_align(self, current_patches: torch.Tensor, middle_imgs: torch.Tensor,
        middle_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Do recurrent tracking.

        Args:
        current_patches: initial patch to track, shape = (n, c, h, w)
        middle_imgs: `t` images to track through, shape = (n, t, c, sh, sw)
        middle_norm: `F.normalize(middle_imgs, p=2, dim=1)`, shape = (n, c, t, sh, sw)

        Returns:
        transform_theta: the transformation matrixs between each two frames, frame_i -> patch_i, shape: (t, n, 2, 3)
        last_patches: the last patches sampled from last frame, used to do cycle back tracking, shape: (n, c, h, w)
        '''
        n, t = middle_imgs.shape[:2]
        transform_thetas = torch.zeros(t, n, 2, 3)

        for i in range(t):
        current_base_feat = middle_imgs[:, i].squeeze(dim=1) # shape: (n, c, sh, sw)
        current_base_norm = middle_norm[:, :, i] # shape: (n, c, 1, sh, sw)
        current_theta_mat, current_patches = self.track_and_sample(current_patches, current_base_feat, current_base_norm)
        # save each step's transform
        transform_thetas[i] = current_theta_mat

        return transform_thetas, current_patches


    def track_and_sample(self, patches: torch.Tensor, base_feat: torch.Tensor, \
            base_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Predict the transformation: base frames -> patches.

        Args:
            patches: shape: (n * t, c, h, w), default: `t = 1`
            base_feat: shape: (n * t, c, sh, sw)
            base_norm: shape: (n * t, c, sh, sw)

        Returns:
            theta_mat: transformation matrix, shape: (n * t, 2, 3), default: `t = 1`
            patches_norm: sampled normalized patches, shpae: (n * t, c, h, w)
        '''
        # compute transformation: base feat norm -> current patch
        corr_mat = self.compute_correlation_patche2images(patches, base_norm)
        theta_vector = self.transform_predictor(corr_mat)
        theta_mat = self.transform_vector2mat(theta_vector) # shape: (n, 2, 3)
        # sample next patch: base feat -> next patch -> normalized
        patch_feat = self.GeoSampler(base_feat, theta_mat) # shape: (n, c, h, w)
        patches_norm = F.normalize(patch_feat, p=2, dim=1)

        return theta_mat, patches_norm


    def compute_correlation_patche2images(self, patch: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        '''Compute correlation map between patch and images (with time length of `t`) feature maps.

        Args:
            patch: patch features, shape: (n, c, h, w)  
            images: image features, shape: (n, c, t, h * s, w * s), `s` denotes scale factor  

        Returns:
            corr_feat: correlation matrix, shape: (n, t * sh * sw, h, w)
        '''
        h, w = patch.shape[-2:]
        n, c, t, sh, sw = images.shape

        patch_vector = patch.view(n, c, -1)  # shape: (n, c, h * w)
        images_vector = images.view(n, c, -1).transpose(1, 2)  # shape: (n, t * sh * sw, c)

        # shape: (n, t * sh * sw, h * w)
        corr_feat = torch.matmul(images_vector, patch_vector)

        # shape: (n, t, sh * sw, h, w)
        corr_feat = corr_feat.view(n, t, sh * sw, h, w)
        # compute softmax along sh * sw dim
        corr_feat = F.softmax(corr_feat, dim=2)

        return corr_feat.view(n, t * sh * sw, h, w)


    def compute_correlation_patches2image(self, patches: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        '''Compute correlation map between patches (with time length of `t`) and image feature maps.

        Args:
            patches: patch features, shape: (n, c, t, h, w)  
            image: image features, shape: (n, c, h * s, w * s), `s` denotes scale factor  

        Returns:
            corr_feat: correlation matrix, shape: (n, t * sh * sw, h, w)
        '''
        h, w = patches.shape[-2:]
        n, c, t, sh, sw = image.shape

        patches_vector = patches.transpose(n, c, -1).transpose(1, 2)  # shape: (n, t * h * w, c)
        image_vector = image.transpose(2, 3).view(n, c, -1)  # shape: (n, c, sw * sh)

        # shape: (n, t * h * w, sw * sh)
        corr_feat = torch.matmul(patches_vector, image_vector)

        # shape: (n, t, h * w, sh * sw)
        corr_feat = corr_feat.view(n, t, h * w, sh * sw)
        # compute softmax along sh * sw dim
        corr_feat = F.softmax(corr_feat, dim=3)
        corr_feat = corr_feat.transpose(2, 3) # shape: (n, t, sh * sw, h * w)

        return corr_feat.view(n, t * sh * sw, h, w)


    def transform_vector2mat(self, transform_vector: torch.Tensor):
        '''Convert transform vector (3, ) to matrix (2 * 3).

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