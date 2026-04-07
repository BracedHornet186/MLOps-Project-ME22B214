from typing import List, Optional

import kornia.augmentation
import numpy as np
import torch
import torchvision.transforms as T
from kornia.geometry.transform import (get_affine_matrix2d, warp_affine,
                                       warp_perspective)
from kornia.geometry.linalg import transform_points
from kornia.utils import image_to_tensor


class HomographyAdaptation:
    def __init__(self, height: int, width: int,
                 angles: List[int],
                 device: Optional[torch.device] = None):
        device = device or torch.device('cpu')

        self.height = height
        self.width = width
        self.angles = angles
        self.device = device
        self._dummy_tensor = torch.ones(1).to(self.device)

        self.base_aug = kornia.augmentation.RandomAffine(90) # Dummy params

        matrix_list = []
        inv_matrix_list = []
        for angle in self.angles:
            size = (1, self.height, self.width)
            params = self.base_aug.generate_parameters(size)    # type: ignore
            params['angle'].fill_(angle)

            # NOTE:
            #   mat: Shape(1, 3, 3)
            #   inv_mat: Shape(1, 3, 3)
            mat = self.base_aug.compute_transformation(
                self._dummy_tensor, params=params, flags={}
            )
            inv_mat = self.base_aug.compute_inverse_transformation(mat)

            matrix_list.append(mat)
            inv_matrix_list.append(inv_mat)

        self.matrix_list = matrix_list
        self.inv_matrix_list = inv_matrix_list

        self.to_pil = T.ToPILImage()

    def transform_homography_variants(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Args
        ----
        img : np.ndarray
            Shape(H, W, 3), dtype: np.uint8, RGB order

        Returns
        -------
        np.ndarray
            Shape(#matrix_list, H, W, 3), dtype: np.uint8, RGB order
        """
        x = image_to_tensor(img)[None].float().to(self.device) / 255
        dsize = (self.height, self.width)

        variants = []
        for mat in self.matrix_list:
            #y = warp_affine(x, mat[:, :2, :], dsize)
            y = warp_perspective(x, mat, dsize)
            y = np.array(self.to_pil(y[0].cpu()))
            variants.append(y)

        return np.stack(variants)

    def transform_homography_variants_tensor(
        self,
        img: torch.Tensor
    ) -> torch.Tensor:
        """
        Args
        ----
        img : torch.Tensor
            Shape(1, 3, H, W), dtype: torch.uint8, RGB order

        Returns
        -------
        torch.Tensor
            Shape(#matrix_list, 3, H, W), dtype: torch.uint8, RGB order
        """
        dsize = (self.height, self.width)

        assert img.shape[0] == 1

        variants = []
        for mat in self.matrix_list:
            y = warp_perspective(img, mat, dsize)
            y = y.squeeze(0)    # (3, H, W)
            variants.append(y)

        return torch.stack(variants)

    def transform_inverse_homography_variants(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Args
        ----
        img : np.ndarray
            Shape(#matrix_list, H, W, 3), dtype: np.uint8, RGB order

        Returns
        -------
        np.ndarray
            Shape(#matrix_list, H, W, 3), dtype: np.uint8, RGB order
        """
        xs = image_to_tensor(img).float().to(self.device) / 255
        dsize = (self.height, self.width)

        variants = []
        for x, mat in zip(xs, self.inv_matrix_list):
            y = warp_perspective(x[None], mat, dsize)
            y = np.array(self.to_pil(y[0].cpu()))
            variants.append(y)

        return np.stack(variants)

    def inverse_transform_keypoints(
        self,
        kpts: np.ndarray,
        inv_mat: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Args
        ----
        kpts : np.ndarray
            Shape(#keypoints, 2)

        inv_mat : torch.Tensor
            Inverse transformation matrix
        """
        if inv_mat is None:
            assert len(self.inv_matrix_list) == 1
            inv_mat = self.inv_matrix_list[0]
        assert isinstance(inv_mat, torch.Tensor)
        kpts = kpts.copy()
        kpts = kpts[None, :]    # (1, N, 2)
        keypoints = torch.from_numpy(kpts).float().to(self.device)
        inv_projected_keypoints = transform_points(inv_mat, keypoints)
        inv_projected_keypoints = inv_projected_keypoints.cpu().numpy()
        return inv_projected_keypoints[0]
    
    def inverse_transform_keypoints_tensor(
        self,
        kpts: torch.Tensor,
        inv_mat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = kpts.device
        _kpts = self.inverse_transform_keypoints(kpts.cpu().numpy(),
                                                 inv_mat=inv_mat)
        return torch.from_numpy(_kpts).float().to(device, non_blocking=True)
