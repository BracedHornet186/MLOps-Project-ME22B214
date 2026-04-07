import copy
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from sklearn.cluster import MeanShift
from sklearn.neighbors import LocalOutlierFactor

from preprocesses.config import OverlapRegionEstimatorConfig


class OverlapRegionEstimator:
    def __init__(self, conf: OverlapRegionEstimatorConfig):
        self.conf = conf
        self.meanshift_kwargs = {
            "bin_seeding": conf.ms_bin_seeding,
            "cluster_all": conf.ms_cluster_all,
        }
        self.lof_kwargs = {"n_neighbors": conf.lof_n_neighbors}

    def get_paired_bboxes(
        self,
        matched_kpts1: np.ndarray,
        matched_kpts2: np.ndarray,
        shape1: Tuple[int, int],
        shape2: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args
        ----
        matched_kpts1 : np.ndarray
        matched_kpts2 : np.ndarray
            Shape(#kpts, 2) xy order

        shape1: Tuple[int, int]
        shape2: Tuple[int, int]
            (height, width) of image1 and image2

        Returns
        -------
        np.ndarray
            Shape(#bbox, 4), xyxy order
        """
        conf = self.conf

        if len(matched_kpts1) < conf.min_matches_required:
            bboxes1 = whole_image_bboxes(shape1)
            bboxes2 = whole_image_bboxes(shape2)
            return bboxes1, bboxes2

        if conf.method == "meanshift":
            bboxes1, bboxes2 = estimate_overlap_from_matches_meanshift(
                matched_kpts1,
                matched_kpts2,
                bidirectional=conf.bidirectional,
                bidirectional_merging=conf.bidirectional_merging,
                meanshift_kwargs=self.meanshift_kwargs,
            )
        elif conf.method == "lof":
            bboxes1, bboxes2 = estimate_overlap_from_matches_lof(
                matched_kpts1,
                matched_kpts2,
                bidirectional=conf.bidirectional,
                bidirectional_merging=conf.bidirectional_merging,
                lof_kwargs=self.lof_kwargs,
            )
        else:
            raise ValueError(conf.method)

        # Post-process
        bboxes1 = adjust_bboxes(
            bboxes1,
            shape1,
            bbox_expansion=conf.bbox_expansion,
            border_padding=conf.border_padding,
        )
        bboxes2 = adjust_bboxes(
            bboxes2,
            shape2,
            bbox_expansion=conf.bbox_expansion,
            border_padding=conf.border_padding,
        )

        return bboxes1, bboxes2


class Cropper:
    def __init__(
        self,
        bbox: np.ndarray,
        orig_shape: Tuple[int, int],
    ) -> None:
        self.bbox = bbox.astype(np.int32)  # xyxy
        self.orig_shape = orig_shape
        self.orig_img = None
        self.bbox_width = self.bbox[2] - self.bbox[0]
        self.bbox_height = self.bbox[3] - self.bbox[1]
    
    def convert_cropped_to_original_coordinates(
        self,
        kpts: np.ndarray,
    ) -> np.ndarray:
        kpts = kpts.copy()

        kpts[:, 0] = kpts[:, 0] + self.bbox[0]
        kpts[:, 1] = kpts[:, 1] + self.bbox[1]

        return kpts

    def convert_cropped_to_original_coordinates_torch(
        self, kpts: torch.Tensor
    ) -> torch.Tensor:
        kpts[:, 0] = kpts[:, 0] + self.bbox[0]
        kpts[:, 1] = kpts[:, 1] + self.bbox[1]
        return kpts

    def remove_keypoints_out_of_bbox(
        self, kpts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        H = self.orig_shape[0]
        W = self.orig_shape[1]
        kpts[:, 0] = np.clip(kpts[:, 0], 0, W - 1)
        kpts[:, 1] = np.clip(kpts[:, 1], 0, H - 1)

        left = self.bbox[0]
        top = self.bbox[1]
        right = self.bbox[2]
        bottom = self.bbox[3]

        toremoveW = np.logical_or(kpts[:, 0] < left, kpts[:, 0] > right)
        toremoveH = np.logical_or(kpts[:, 1] < top, kpts[:, 1] > bottom)
        toremove = np.logical_or(toremoveW, toremoveH)
        kpts = kpts[~toremove, :]
        keeps = np.where(~toremove)
        return kpts, keeps[0]

    def set_original_image(self, img: Union[np.ndarray, torch.Tensor]) -> "Cropper":
        self.orig_img = copy.deepcopy(img)
        return self

    def crop_ndarray_image_gray(
        self,
        img: np.ndarray,  # Shape(H, W)
    ) -> np.ndarray:
        bbox = self.bbox
        img = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        return img

    def crop_ndarray_image(
        self,
        img: np.ndarray,  # Shape(H, W, 3)
    ) -> np.ndarray:
        bbox = self.bbox
        img = img[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
        return img

    def crop_tensor_image(
        self,
        img: torch.Tensor,  # Shape(3, H, W)
    ) -> torch.Tensor:
        bbox = self.bbox
        img = img[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]
        return img
    
    def crop_keypoints_ndarray(
        self,
        kpts: np.ndarray
    ) -> np.ndarray:
        kpts = kpts.copy()
        kpts, _ = self.remove_keypoints_out_of_bbox(kpts)
        kpts[:, 0] = kpts[:, 0] - self.bbox[0]
        kpts[:, 1] = kpts[:, 1] - self.bbox[1]
        kpts[:, 0] = np.clip(kpts[:, 0], 0, self.bbox_width - 1)
        kpts[:, 1] = np.clip(kpts[:, 1], 0, self.bbox_height - 1)
        return kpts


class OverlapRegionCropper:
    def __init__(
        self, cropper1: Cropper, cropper2: Cropper, enable_scale_alignment: bool = True
    ):
        self.enable_scale_alignment = enable_scale_alignment
        self.cropper1 = cropper1
        self.cropper2 = cropper2
        resize1, resize2 = self.compute_overlap_region_rescaled_size()
        self.resize1 = resize1
        self.resize2 = resize2
    
    def get_bbox_sizes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        bbox1 = self.cropper1.bbox
        bbox2 = self.cropper2.bbox
        w1 = bbox1[2] - bbox1[0]
        h1 = bbox1[3] - bbox1[1]
        w2 = bbox2[2] - bbox2[0]
        h2 = bbox2[3] - bbox2[1]
        return ((h1, w1), (h2, w2))

    def compute_overlap_region_rescaled_size(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        (h1, w1), (h2, w2) = self.get_bbox_sizes()
        if not self.enable_scale_alignment:
            # No rescaling
            return ((h1, w1), (h2, w2))

        longest1 = max(h1, w1)
        longest2 = max(h2, w2)

        if longest1 >= longest2:
            if longest1 == h1:
                # img1 is a vertical image
                base_height = h1
                resize1 = (h1, w1)
                resize2 = (base_height, int(w2 * base_height / h2))
            else:
                # img1 is a horizontal image
                base_width = w1
                resize1 = (h1, w1)
                resize2 = (int(h2 * base_width / w2), base_width)
        else:
            if longest2 == h2:
                # img2 is a vertical image
                base_height = h2
                resize1 = (base_height, int(w1 * base_height / h1))
                resize2 = (h2, w2)
            else:
                # img2 is a horizontal image
                base_width = w2
                resize1 = (int(h1 * base_width / w1), base_width)
                resize2 = (h2, w2)

        return resize1, resize2

    def convert_cropped_to_original_coordinates(
        self,
        kpts1: np.ndarray,
        kpts2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        kpts1, kpts2 = self._unresize_alignment_kpts(kpts1, kpts2)
        return (
            self.cropper1.convert_cropped_to_original_coordinates(kpts1),
            self.cropper2.convert_cropped_to_original_coordinates(kpts2),
        )

    def convert_cropped_to_original_coordinates_torch(
        self, kpts1: torch.Tensor, kpts2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kpts1, kpts2 = self._unresize_alignment_kpts_torch(kpts1, kpts2)
        return (
            self.cropper1.convert_cropped_to_original_coordinates_torch(kpts1),
            self.cropper2.convert_cropped_to_original_coordinates_torch(kpts2),
        )

    def set_original_image(
        self,
        img1: Union[np.ndarray, torch.Tensor],
        img2: Union[np.ndarray, torch.Tensor],
    ) -> "OverlapRegionCropper":
        self.cropper1.set_original_image(img1)
        self.cropper2.set_original_image(img2)
        return self

    def crop_ndarray_image_gray(
        self,
        img1: np.ndarray,  # Shape(H, W)
        img2: np.ndarray,  # Shape(H, W)
    ) -> tuple[np.ndarray, np.ndarray]:
        img1 = self.cropper1.crop_ndarray_image_gray(img1)
        img1 = self.cropper2.crop_ndarray_image_gray(img2)
        img1, img2 = self._resize_alignment(img1, img2)
        return img1, img2

    def crop_ndarray_image(
        self,
        img1: np.ndarray,  # Shape(H, W, 3)
        img2: np.ndarray,  # Shape(H, W, 3)
    ) -> tuple[np.ndarray, np.ndarray]:
        img1 = self.cropper1.crop_ndarray_image(img1)
        img2 = self.cropper2.crop_ndarray_image(img2)
        img1, img2 = self._resize_alignment(img1, img2)
        return img1, img2

    def crop_tensor_image(
        self,
        img1: torch.Tensor,  # Shape(3, H, W)
        img2: torch.Tensor,  # Shape(3, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img1 = self.cropper1.crop_tensor_image(img1)
        img2 = self.cropper2.crop_tensor_image(img2)
        img1, img2 = self._resize_alignment_torch(img1, img2)
        return img1, img2

    def _resize_alignment(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.enable_scale_alignment:
            return img1, img2

        resize_h1, resize_w1 = self.resize1
        resize_h2, resize_w2 = self.resize2
        resized_img1 = cv2.resize(
            img1.copy(), (resize_w1, resize_h1), interpolation=cv2.INTER_LINEAR
        )
        resized_img2 = cv2.resize(
            img2.copy(), (resize_w2, resize_h2), interpolation=cv2.INTER_LINEAR
        )
        return resized_img1, resized_img2

    def _resize_alignment_torch(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enable_scale_alignment:
            return img1, img2

        assert img1.ndim in (3, 4)  # (C, H, W) or (B, C, H, W)
        assert img2.ndim in (3, 4)  # (C, H, W) or (B, C, H, W)
        is4d = True
        if img1.ndim == 3:
            assert img2.ndim == 3
            is4d = False
            img1 = img1[None]
            img2 = img2[None]

        resized_img1 = torch.nn.functional.interpolate(img1, size=self.resize1)
        resized_img2 = torch.nn.functional.interpolate(img2, size=self.resize2)

        if not is4d:
            resized_img1 = resized_img1[0]
            resized_img2 = resized_img2[0]

        return resized_img1, resized_img2

    def _unresize_alignment_kpts(
        self, kpts1: np.ndarray, kpts2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.enable_scale_alignment:
            return kpts1, kpts2
        resize_h1, resize_w1 = self.resize1
        resize_h2, resize_w2 = self.resize2
        (bbox_h1, bbox_w1), (bbox_h2, bbox_w2) = self.get_bbox_sizes()

        kpts1[:, 0] *= float(bbox_w1) / float(resize_w1)
        kpts1[:, 1] *= float(bbox_h1) / float(resize_h1)
        kpts2[:, 0] *= float(bbox_w2) / float(resize_w2)
        kpts2[:, 1] *= float(bbox_h2) / float(resize_h2)

        return kpts1, kpts2

    def _unresize_alignment_kpts_torch(
        self, kpts1: torch.Tensor, kpts2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enable_scale_alignment:
            return kpts1, kpts2
        resize_h1, resize_w1 = self.resize1
        resize_h2, resize_w2 = self.resize2
        (bbox_h1, bbox_w1), (bbox_h2, bbox_w2) = self.get_bbox_sizes()

        kpts1[:, 0] *= float(bbox_w1) / float(resize_w1)
        kpts1[:, 1] *= float(bbox_h1) / float(resize_h1)
        kpts2[:, 0] *= float(bbox_w2) / float(resize_w2)
        kpts2[:, 1] *= float(bbox_h2) / float(resize_h2)

        return kpts1, kpts2


def keypoints_to_bbox(kpts: np.ndarray) -> np.ndarray:
    """
    Returns
    -------
    np.ndarray
        Shape(4,)
    """
    return np.array(
        [kpts[:, 0].min(), kpts[:, 1].min(), kpts[:, 0].max(), kpts[:, 1].max()]
    )


def adjust_bboxes(
    bboxes: np.ndarray,
    shape: Tuple[int, int],
    bbox_expansion: int = 8,
    border_padding: int = 8,
) -> np.ndarray:
    h, w = shape

    # Expand bbox's edge length
    bboxes[:, 0] = bboxes[:, 0] - bbox_expansion
    bboxes[:, 1] = bboxes[:, 1] - bbox_expansion
    bboxes[:, 2] = bboxes[:, 2] + bbox_expansion
    bboxes[:, 3] = bboxes[:, 3] + bbox_expansion

    # Clip borders
    pad = border_padding
    bboxes[:, 0] = np.clip(bboxes[:, 0], pad, (w - 1) - pad)
    bboxes[:, 1] = np.clip(bboxes[:, 1], pad, (h - 1) - pad)
    bboxes[:, 2] = np.clip(bboxes[:, 2], pad, (w - 1) - pad)
    bboxes[:, 3] = np.clip(bboxes[:, 3], pad, (h - 1) - pad)

    return bboxes


def whole_image_bbox(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    whole_bbox = np.array([0, 0, w - 1, h - 1])
    return whole_bbox


def whole_image_bboxes(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    whole_bbox = np.array([0, 0, w - 1, h - 1])
    whole_bboxes = whole_bbox[None, :]
    return whole_bboxes


def shorter_bbox_length(bbox: np.ndarray) -> int:
    return min(bbox[2] - bbox[0], bbox[3] - bbox[1])


def compute_main_region_keypoints_meanshift(
    kpts: np.ndarray,
    meanshift_kwargs: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    meanshift_kwargs = meanshift_kwargs or {"bin_seeding": True, "cluster_all": True}
    clustering = MeanShift(**meanshift_kwargs)
    clustering.fit(kpts)
    cluster_labels, counts = np.unique(clustering.labels_, return_counts=True)
    main_cluster_id_cands = cluster_labels[np.argsort(-counts)[:2]]
    if len(main_cluster_id_cands) == 2:
        if main_cluster_id_cands[0] == -1:
            main_cluster_id = main_cluster_id_cands[1]
        else:
            main_cluster_id = main_cluster_id_cands[0]
    else:
        main_cluster_id = main_cluster_id_cands[0]

    keeps, *_ = np.where(clustering.labels_ == main_cluster_id)
    main_kpts = kpts[keeps]

    return main_kpts, keeps


def compute_main_region_keypoints_lof(
    kpts: np.ndarray,
    lof_kwargs: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    lof_kwargs = lof_kwargs or {}
    clf = LocalOutlierFactor(**lof_kwargs)
    labels = clf.fit_predict(kpts)

    keeps, *_ = np.where(labels == 1)
    main_kpts = kpts[keeps]

    return main_kpts, keeps


def merge_bidirectional_estimations(
    matched_kpts1: np.ndarray,
    matched_kpts2: np.ndarray,
    keeps1: np.ndarray,
    keeps2: np.ndarray,
    bidirectional_merging: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if bidirectional_merging == "intersection":
        merged_keeps = np.intersect1d(keeps1, keeps2)
    elif bidirectional_merging == "union":
        merged_keeps = np.union1d(keeps1, keeps2)
    else:
        raise ValueError(bidirectional_merging)
    main_kpts1 = matched_kpts1[merged_keeps]
    main_kpts2 = matched_kpts2[merged_keeps]

    return main_kpts1, main_kpts2


def estimate_overlap_from_matches_meanshift(
    matched_kpts1: np.ndarray,
    matched_kpts2: np.ndarray,
    bidirectional: bool = True,
    bidirectional_merging: Optional[str] = "intersection",
    meanshift_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    np.ndarray
        Shape(#bbox, 4), xyxy order
    """
    meanshift_kwargs = meanshift_kwargs or {}
    main_kpts1, keeps1 = compute_main_region_keypoints_meanshift(
        matched_kpts1, meanshift_kwargs
    )

    if bidirectional:
        main_kpts2, keeps2 = compute_main_region_keypoints_meanshift(
            matched_kpts2, meanshift_kwargs
        )
        if bidirectional_merging:
            main_kpts1, main_kpts2 = merge_bidirectional_estimations(
                matched_kpts1, matched_kpts2, keeps1, keeps2, bidirectional_merging
            )
    else:
        main_kpts2 = matched_kpts2[keeps1]

    bbox1 = keypoints_to_bbox(main_kpts1)
    bbox2 = keypoints_to_bbox(main_kpts2)

    bboxes1 = bbox1[None, :]  # Shape(1, 4)
    bboxes2 = bbox2[None, :]  # Shape(1, 4)

    return bboxes1, bboxes2


def estimate_overlap_from_matches_lof(
    matched_kpts1: np.ndarray,
    matched_kpts2: np.ndarray,
    bidirectional: bool = True,
    bidirectional_merging: Optional[str] = "intersection",
    lof_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    np.ndarray
        Shape(#bbox, 4), xyxy order
    """
    lof_kwargs = lof_kwargs or {}
    main_kpts1, keeps1 = compute_main_region_keypoints_lof(matched_kpts1, lof_kwargs)

    if bidirectional:
        main_kpts2, keeps2 = compute_main_region_keypoints_lof(
            matched_kpts2, lof_kwargs
        )
        if bidirectional_merging:
            main_kpts1, main_kpts2 = merge_bidirectional_estimations(
                matched_kpts1, matched_kpts2, keeps1, keeps2, bidirectional_merging
            )
    else:
        main_kpts2 = matched_kpts2[keeps1]

    bbox1 = keypoints_to_bbox(main_kpts1)
    bbox2 = keypoints_to_bbox(main_kpts2)

    bboxes1 = bbox1[None, :]  # Shape(1, 4)
    bboxes2 = bbox2[None, :]  # Shape(1, 4)

    return bboxes1, bboxes2
