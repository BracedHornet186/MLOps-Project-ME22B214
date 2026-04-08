from typing import Any, Optional, Tuple, TypeVar
import cv2
import kornia
import numpy as np
import torch
from PIL import Image

from preprocesses.config import ResizeConfig, PairedPreResizeConfig


T = TypeVar('T', np.ndarray, Image.Image)


def resize_image_tensor(x: torch.Tensor, conf: ResizeConfig) -> torch.Tensor:
    if conf.func == 'kornia':
        if conf.small_edge_length:
            assert conf.size is None and conf.long_edge_length is None
            x = kornia.geometry.resize(x, conf.small_edge_length,
                                       side='short',
                                       antialias=conf.antialias)
        elif conf.long_edge_length:
            assert conf.size is None and conf.small_edge_length is None
            x = kornia.geometry.resize(x, conf.long_edge_length,
                                       side='long',
                                       antialias=conf.antialias)
        else:
            assert conf.size
            assert conf.small_edge_length is None
            assert conf.long_edge_length is None
            x = kornia.geometry.resize(x, conf.size,
                                       side='long',
                                       antialias=conf.antialias)
    else:
        raise ValueError(conf.func)
    return x


def resize_image_opencv(img: np.ndarray,
                        conf: ResizeConfig,
                        order3ch: str = 'chw') -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if conf.func == 'opencv-divisible':
        if conf.small_edge_length:
            raise NotImplementedError
        elif conf.long_edge_length:
            assert conf.size is None and conf.small_edge_length is None
            w, h = img.shape[1], img.shape[0]
            w_new, h_new = _get_resized_wh(w, h, conf.long_edge_length)
            if conf.divisible_factor is not None:
                w_new, h_new = _get_divisible_wh(w_new, h_new,
                                                 conf.divisible_factor)
            
            if conf.cv2_interpolation == 'LANCZOS4':
                img = cv2.resize(img, (w_new, h_new),
                                 interpolation=cv2.INTER_LANCZOS4)
            else:
                img = cv2.resize(img, (w_new, h_new))
            scale = np.array([w / w_new, h / h_new])

            if conf.pad_bottom_right:  # padding
                pad_to = max(h_new, w_new)
                img, mask = _pad_bottom_right(img, pad_to,
                                              ret_mask=True,
                                              order3ch=order3ch)
            else:
                mask = None
        else:
            raise NotImplementedError
    else:
        raise ValueError(conf.func)
    return img, scale, mask


def paired_pre_resize(
    qimg: T,
    ximg: T,
    conf: PairedPreResizeConfig,
) -> Tuple[T, T]:
    if conf.func == 'same_smaller_width':
        if conf.img_object_type == 'cv2':
            assert isinstance(qimg, np.ndarray) and isinstance(ximg, np.ndarray)
            return make_image_pairs_with_same_smaller_width_cv2(
                qimg, ximg,
                limit_size=conf.limit_size,
                interpolation=conf.get_cv2_interp()
            )
        elif conf.img_object_type == 'pil':
            assert isinstance(qimg, Image.Image) and isinstance(ximg, Image.Image)
            return make_image_pairs_with_same_smaller_width_pil(
                qimg, ximg,
                limit_size=conf.limit_size,
            )
    elif conf.func == 'same_longest_edge':
        if conf.img_object_type == 'cv2':
            assert isinstance(qimg, np.ndarray) and isinstance(ximg, np.ndarray)
            return make_image_pairs_with_same_longest_edge_cv2(
                qimg, ximg,
                interpolation=conf.get_cv2_interp()
            )
        elif conf.img_object_type == 'pil':
            raise NotImplementedError
    raise ValueError(conf.func)


def make_image_pairs_with_same_smaller_width_pil(
    qimg: Image.Image,
    ximg: Image.Image,
    limit_size: Optional[int] = None,
) -> Tuple[Image.Image, Image.Image]:
    if qimg.width >= ximg.width:
        base_width = ximg.width
        resized_qimg = qimg.resize((base_width, int(qimg.height * base_width / qimg.width)))
        resized_ximg = ximg
    else:
        base_width = qimg.width
        resized_qimg = qimg
        resized_ximg = ximg.resize((base_width, int(ximg.height * base_width / ximg.width)))
    
    if limit_size and base_width > limit_size:
        resized_qimg = resized_qimg.resize((limit_size, int(qimg.height * limit_size / qimg.width)))
        resized_ximg = resized_ximg.resize((limit_size, int(ximg.height * limit_size / ximg.width)))
    
    return resized_qimg, resized_ximg


def make_image_pairs_with_same_smaller_width_cv2(
    qimg: np.ndarray,       # Shape(H, W) or Shape(H, W, C)
    ximg: np.ndarray,       # Shape(H, W) or Shape(H, W, C)
    limit_size: Optional[int] = None,
    interpolation: int = cv2.INTER_LINEAR
) -> Tuple[np.ndarray, np.ndarray]:
    assert len(qimg.shape) == 2 or len(qimg.shape) == 3
    assert len(ximg.shape) == 2 or len(ximg.shape) == 3
    q_height, q_width = qimg.shape[:2]
    x_height, x_width = ximg.shape[:2]
    if q_width >= x_width:
        base_width = x_width
        resized_qimg = cv2.resize(qimg,
                                  (base_width, int(q_height * base_width / q_width)),
                                  interpolation=interpolation)
        resized_ximg = ximg
    else:
        base_width = q_width
        resized_qimg = qimg
        resized_ximg = cv2.resize(ximg,
                                  (base_width, int(x_height * base_width / x_width)),
                                  interpolation=interpolation)
    
    if limit_size and base_width > limit_size:
        resized_qimg = cv2.resize(resized_qimg,
                                  (limit_size, int(q_height * limit_size / q_width)),
                                  interpolation=interpolation)
        resized_ximg = cv2.resize(resized_ximg,
                                  (limit_size, int(x_height * limit_size / x_width)),
                                  interpolation=interpolation)
    
    return resized_qimg, resized_ximg


def make_image_pairs_with_same_longest_edge_cv2(
    qimg: np.ndarray,       # Shape(H, W) or Shape(H, W, C)
    ximg: np.ndarray,       # Shape(H, W) or Shape(H, W, C)
    interpolation: int = cv2.INTER_LINEAR
) -> Tuple[np.ndarray, np.ndarray]:
    assert len(qimg.shape) == 2 or len(qimg.shape) == 3
    assert len(ximg.shape) == 2 or len(ximg.shape) == 3
    q_height, q_width = qimg.shape[:2]
    x_height, x_width = ximg.shape[:2]

    q_longest = max(q_height, q_width)
    x_longest = max(x_height, x_width)

    if q_longest >= x_longest:
        if q_height == q_longest:
            # The query is larger than the reference.
            # The query is a vertical image
            base_height = q_height
            resized_qimg = qimg.copy()
            resized_ximg = cv2.resize(ximg.copy(),
                                      (int(x_width * base_height / x_height), base_height),
                                      interpolation=interpolation)
        else:
            assert q_width == q_longest
            # The query is larger than the reference.
            # The query is a horizontal image
            base_width = q_width
            resized_qimg = qimg.copy()
            resized_ximg = cv2.resize(ximg.copy(),
                                      (base_width, int(x_height * base_width / x_width)),
                                      interpolation=interpolation)
    else:
        assert q_longest < x_longest
        if x_height == x_longest:
            # The reference is larger than the query
            # The reference is a vertical image
            base_height = x_height
            resized_qimg = cv2.resize(qimg.copy(),
                                      (int(q_width * base_height / q_height), base_height),
                                      interpolation=interpolation)
            resized_ximg = ximg.copy()
        else:
            assert x_width == x_longest
            # The reference is larger than the query
            # The reference is a horizontal image
            base_width = x_width
            resized_qimg = cv2.resize(qimg.copy(),
                                      (base_width, int(q_height * base_width / q_width)),
                                      interpolation=interpolation)
            resized_ximg = ximg.copy()

    return resized_qimg, resized_ximg


def _get_resized_wh(
    w: int, h: int, size: Optional[int] = None
) -> Tuple[int, int]:
    if size is not None:  # resize the longer edge
        scale = size / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def _get_divisible_wh(
    w: int, h: int, factor: Optional[int] = None
) -> Tuple[int, int]:
    if factor is not None:
        w_new, h_new = map(lambda x: int(x // factor * factor), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def _pad_bottom_right(
    inp: np.ndarray,
    pad_size: int,
    ret_mask: bool = False,
    order3ch: str = 'chw'
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        if order3ch == 'chw':
            padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
            padded[:, :inp.shape[1], :inp.shape[2]] = inp
            if ret_mask:
                mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
                mask[:, :inp.shape[1], :inp.shape[2]] = True
        elif order3ch == 'hwc':
            padded = np.zeros((pad_size, pad_size, inp.shape[2]), dtype=inp.dtype)
            padded[:inp.shape[0], :inp.shape[1], :] = inp
            if ret_mask:
                mask = np.zeros((pad_size, pad_size, inp.shape[2]), dtype=bool)
                mask[:inp.shape[0], :inp.shape[1], :] = True
        else:
            raise ValueError
    else:
        raise NotImplementedError()
    return padded, mask