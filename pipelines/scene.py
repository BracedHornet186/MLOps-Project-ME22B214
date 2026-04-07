from __future__ import annotations

import concurrent.futures
import contextlib
import gc
import tempfile
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
from PIL import Image

from data import (
    DEFAULT_OUTLIER_SCENE_NAME,
    DEFAULT_SPACE_NAME,
    DEFAULT_TMP_DIR,
    IS_SCENE_SPACE_DIR_PERSISTENT,
    DirPath,
    FilePath,
)
from data_schema import DataSchema
from preprocesses.region import Cropper, OverlapRegionCropper, whole_image_bbox

IMAGE_CACHE_NUM_LIMIT = 1000
# IMAGE_CACHE_SIZE_LIMIT = 1024   # MB
# IMAGE_CACHE_SIZE_LIMIT = 1536  # MB
IMAGE_CACHE_SIZE_LIMIT = 4096  # MB


def default_image_reader(path: str) -> tuple[str, np.ndarray]:
    return (str(path), cv2.imread(str(path)))


@dataclass
class Scene:
    dataset: str
    scene: str
    image_paths: list[FilePath]
    image_dir: DirPath
    data_schema: DataSchema

    output_dir: Optional[Path] = None

    # Shortlist
    shortlist: Optional[list[tuple[int, int]]] = None
    topk_ranks: Optional[np.ndarray] = None
    topk_dists: Optional[np.ndarray] = None

    # Additional directories
    additional_output_dirs: Optional[list[Path]] = None

    # For clustered scene
    indices_in_parent_scene: Optional[np.ndarray] = None

    # Overlap regions of (path(i), path(j)) pair
    overlap_regions: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    # RoI of images
    bboxes: dict[str, np.ndarray] = field(default_factory=dict)

    # Mask regions
    mask_bboxes: dict[str, list[np.ndarray]] = field(default_factory=dict)

    # Orientations
    orientations: dict[str, int] = field(default_factory=dict)

    # Cached images (key=path, val=img)
    images: dict[str, np.ndarray] = field(default_factory=dict)

    # Cached deblurred images (key=original_image_path, val=img)
    deblurred_images: dict[str, np.ndarray] = field(default_factory=dict)

    # Cached depth images (key=original_image_path, val=img)
    depth_images: dict[str, np.ndarray] = field(default_factory=dict)

    # Cached segmentation mask images (key=original_image_path, val=img)
    segmentation_mask_images: dict[str, np.ndarray] = field(default_factory=dict)

    # Cached image shapes
    image_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    # Metadata dict
    # metadata: Dict[str, list[Any]] = field(default_factory=)

    # Cached key-index mapping
    _full_key_to_idx: Optional[dict[str, int]] = None
    _short_key_to_idx: Optional[dict[str, int]] = None
    _output_key_to_idx: Optional[dict[str, int]] = None

    def __repr__(self) -> str:
        return f'Scene(dataset="{self.dataset}", scene="{self.scene}")'

    def __str__(self) -> str:
        return repr(self)

    @property
    def name(self) -> str:
        return self.scene

    @property
    def reconstruction_dir(self) -> Path:
        assert self.output_dir
        return self.output_dir / "colmap_rec"

    @property
    def database_path(self) -> Path:
        assert self.output_dir
        return self.output_dir / "colmap.sqlite3"

    @property
    def hloc_dir(self) -> Path:
        assert self.output_dir
        return self.output_dir / "hloc"

    @property
    def deblur_image_dir(self) -> Path:
        assert self.output_dir
        return self.output_dir / "deblur_images"

    @contextlib.contextmanager
    def create_space(self) -> Generator:
        try:
            if IS_SCENE_SPACE_DIR_PERSISTENT:
                output_dir = DEFAULT_TMP_DIR / DEFAULT_SPACE_NAME / self.name
                output_dir.mkdir(parents=True, exist_ok=True)
                self.output_dir = output_dir
                self._make_dirs()
                yield self
            else:
                with tempfile.TemporaryDirectory() as directory:
                    output_dir = Path(directory)
                    self.output_dir = output_dir
                    self._make_dirs()
                    yield self
        finally:
            self.output_dir = None

    def _make_dirs(self):
        self.reconstruction_dir.mkdir(parents=True, exist_ok=True)
        self.hloc_dir.mkdir(parents=True, exist_ok=True)
        self.deblur_image_dir.mkdir(parents=True, exist_ok=True)

    def make_output_dir_for_child_scene(self, child_scene: Scene):
        if self.additional_output_dirs is None:
            self.additional_output_dirs = []
        assert isinstance(self.additional_output_dirs, list)
        assert isinstance(self.output_dir, Path)
        new_dir = self.output_dir / child_scene.name
        new_dir.mkdir(parents=True, exist_ok=True)
        child_scene.output_dir = new_dir
        child_scene._make_dirs()
        self.additional_output_dirs.append(new_dir)

    def is_outlier_scene(self) -> bool:
        return self.scene == DEFAULT_OUTLIER_SCENE_NAME

    def idx_to_parent_scene_idx(self, idx: int) -> int:
        assert self.indices_in_parent_scene is not None
        return int(self.indices_in_parent_scene[idx])

    def get_image(self, path: FilePath, use_original: bool = False) -> np.ndarray:
        path = str(path)

        if use_original:
            pass
        else:
            if path in self.deblurred_images:
                print(f"[get_image] Use deblurred image for {path}")
                return self.deblurred_images[path].copy()

        if path not in self.images:
            # print(f'No cached: {path}')
            return cv2.imread(str(path))
        return self.images[path].copy()

    def get_depth_image(self, path: FilePath) -> Optional[np.ndarray]:
        path = str(path)
        img = self.depth_images.get(path)
        if img is None:
            return None
        return img.copy()

    def update_depth_image(
        self, path: FilePath, depth_image: np.ndarray
    ) -> Optional[np.ndarray]:
        path = str(path)
        if path in self.depth_images:
            print(f"Warning! {path} has a depth image. Replacing")
        self.depth_images[path] = depth_image.copy()

    def update_segmentation_mask_image(
        self, path: FilePath, mask_image: np.ndarray
    ) -> Optional[np.ndarray]:
        path = str(path)
        if path in self.depth_images:
            print(f"Warning! {path} has a mask image. Replacing")
        self.segmentation_mask_images[path] = mask_image.copy()

    def get_unique_resolution_num(self) -> int:
        resolutions = set([shape for shape in self.image_shapes.values()])
        return len(resolutions)

    def cache_all_images(
        self, max_workers: int = 2, limit_size: int = IMAGE_CACHE_SIZE_LIMIT
    ):
        all_size = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            image_paths = self.image_paths[:IMAGE_CACHE_NUM_LIMIT]
            for i, (path, img) in enumerate(
                executor.map(default_image_reader, image_paths)
            ):
                assert self.image_paths[i] == str(path)
                all_size += int(img.nbytes / 1024 / 1024)
                if all_size > limit_size:
                    print(f"No left cache space. skip: {path}")
                    self.image_shapes[path] = tuple(img.shape[:2])
                else:
                    self.images[path] = img.copy()
                    self.image_shapes[path] = tuple(img.shape[:2])
                    print(f"Cached: {path}")

    def release_cached_images(self) -> None:
        if self.images is None:
            return

        keys = list(self.images.keys())
        for k in keys:
            print(f"Release from cache: {k}")
            del self.images[k]
        self.images = {}
        gc.collect()

    def release_all(self) -> None:
        if self.topk_dists is not None:
            del self.topk_dists
            self.topk_dists = None
        if self.topk_ranks is not None:
            del self.topk_ranks
            self.topk_ranks = None
        if self.shortlist is not None:
            del self.shortlist
            self.shortlist = None
        if self.image_shapes is not None:
            del self.image_shapes
            self.image_shapes = {}
        gc.collect()

    def release_topk_dists_and_ranks(self) -> Scene:
        del self.topk_dists
        self.topk_dists = None
        del self.topk_ranks
        self.topk_ranks = None
        return self

    def idx_to_key(self, i: int, key_type: str = "short_key"):
        path = self.image_paths[i]
        if key_type == "short_key":
            return Path(path).name
        elif key_type == "full_key":
            return str(path)
        else:
            raise ValueError

    def get_retrieval_dict(self, key_type: str = "short_key") -> dict:
        assert self.topk_ranks is not None
        retrieval_dict = defaultdict(list)
        for i in range(len(self.topk_ranks)):
            q = self.idx_to_key(i, key_type=key_type)
            ranks = self.topk_ranks[i]
            for j in ranks:
                j = int(j)
                if i == j:
                    continue
                r = self.idx_to_key(j, key_type=key_type)
                retrieval_dict[q].append(r)
        return retrieval_dict

    def update_shortlist(self, pairs: list[tuple[int, int]]) -> Scene:
        if self.shortlist is not None:
            print("[Scene] Warning! Shortlist already exists")
            print(f"[Scene] Updated: {len(self.shortlist)} -> {len(pairs)}")
        self.shortlist = pairs
        return self

    def update_topk_table(
        self,
        topk_ranks: Optional[np.ndarray] = None,
        topk_dists: Optional[np.ndarray] = None,
    ) -> Scene:
        if self.topk_dists is not None or self.topk_ranks is not None:
            print("[Scene] Warning! Topk table already exists")
        if topk_ranks is not None:
            self.topk_ranks = topk_ranks.copy()
        if topk_dists is not None:
            self.topk_dists = topk_dists.copy()
        return self

    def update_overlap_regions(
        self, path1: FilePath, path2: FilePath, bbox1: np.ndarray, bbox2: np.ndarray
    ) -> Scene:
        path1 = str(path1)
        path2 = str(path2)
        if path1 not in self.overlap_regions:
            self.overlap_regions[path1] = {}
        if path2 not in self.overlap_regions:
            self.overlap_regions[path2] = {}
        if path2 in self.overlap_regions[path1]:
            # print(
            #    f"[Scene] Warning! Overlap region of ({path1}, {path2}) "
            #    f"already exists"
            # )
            pass
        if path1 in self.overlap_regions[path2]:
            # print(
            #    f"[Scene] Warning! Overlap region of ({path1}, {path2}) "
            #    f"already exists"
            # )
            pass
        self.overlap_regions[path1][path2] = bbox1.copy()
        self.overlap_regions[path2][path1] = bbox2.copy()
        return self

    def update_orientation(self, path: FilePath, degree: int) -> Scene:
        assert degree in (0, 90, 180, 270)
        self.orientations[Path(path).name] = degree
        return self

    def get_overlap_regions(
        self,
        path1: FilePath,
        path2: FilePath,
    ) -> tuple[np.ndarray, np.ndarray]:
        path1 = str(path1)
        path2 = str(path2)
        bbox1 = self.overlap_regions[path1][path2]
        bbox2 = self.overlap_regions[path2][path1]
        return bbox1, bbox2

    def get_overlap_regions_if_exists(
        self,
        path1: FilePath,
        path2: FilePath,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if path1 in self.overlap_regions and path2 in self.overlap_regions:
            if (
                path2 in self.overlap_regions[path1]
                and path1 in self.overlap_regions[path2]
            ):
                return self.get_overlap_regions(path1, path2)
        return None, None

    def create_roi_region_cropper(self, path: FilePath) -> Cropper:
        bbox = self.bboxes.get(str(path))
        if bbox is None:
            bbox = whole_image_bbox(self.get_image_shape(path))
        cropper = Cropper(bbox, self.get_image_shape(path))
        return cropper

    def create_roi_region_cropper_if_exists(self, path: FilePath) -> Optional[Cropper]:
        cropper = None
        bbox = self.bboxes.get(str(path))
        if bbox is not None:
            cropper = Cropper(bbox, self.get_image_shape(path))
        return cropper

    def create_overlap_region_cropper(
        self,
        path1: FilePath,
        path2: FilePath,
        enable_scale_alignment: bool = True,
        cropper_type: Literal["overlap", "roi", "overlap-or-roi", "ignore"] = "overlap",
    ) -> Optional[OverlapRegionCropper]:
        if cropper_type == "overlap":
            bbox1, bbox2 = self.get_overlap_regions_if_exists(path1, path2)
            if bbox1 is None or bbox2 is None:
                return None
            cropper1 = Cropper(bbox1, self.get_image_shape(path1))
            cropper2 = Cropper(bbox2, self.get_image_shape(path2))
            return OverlapRegionCropper(
                cropper1, cropper2, enable_scale_alignment=enable_scale_alignment
            )
        elif cropper_type == "roi":
            cropper1 = self.create_roi_region_cropper(path1)
            cropper2 = self.create_roi_region_cropper(path2)
            return OverlapRegionCropper(
                cropper1, cropper2, enable_scale_alignment=enable_scale_alignment
            )
        elif cropper_type == "overlap-or-roi":
            cropper = self.create_overlap_region_cropper(
                path1,
                path2,
                enable_scale_alignment=enable_scale_alignment,
                cropper_type="overlap",
            )
            if cropper is None:
                cropper = self.create_overlap_region_cropper(
                    path1,
                    path2,
                    enable_scale_alignment=enable_scale_alignment,
                    cropper_type="roi",
                )
            return cropper
        elif cropper_type == "ignore":
            return None
        else:
            raise ValueError(cropper_type)

    def make_roi_from_overlap_regions(self):
        assert len(self.overlap_regions) > 0
        for path1 in self.image_paths:
            path1 = str(path1)
            regions = []
            try:
                for _, region in self.overlap_regions[path1].items():
                    regions.append(region)
                if len(regions) == 0:
                    continue
                bboxes = np.stack(regions)
                fusion_bbox = np.array(
                    [
                        bboxes[:, 0].min(),
                        bboxes[:, 1].min(),
                        bboxes[:, 2].max(),
                        bboxes[:, 3].max(),
                    ]
                )
                if (
                    fusion_bbox[2] - fusion_bbox[0] < 64
                    and fusion_bbox[3] - fusion_bbox[1] < 64
                ):
                    continue
                self.bboxes[path1] = fusion_bbox.copy()
            except Exception as e:
                print(f"[make_roi_from_overlap_regions] Error: {e}")

    def make_mask_regions_from_overlap_regions(
        self,
        overlap_delta: int = 10,
        border_delta: int = 100,
        max_area_ratio: float = 0.1,
    ):
        assert len(self.overlap_regions) > 0
        for path1 in self.image_paths:
            try:
                for path2 in self.overlap_regions[str(path1)].keys():
                    path1 = str(path1)
                    path2 = str(path2)

                    bbox1, bbox2 = self.get_overlap_regions_if_exists(path1, path2)
                    if bbox1 is None or bbox2 is None:
                        continue

                    height1, width1 = self.get_image_shape(path1)
                    height2, width2 = self.get_image_shape(path2)

                    area_threshold1 = int(height1 * width1 * max_area_ratio)
                    area_threshold2 = int(height2 * width2 * max_area_ratio)

                    w1 = bbox1[2] - bbox1[0]
                    h1 = bbox1[3] - bbox1[1]
                    w2 = bbox2[2] - bbox2[0]
                    h2 = bbox2[3] - bbox2[1]

                    if (w1 * h1) > area_threshold1:
                        continue

                    if (w2 * h2) > area_threshold2:
                        continue

                    diffs = [abs(w1 - w2), abs(h1 - h2)]
                    # print(f'DIFF: {diffs} ({max(diffs)})')
                    if max(diffs) > overlap_delta:
                        continue

                    height, width = self.get_image_shape(path1)
                    left = bbox1[0]
                    up = bbox1[1]
                    right = bbox1[2]
                    bottom = bbox1[3]
                    if border_delta < left and right < (width - border_delta):
                        if border_delta < up and bottom < (height - border_delta):
                            continue

                    if path1 not in self.mask_bboxes:
                        self.mask_bboxes[path1] = [bbox1.copy()]
                    else:
                        self.mask_bboxes[path1].append(bbox1.copy())

                    if path2 not in self.mask_bboxes:
                        self.mask_bboxes[path2] = [bbox2.copy()]
                    else:
                        self.mask_bboxes[path2].append(bbox2.copy())
                    print(
                        f"[make_mask_regions_from_overlap_regions] "
                        f"Added mask regions from the overlap between {path1} and {path2}"
                    )
            except Exception as e:
                print(f"[mask_mask_regions_from_overlap_regions] Error: {e}")

    def get_mask_regions(self, path: FilePath) -> list[np.ndarray]:
        return self.mask_bboxes.get(str(path)) or []

    def get_segmentation_mask(self, path: FilePath) -> Optional[np.ndarray]:
        return self.segmentation_mask_images.get(str(path))

    def get_orientation_degree(self, path: FilePath) -> Optional[int]:
        return self.orientations.get(Path(path).name)

    def image_name_to_full_key(self, name: str) -> str:
        return f"{self.dataset}/{self.scene}/images/{name}"

    def image_path_to_full_key(self, path: FilePath) -> str:
        return self.image_name_to_full_key(Path(path).name)

    def output_key_to_idx(self, output_key: str) -> int:
        if not self._output_key_to_idx:
            print("[Scene] There is no key(for output)-idx mapping. Creating.")
            self._output_key_to_idx = {
                self.data_schema.format_output_key(
                    self.dataset, self.scene, Path(path).name
                ): i
                for i, path in enumerate(self.image_paths)
            }
            print(self._output_key_to_idx)
        return self._output_key_to_idx[output_key]

    def full_key_to_idx(self, full_key: str) -> int:
        if not self._full_key_to_idx:
            print("[Scene] There is no key-idx mapping. Creating.")
            self._full_key_to_idx = {
                self.image_path_to_full_key(path): i
                for i, path in enumerate(self.image_paths)
            }
        return self._full_key_to_idx[full_key]

    def short_key_to_idx(self, short_key: str) -> int:
        if not self._short_key_to_idx:
            print("[Scene] There is no shortkey-idx mapping. Creating.")
            self._short_key_to_idx = {
                Path(path).name: i for i, path in enumerate(self.image_paths)
            }
        return self._short_key_to_idx[short_key]

    def short_key_to_image_path(self, short_key: str) -> str:
        return str(self.image_paths[self.short_key_to_idx(short_key)])

    def get_nearest_neighbors_indices(self, full_key: str) -> list[int]:
        if self.topk_ranks is None or self.topk_dists is None:
            print(
                "[Scene] Warning! Cannot return nearest neighbors, "
                "because topk_ranks and topk_dists are not updated"
            )
            return []
        i = self.full_key_to_idx(full_key)
        assert isinstance(self.topk_ranks, np.ndarray)
        return self.topk_ranks[i].tolist()

    def get_nearest_neighbors_full_keys(self, full_key: str) -> list[str]:
        indices = self.get_nearest_neighbors_indices(full_key)
        return [self.image_path_to_full_key(self.image_paths[i]) for i in indices]

    def get_image_shape(self, path: FilePath) -> tuple[int, int]:
        """Return a shape of given image

        Returns
        -------
        Tuple[int, int]
            (height, width)
        """
        path = str(path)
        if path in self.image_shapes:
            return self.image_shapes[path]
        img = self.get_image(path)
        shape = tuple(img.shape[:2])
        self.image_shapes[path] = shape
        return shape

    def get_paired_names(self, pair: tuple[int, int]) -> tuple[str, str]:
        path1, path2 = self.get_paired_image_paths(pair)
        return Path(path1).name, Path(path2).name

    def get_paired_image_paths(
        self, pair: tuple[int, int]
    ) -> tuple[FilePath, FilePath]:
        i, j = pair
        return self.image_paths[i], self.image_paths[j]

    def get_paired_image(
        self, pair: tuple[int, int]
    ) -> tuple[Image.Image, Image.Image]:
        path1, path2 = self.get_paired_image_paths(pair)
        image1 = Image.open(path1).convert("RGB")
        image2 = Image.open(path2).convert("RGB")
        return image1, image2

    def make_shortlist_index(
        self, pairs: list[tuple[int, int]], full: bool = False
    ) -> dict[int, list[int]]:
        query2refs = defaultdict(list)
        for q, r in pairs:
            query2refs[q].append(r)
        if not full:
            return query2refs

        index = {}
        for i in range(len(self.image_paths)):
            index[i] = sorted(list(query2refs[i]))
        return index

    def make_scene_graph(
        self,
        ax: Optional[Any] = None,
    ) -> dict:
        assert self.shortlist
        return make_scene_graph(self.shortlist, self.image_paths, ax=ax)


def make_scene_graph(
    pairs: list[tuple[int, int]],
    image_paths: list[FilePath],
    ax: Optional[Any] = None,
) -> dict:
    G = nx.Graph()
    G.add_nodes_from(range(len(image_paths)))
    for pair in pairs:
        i, j = pair
        G.add_edge(i, j)

    isolated_nodes = []
    for ci, cc in enumerate(nx.connected_components(G)):
        print(f"CC[{ci}] #nodes={len(cc)}")
        if len(cc) == 1:
            i = list(cc)[0]
            isolated_nodes.append(i)
            print(f"  - i={i}, path={image_paths[i]}")
    G.remove_nodes_from(isolated_nodes)

    scene_graph = {
        "graph": G,
        "isolated_nodes": isolated_nodes,
        "center_node": int(
            list(sorted(list(nx.degree_centrality(G).items()), key=lambda t: -t[1]))[0][
                0
            ]
        ),
    }
    print("Scene graph info")
    print("================")
    print(f"  - isolated_nodes: {scene_graph['isolated_nodes']}")
    print(f"  - center_node: {scene_graph['center_node']}")

    if ax is not None:
        pos = nx.spring_layout(G, seed=0)
        nx.draw_networkx(G, pos=pos, ax=ax, with_labels=True)

    return scene_graph
