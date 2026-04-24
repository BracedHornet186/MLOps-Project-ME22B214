from typing import List, Optional, Tuple
from collections import defaultdict

import networkx as nx
import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from torchvision.utils import make_grid

from scripts.data import FilePath, load_train_df
from pipelines.common import create_data_dict, iterate_scenes
from pipelines.scene import Scene
from shortlists.base import ShortlistGenerator
from shortlists.config import ShortlistGeneratorConfig
from shortlists.factory import create_shortlist_generator
from shortlists.connected_component import GlobalDescriptorConnectedComponentShortlistGenerator


class ShortlistVisualizer:
    def __init__(self,
                 generator: ShortlistGenerator,
                 scenes: Optional[List[Scene]] = None):
        self.generator = generator
        if not scenes:
            df = load_train_df(replace_abs_path=True)
            data_dict = create_data_dict(df=df)
            scenes = list(iterate_scenes(data_dict))
        self.scenes = scenes
        self.scene: Optional[Scene] = None
        self.pairs: List[Tuple[int, int]] = []
    
    @classmethod
    def from_config(cls, config_file_path: FilePath) -> 'ShortlistVisualizer':
        device = torch.device('cuda')
        conf = ShortlistGeneratorConfig.load_config(str(config_file_path))
        
        return ShortlistVisualizer(
            create_shortlist_generator(conf, device=device)
        )
    
    def set_scene(self,
                  i: Optional[int] = None,
                  name: Optional[str] = None,
                  scene: Optional[Scene] = None) -> 'ShortlistVisualizer':
        if i is not None:
            self.scene = self.scenes[i]
            return self.generate()
        if name is not None:
            self.scene = [
                scene for scene in self.scenes
                if scene.name == name
            ][0]
            return self.generate()
        if scene is not None:
            self.scene = scene
            return self.generate()
        raise ValueError
    
    def next_scene(self) -> 'ShortlistVisualizer':
        assert self.scene
        self.scene = self.scenes[
            (self.scenes.index(self.scene) + 1) % len(self.scenes)
        ]
        return self
    
    def generate(self) -> 'ShortlistVisualizer':
        assert self.scene
        progress_bar = tqdm.tqdm(range(1), total=1)
        pairs = self.generator(self.scene, progress_bar=progress_bar)
        self.pairs = pairs
        return self
    
    def show_current_scene_info(self, full: bool = False) -> None:
        assert self.scene
        assert self.pairs
        scene = self.scene
        pairs = self.pairs
        print(f'Scene: {scene}')
        print('------------------------')
        print(f'# of images: {len(scene.image_paths)}')
        print(f'# of pairs: {len(pairs)}')
        shortlist_index = scene.make_shortlist_index(pairs, full=full)
        for q, rs in shortlist_index.items():
            print(f' - query={q}, refs={rs} ({len(rs)} pairs)')
    
    def show_pair_at(self, i: int) -> Image.Image:
        assert self.scene
        assert self.pairs
        image1, image2 = self.scene.get_paired_image(self.pairs[i])
        image1.thumbnail((128, 128))
        image2.thumbnail((128, 128))
        image1 = pad2square(image1)
        image2 = pad2square(image2)
        x = torch.stack([
            torch.from_numpy(np.array(image1)).permute(2, 0, 1),
            torch.from_numpy(np.array(image2)).permute(2, 0, 1)
        ])
        grid_img_tensor = make_grid(x)
        return Image.fromarray(
            grid_img_tensor.cpu().numpy().transpose(1, 2, 0)
        )
    
    def show_shortlist_image(self,
                             full: bool = False,
                             cell_size: int = 64) -> Image.Image:
        assert self.scene
        assert self.pairs
        size = (cell_size, cell_size)
        scene = self.scene
        shortlist_index = scene.make_shortlist_index(self.pairs, full=full)
        images = []
        for q, rs in shortlist_index.items():
            image1 = Image.open(scene.image_paths[q]).convert('RGB')
            image1.thumbnail(size)
            image1 = pad2square(image1)
            row_images = [image1]
            for r in rs:
                image2 = Image.open(scene.image_paths[r]).convert('RGB')
                image2.thumbnail(size)
                image2 = pad2square(image2)
                row_images.append(image2)
            images.append(row_images)
        
        max_cols = max([len(row_images) for row_images in images])

        tensors = []
        for row_images in images:
            x = torch.stack([
                torch.from_numpy(np.array(image)).permute(2, 0, 1)
                for image in row_images
            ] + [
                torch.zeros((3, cell_size, cell_size), dtype=torch.uint8)
                for _ in range(max_cols - len(row_images))
            ])
            tensors.append(x)
        
        tensor_image = torch.cat(tensors)
        grid_img_tensor = make_grid(tensor_image, nrow=max_cols)
        return Image.fromarray(
            grid_img_tensor.cpu().numpy().transpose(1, 2, 0)
        )

    def show_cc_graph_image(self,
                            full: bool = False,
                            cell_size: int = 64) -> Image.Image:
        assert self.scene
        assert isinstance(self.generator, GlobalDescriptorConnectedComponentShortlistGenerator)
        size = (cell_size, cell_size)
        scene = self.scene

        progress_bar = tqdm.tqdm(range(1), total=1)
        G = self.generator.compute_graph(self.scene, progress_bar=progress_bar)

        all_idxes = set(np.arange(len(scene.image_paths)))
        cluster_idxes = set()

        clusters = []
        for ci, cc in enumerate(nx.connected_components(G)):
            cc = list(cc)
            q = cc[0]
            xs = cc[1:]
            pairs = [(q, x) for x in xs]
            for pair in pairs:
                clusters.append(pair)

            cluster_idxes.add(q)
            for x in xs:
                cluster_idxes.add(x)
        
        for q in (all_idxes - cluster_idxes):
            clusters.append((q, q))

        shortlist_index = scene.make_shortlist_index(clusters, full=full)
        images = []
        for q, rs in shortlist_index.items():
            image1 = Image.open(scene.image_paths[q]).convert('RGB')
            image1.thumbnail(size)
            image1 = pad2square(image1)
            row_images = [image1]
            for r in rs:
                image2 = Image.open(scene.image_paths[r]).convert('RGB')
                image2.thumbnail(size)
                image2 = pad2square(image2)
                row_images.append(image2)
            images.append(row_images)
        
        max_cols = max([len(row_images) for row_images in images])

        tensors = []
        for row_images in images:
            x = torch.stack([
                torch.from_numpy(np.array(image)).permute(2, 0, 1)
                for image in row_images
            ] + [
                torch.zeros((3, cell_size, cell_size), dtype=torch.uint8)
                for _ in range(max_cols - len(row_images))
            ])
            tensors.append(x)
        
        tensor_image = torch.cat(tensors)
        grid_img_tensor = make_grid(tensor_image, nrow=max_cols)
        return Image.fromarray(
            grid_img_tensor.cpu().numpy().transpose(1, 2, 0)
        )


def pad2square(img: Image.Image) -> Image.Image:
    width = img.width
    height = img.height
    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), (0, 0, 0))
        result.paste(img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(img.mode, (height, height), (0, 0, 0))
        result.paste(img, ((height - width) // 2, 0))
        return result