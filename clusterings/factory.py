from __future__ import annotations

import torch

from clusterings.base import Clustering
from clusterings.config import ClusteringConfig
from clusterings.connected_component import ConnectedComponentClustering
from clusterings.dbscan import DBSCANClustering
from clusterings.debug import DebugArraySplitClustering
from clusterings.fps import VGGTFPSClustering
from clusterings.fps_mast3r import MASt3RFPSClustering
from clusterings.vggt import VGGTClustering
from retrievers.factory import create_retriever
from shortlists.global_descriptor import create_global_descriptor_extractor


def create_clustering(conf: ClusteringConfig, device: torch.device) -> Clustering:
    if conf.type == "connected_component":
        assert conf.connected_component
        if conf.connected_component.global_desc:
            extractor = create_global_descriptor_extractor(
                conf.connected_component.global_desc,
                device=device,
            )
            batch_size = conf.connected_component.global_desc.batch_size
        elif conf.connected_component.retriever:
            extractor = create_retriever(
                conf.connected_component.retriever, device=device
            )
            batch_size = 1
        else:
            raise ValueError

        clustering = ConnectedComponentClustering(
            extractor,
            conf.connected_component.topk,
            conf.connected_component.dist_threshold,
            conf.connected_component.min_cluster_size,
            conf.connected_component.use_noisy_cluster_as_one_cluster,
            degree_threshold=conf.connected_component.degree_threshold,
            batch_size=batch_size,
        )
    elif conf.type == "dbscan":
        assert conf.dbscan
        extractor = create_global_descriptor_extractor(
            conf.dbscan.global_desc,
            device=device,
        )
        clustering = DBSCANClustering(
            extractor,
            batch_size=conf.dbscan.global_desc.batch_size,
            eps=conf.dbscan.eps,
            min_samples=conf.dbscan.min_samples,
        )
    elif conf.type == "debug_array_split":
        assert conf.debug_array_split
        clustering = DebugArraySplitClustering(
            n_clusters=conf.debug_array_split.n_clusters
        )
    elif conf.type == "vggt":
        assert conf.vggt
        clustering = VGGTClustering(conf.vggt, device=device)
    elif conf.type == "vggt_fps":
        assert conf.vggt_fps
        clustering = VGGTFPSClustering(conf.vggt_fps, device=device)
    elif conf.type == "mast3r_fps":
        assert conf.mast3r_fps
        clustering = MASt3RFPSClustering(conf.mast3r_fps, device=device)
    else:
        raise NotImplementedError(conf.type)

    return clustering
