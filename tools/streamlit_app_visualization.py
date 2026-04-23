from pathlib import Path

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from scripts.data import resolve_model_path
from pipelines.scene import make_scene_graph
from pipelines.snapshot import SceneSnapshot, find_snapshots
from models.config import DoppelGangersModelConfig
from postprocesses.doppelgangers import DoppelGangersTwoViewGeometryPruner
from matchers.visualizer import draw as plot_matches

st.set_page_config(layout="wide")


@st.cache_resource
def load_snapshot(snapshot_file: str) -> SceneSnapshot:
    snapshot = SceneSnapshot.load(snapshot_file=snapshot_file)
    return snapshot


@st.cache_resource
def load_doppelgangers_classifier() -> DoppelGangersTwoViewGeometryPruner:
    classifier = DoppelGangersTwoViewGeometryPruner(
        DoppelGangersModelConfig(
            weight_path="DOPPELGANGERS",
            loftr_weight_path="DOPPELGANGERS_LOFTR",
            threshold=0.8,
        ),
        device=torch.device("cuda:0"),
    )
    return classifier


def plot_keypoints(image: Image.Image, keypoints: np.ndarray):
    img = np.array(image)
    fig, ax = plt.subplots(1, 2, figsize=(10, 20))
    ax[0].imshow(img)
    ax[1].imshow(img)
    ax[1].scatter(keypoints[:, 0], keypoints[:, 1], s=0.5, c="red")
    st.pyplot(fig=fig, clear_figure=True, use_container_width=False)


def draw_verified_matches(
    img1: Image.Image, img2: Image.Image, kpts1: np.ndarray, kpts2: np.ndarray
):
    _kpts1 = [cv2.KeyPoint(x, y, 1.0) for x, y in kpts1]
    _kpts2 = [cv2.KeyPoint(x, y, 1.0) for x, y in kpts2]
    matches_1to2 = [cv2.DMatch(idx, idx, 0.0) for idx in range(len(_kpts1))]
    image = cv2.drawMatches(
        np.array(img1), _kpts1, np.array(img2), _kpts2, matches_1to2, None
    )

    fig, ax = plt.subplots()
    ax.imshow(image)
    st.pyplot(fig=fig, clear_figure=True, use_container_width=False)


def main():
    snapshot_dict = find_snapshots()
    classifier = load_doppelgangers_classifier()

    pipeline_id = str(st.sidebar.selectbox("Pipeline ID", options=snapshot_dict.keys()))
    snapshot_file = str(
        st.sidebar.selectbox(
            "Snapshot",
            options=snapshot_dict[pipeline_id],
            format_func=lambda x: Path(x).stem,
        )
    )
    pair_id = int(st.sidebar.number_input("Pair ID", min_value=0))

    show_overlap = st.sidebar.checkbox("Show Overlap", value=True)
    show_roi = st.sidebar.checkbox("Show RoI", value=True)
    show_mask = st.sidebar.checkbox("Show Mask", value=True)

    snapshot = load_snapshot(snapshot_file)

    assert snapshot.scene.shortlist
    pair = snapshot.scene.shortlist[pair_id]
    i, j = pair

    path1 = snapshot.scene.image_paths[i]
    path2 = snapshot.scene.image_paths[j]

    orientation1 = snapshot.scene.get_orientation_degree(path1)
    orientation2 = snapshot.scene.get_orientation_degree(path2)
    doppelgangers_score = classifier.classify(path1, path2, scene=snapshot.scene)

    img1 = Image.open(path1)
    img2 = Image.open(path2)

    bbox1, bbox2 = snapshot.scene.get_overlap_regions_if_exists(path1, path2)
    if show_overlap and bbox1 is not None and bbox2 is not None:
        draw1 = ImageDraw.ImageDraw(img1)
        draw1.rectangle(bbox1.tolist(), outline=(0, 255, 0), width=3)
        draw2 = ImageDraw.ImageDraw(img2)
        draw2.rectangle(bbox2.tolist(), outline=(0, 255, 0), width=3)

    bbox1 = snapshot.scene.bboxes.get(str(path1))
    bbox2 = snapshot.scene.bboxes.get(str(path2))
    if show_roi and bbox1 is not None:
        draw1 = ImageDraw.ImageDraw(img1)
        draw1.rectangle(bbox1.tolist(), outline=(0, 0, 255), width=3)
    if show_roi and bbox2 is not None:
        draw2 = ImageDraw.ImageDraw(img2)
        draw2.rectangle(bbox2.tolist(), outline=(0, 0, 255), width=3)

    if hasattr(snapshot.scene, "mask_bboxes"):
        bboxes1 = snapshot.scene.get_mask_regions(path1)
        bboxes2 = snapshot.scene.get_mask_regions(path2)
        if show_mask:
            for bbox1 in bboxes1:
                if bbox1 is not None:
                    draw1 = ImageDraw.ImageDraw(img1)
                    draw1.rectangle(bbox1.tolist(), outline=(255, 0, 0), width=3)
            for bbox2 in bboxes2:
                if bbox2 is not None:
                    draw2 = ImageDraw.ImageDraw(img2)
                    draw2.rectangle(bbox2.tolist(), outline=(255, 0, 0), width=3)

    st.image([img1, img2])
    st.json(
        {
            "orientation1": orientation1,
            "orientation2": orientation2,
            "doppelgangers_score": doppelgangers_score,
        }
    )

    kpts1 = snapshot.keypoint_storage.get(path1)
    kpts2 = snapshot.keypoint_storage.get(path2)
    assert isinstance(kpts1, np.ndarray)
    assert isinstance(kpts2, np.ndarray)

    img1 = Image.open(path1)
    img2 = Image.open(path2)
    plot_keypoints(img1, kpts1)
    st.caption(f"Keypoints: {len(kpts1)}")
    plot_keypoints(img2, kpts2)
    st.caption(f"Keypoints: {len(kpts2)}")

    if not snapshot.matching_storage.has(path1, path2):
        return

    inliers = snapshot.matching_storage.get(path1, path2)
    mkpts1 = kpts1[inliers[:, 0]]
    mkpts2 = kpts2[inliers[:, 1]]
    # matching_fig, ax = plt.subplots()
    # plot_matches(path1, path2, mkpts1, mkpts2, ax=ax)
    # st.pyplot(fig=matching_fig, clear_figure=True, use_container_width=False)
    draw_verified_matches(img1, img2, mkpts1, mkpts2)
    st.caption(f"Matches: {len(mkpts1)}")

    if snapshot.two_view_geometry_storage:
        idx, F = snapshot.two_view_geometry_storage.get(path1, path2)
        verified_mkpts1 = kpts1[idx[:, 0]]
        verified_mkpts2 = kpts2[idx[:, 1]]

        draw_verified_matches(img1, img2, verified_mkpts1, verified_mkpts2)
        st.caption(f"Verified matches: {len(verified_mkpts1)}")


if __name__ == "__main__":
    main()
