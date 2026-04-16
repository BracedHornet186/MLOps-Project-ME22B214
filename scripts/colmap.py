"""Adapted from https://www.kaggle.com/code/eduardtrulls/imc-2023-submission-example"""
# Code to manipulate a colmap database.
# Forked from https://github.com/colmap/colmap/blob/dev/scripts/python/database.py

# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)
# This script is based on an original implementation by True Price.
from __future__ import annotations

import copy
import gc
import sqlite3
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pycolmap
import tqdm
from PIL import ExifTags, Image

from data import DEFAULT_OUTLIER_SCENE_NAME
from localizers.base import PostLocalizer
from pipelines.common import Scene
from postprocesses.localize_pixloc import localize_pixloc
from postprocesses.localize_sfm import localize_sfm
from storage import (
    InMemoryKeypointStorage,
    InMemoryLocalFeatureStorage,
    InMemoryMatchingStorage,
    InMemoryTwoViewGeometryStorage,
)

MAX_IMAGE_ID = 2**31 - 1


CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = f"""CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {MAX_IMAGE_ID}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
"""

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)


class COLMAPDatabase(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = lambda: self.executescript(
            CREATE_DESCRIPTORS_TABLE
        )
        self.create_images_table = lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = lambda: self.executescript(
            CREATE_TWO_VIEW_GEOMETRIES_TABLE
        )
        self.create_keypoints_table = lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def add_camera(
        self, model, width, height, params, prior_focal_length=False, camera_id=None
    ):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                model,
                width,
                height,
                array_to_blob(params),
                prior_focal_length,
            ),
        )
        return cursor.lastrowid

    def add_image(
        self, name, camera_id, prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None
    ):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_id,
                name,
                camera_id,
                prior_q[0],
                prior_q[1],
                prior_q[2],
                prior_q[3],
                prior_t[0],
                prior_t[1],
                prior_t[2],
            ),
        )
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]

        keypoints = np.asarray(keypoints, np.float32)
        # move this block?
        keypoints[:, 0] += 0.5
        keypoints[:, 1] += 0.5
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),),
        )

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),),
        )

    def add_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),),
        )

    def add_two_view_geometry(
        self,
        image_id1,
        image_id2,
        matches,
        F=np.eye(3),
        E=np.eye(3),
        H=np.eye(3),
        config=2,
    ):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,)
            + matches.shape
            + (
                array_to_blob(matches),
                config,
                array_to_blob(F),
                array_to_blob(E),
                array_to_blob(H),
            ),
        )

    def remove_matches(self, image_id1, image_id2):
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        self.execute("DELETE from matches where pair_id = ?", (pair_id,))

    def remove_two_view_geometry(self, image_id1, image_id2):
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        self.execute("DELETE from two_view_geometries where pair_id = ?", (pair_id,))

    def get_image_id_to_name(self) -> dict[int, str]:
        images = self.execute("SELECT image_id, name FROM images").fetchall()
        image_id_to_name = {i: n for i, n in images}
        return image_id_to_name

    def get_name_to_image_id(self) -> dict[str, int]:
        images = self.execute("SELECT image_id, name FROM images").fetchall()
        image_id_to_name = {n: i for i, n in images}
        return image_id_to_name


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    return array.tostring()


def blob_to_array(blob, dtype, shape=(-1,)):
    return np.fromstring(blob, dtype=dtype).reshape(*shape)


def get_focal(image_path: str, err_on_default: bool = False) -> float:
    image = Image.open(image_path)
    max_size = max(image.size)

    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == "FocalLengthIn35mmFilm":
                focal_35mm = float(value)
                break

        if focal_35mm is not None:
            focal = focal_35mm / 35.0 * max_size

    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length")

        # failed to find it in exif, use prior
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    return focal


def create_camera(
    db: COLMAPDatabase, image_path: str, camera_model: str
) -> Optional[int]:
    image = Image.open(image_path)
    width, height = image.size

    focal = get_focal(image_path)

    if camera_model == "simple-pinhole":
        model = 0  # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    elif camera_model == "pinhole":
        model = 1  # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == "simple-radial":
        model = 2  # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == "opencv":
        model = 4  # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0.0, 0.0, 0.0, 0.0])
    else:
        raise ValueError(camera_model)

    return db.add_camera(model, width, height, param_arr)


def add_keypoints(
    db: COLMAPDatabase,
    scene: Scene,
    storage: Union[InMemoryLocalFeatureStorage, InMemoryKeypointStorage],
    camera_model: str,
    single_camera: bool = True,
) -> dict:
    camera_id = None
    fname_to_id = {}

    names = list(storage.keypoints.keys())
    for name in tqdm.tqdm(names, total=len(names), desc="colmap.add_keypoints"):
        keypoints = storage.keypoints[name]

        path = Path(scene.image_dir) / name
        if not path.exists():
            raise ValueError(f"Invalid image path {path}")

        if camera_id is None or not single_camera:
            camera_id = create_camera(db, str(path), camera_model)
        image_id = db.add_image(name, camera_id)
        fname_to_id[name] = image_id

        db.add_keypoints(image_id, keypoints)

    return fname_to_id


def add_matches(
    db: COLMAPDatabase,
    scene: Scene,
    storage: InMemoryMatchingStorage,
    fname_to_id: dict,
):
    added = set()
    n_keys = len(storage.matches.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    with tqdm.tqdm(total=n_total, desc="colmap.add_matches") as pbar:
        for key_1 in storage.matches.keys():
            group = storage.matches[key_1]
            for key_2 in group.keys():  # type: ignore
                id_1 = fname_to_id[key_1]
                id_2 = fname_to_id[key_2]

                pair_id = image_ids_to_pair_id(id_1, id_2)
                if pair_id in added:
                    warnings.warn(f"Pair {pair_id} ({id_1}, {id_2}) already added!")
                    continue

                matches = group[key_2]
                db.add_matches(id_1, id_2, matches)

                added.add(pair_id)
                pbar.update(1)


def add_two_view_geometries(
    db: COLMAPDatabase, storage: InMemoryTwoViewGeometryStorage, fname_to_id: dict
):
    added = set()
    for key1 in storage.inliers.keys():
        group = storage.inliers[key1]
        for key2 in group.keys():
            id1 = fname_to_id[key1]
            id2 = fname_to_id[key2]

            pair_id = image_ids_to_pair_id(id1, id2)
            if pair_id in added:
                warnings.warn(f"Pair {pair_id} ({id1}, {id2}) already added!")
                continue

            inliers = group[key2]
            F = storage.Fs[key1][key2]
            db.add_two_view_geometry(
                id1,
                id2,
                inliers,
                F=F,
                config=int(pycolmap.TwoViewGeometryConfiguration.UNCALIBRATED),
            )

            added.add(pair_id)


def sanity_check_id_name_mappings(
    fname_to_id: dict[str, int], database_path: str = "colmap.db"
) -> bool:
    db: COLMAPDatabase = COLMAPDatabase.connect(database_path)  # type: ignore
    id2name = db.get_image_id_to_name()
    for _id, name in id2name.items():
        print(fname_to_id[name], _id)
        assert fname_to_id[name] == _id
    print("---")
    for name, _id in fname_to_id.items():
        print(name, id2name[_id])
        assert name == id2name[_id]
    print("====")
    return True


def import_into_colmap(
    scene: Scene,
    f_storage: Union[InMemoryLocalFeatureStorage, InMemoryKeypointStorage],
    m_storage: InMemoryMatchingStorage,
    database_path: str = "colmap.db",
    camera_model: str = "simple-radial",
) -> dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(
        db, scene, f_storage, camera_model=camera_model, single_camera=single_camera
    )
    add_matches(db, scene, m_storage, fname_to_id)
    db.commit()

    return fname_to_id


def get_image_id_of_scene_graph_center(
    scene: Scene,
    database_path: str = "colmap.db",
) -> Optional[int]:
    try:
        scene_graph = scene.make_scene_graph()
    except Exception as e:
        print(f"[get_image_id_of_scene_graph_center] Error: {e}")
        return None

    db: COLMAPDatabase = COLMAPDatabase.connect(database_path)
    name2id = db.get_name_to_image_id()

    idx = int(scene_graph["center_node"])
    key = scene.idx_to_key(idx)
    image_id = name2id[key]
    return image_id


def remove_records(
    pair_keys_to_remove: list[tuple[str, str]], database_path: str = "colmap.db"
):
    db: COLMAPDatabase = COLMAPDatabase.connect(database_path)
    name2id = db.get_name_to_image_id()
    for key1, key2 in pair_keys_to_remove:
        image_id1 = name2id[key1]
        image_id2 = name2id[key2]
        db.remove_two_view_geometry(image_id1, image_id2)
    db.commit()


def import_two_view_geometories_into_colmap(
    g_storage: InMemoryTwoViewGeometryStorage,
    fname_to_id: dict[str, int],
    database_path: str = "colmap.db",
):
    db = COLMAPDatabase.connect(database_path)
    add_two_view_geometries(db, g_storage, fname_to_id)
    db.commit()


def export_two_view_geometries_from_colmap(
    g_storage: Optional[InMemoryTwoViewGeometryStorage] = None,
    database_path: str = "colmap.db",
) -> InMemoryTwoViewGeometryStorage:
    db: COLMAPDatabase = COLMAPDatabase.connect(database_path)  # type: ignore
    id2name = db.get_image_id_to_name()

    if g_storage is None:
        g_storage = InMemoryTwoViewGeometryStorage()

    for pair_id, rows, cols, data, F in db.execute(
        "SELECT pair_id, rows, cols, data, F FROM two_view_geometries"
    ):
        id1, id2 = pair_id_to_image_ids(pair_id)
        name1, name2 = id2name[id1], id2name[id2]
        if data is None:
            continue
        F = blob_to_array(F, np.float64).reshape(-1, 3)
        idx = blob_to_array(data, np.uint32, (rows, cols))
        g_storage.add(name1, name2, idx, F)

    return g_storage


def export_correspondence_graph(database_path: str) -> pycolmap.CorrespondenceGraph:
    db = pycolmap.Database(database_path)

    cg = pycolmap.CorrespondenceGraph()
    for image in db.read_all_images():
        kpts = db.read_keypoints(image.image_id)
        cg.add_image(image.image_id, kpts.shape[0])

    pair_ids, two_views = db.read_two_view_geometries()
    for pair_id, two_view in zip(pair_ids, two_views):
        id1, id2 = pair_id_to_image_ids(pair_id)
        if two_view is None:
            continue
        cg.add_correspondences(int(id1), int(id2), two_view.inlier_matches.copy())

    db.close()
    cg.finalize()
    return cg


def get_best_reconstruction(
    maps: dict,
    scene: Scene,
    keypoint_storage: InMemoryKeypointStorage,
    matching_storage: InMemoryMatchingStorage,
    fill_zero_Rt: bool = True,
    fill_nan_Rt: bool = False,
    fill_nearest_position: bool = False,
    failures_to_outliers: bool = False,
    output_best_rec_only: bool = True,
    use_localize_sfm: bool = False,
    use_localize_pixloc: bool = False,
    post_localizer: Optional[PostLocalizer] = None,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> tuple[dict, dict]:
    imgs_registered = 0
    best_idx = None

    outputs = {}
    infos = {"localization_by": {}}

    print(f"# of reconstruction models = {len(maps)}")

    assert output_best_rec_only
    if isinstance(maps, dict):
        for idx1, rec in maps.items():
            print(idx1, rec.summary())
            if len(rec.images) > imgs_registered:
                imgs_registered = len(rec.images)
                best_idx = idx1

    if best_idx is None:
        print("No reconstruction! -> outlier")
        return get_outlier_reconstructions(scene)

    if use_localize_sfm:
        # Add results from COLMAP results
        for k, im in maps[best_idx].images.items():
            key1 = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, im.name
            )
            meta = scene.data_schema.get_output_metadata(
                scene.dataset,
                scene.scene,
                im.name,
            )
            pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            outputs[key1] = {
                "R": copy.deepcopy(pose.rotation.matrix()),
                "t": copy.deepcopy(np.array(pose.translation)),
                "metadata": meta,
            }

        no_registered_query_full_keys = []
        for path in scene.image_paths:
            key1 = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, Path(path).name
            )
            if key1 not in outputs:
                infos["localization_by"][key1] = ""
                no_registered_query_full_keys.append(key1)
            else:
                infos["localization_by"][key1] = "colmap"

        if post_localizer:
            post_localizer.localize(
                maps[best_idx],
                no_registered_query_full_keys,
                outputs,
                scene,
                keypoint_storage,
                matching_storage,
                progress_bar=progress_bar,
            )
            no_registered_query_full_keys = []
            for path in scene.image_paths:
                key1 = scene.data_schema.format_output_key(
                    scene.dataset, scene.scene, Path(path).name
                )
                if key1 not in outputs:
                    no_registered_query_full_keys.append(key1)
                else:
                    infos["localization_by"][key1] = (
                        infos["localization_by"][key1] or "post_localizer"
                    )

        print("Use localize_sfm")
        if len(no_registered_query_full_keys) == 0:
            print("All images have been registered")
        else:
            print(
                f"{len(no_registered_query_full_keys)} images have not been registered yet"
            )
            try:
                additional_outputs = localize_sfm(
                    maps[best_idx],
                    no_registered_query_full_keys,
                    scene,
                    keypoint_storage,
                    matching_storage,
                )
                print(f"COLMAP registered: {len(outputs)}")
                print(f"localize_sfm registered: {len(additional_outputs)}")
                print(
                    f"No registered: {len(scene.image_paths) - len(outputs) - len(additional_outputs)}"
                )
                outputs.update(additional_outputs)
                for key1 in additional_outputs.keys():
                    infos["localization_by"][key1] = (
                        infos["localization_by"][key1] or "localize_sfm"
                    )
            except Exception as e:
                print(f"localize_sfm failed: {e}")
                pass
    elif use_localize_pixloc:
        # Add results from COLMAP results
        for k, im in maps[best_idx].images.items():
            key1 = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, im.name
            )
            pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            outputs[key1] = {
                "R": copy.deepcopy(pose.rotation.matrix()),
                "t": copy.deepcopy(np.array(pose.translation)),
            }

        no_registered_query_full_keys = []
        for path in scene.image_paths:
            key1 = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, Path(path).name
            )
            if key1 not in outputs:
                no_registered_query_full_keys.append(key1)

        print("Use localize_pixloc")
        if len(no_registered_query_full_keys) == 0:
            print("All images have been registered")
        else:
            print(
                f"{len(no_registered_query_full_keys)} images have not been registered yet"
            )
            del maps
            gc.collect()
            reference_sfm_path = str(scene.reconstruction_dir / f"{best_idx}")
            print(f"Loading rec from {reference_sfm_path}")
            try:
                additional_outputs = localize_pixloc(
                    reference_sfm_path, no_registered_query_full_keys, scene
                )
                print(f"COLMAP registered: {len(outputs)}")
                print(f"localize_pixloc registered: {len(additional_outputs)}")
                print(
                    f"No registered: {len(scene.image_paths) - len(outputs) - len(additional_outputs)}"
                )
                outputs.update(additional_outputs)
            except:
                pass
    else:
        for k, im in maps[best_idx].images.items():
            key1 = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, im.name
            )
            metadata = scene.data_schema.get_output_metadata(
                scene.dataset,
                scene.scene,
                im.name,
            )
            pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            outputs[key1] = {
                "R": copy.deepcopy(pose.rotation.matrix()),
                "t": copy.deepcopy(np.array(pose.translation)),
                "metadata": metadata,
            }
            infos["localization_by"][key1] = "colmap"

    if fill_nearest_position:
        for path in scene.image_paths:
            key1 = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, Path(path).name
            )
            if key1 not in outputs:
                proxy_key1 = None
                for nearest_key1 in scene.get_nearest_neighbors_full_keys(key1):
                    if nearest_key1 in outputs:
                        proxy_key1 = nearest_key1
                        break
                if proxy_key1:
                    metadata = scene.data_schema.get_output_metadata(
                        scene.dataset,
                        scene.scene,
                        Path(path).name,
                    )
                    outputs[key1] = {
                        "R": outputs[proxy_key1]["R"].copy(),
                        "t": outputs[proxy_key1]["t"].copy(),
                        "metadata": metadata,
                    }
                    print(
                        f"{scene}[{key1}] No best reconstructions found. "
                        f"Set the R and t from {proxy_key1}"
                    )
                    infos["localization_by"][key1] = "fill_nearest"
                else:
                    print(
                        f"{scene}[{key1}] No best reconstructions found. "
                        f"Besides, No proxy positions were found"
                    )

    if fill_zero_Rt:
        for path in scene.image_paths:
            key1 = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, Path(path).name
            )
            if key1 not in outputs:
                print(
                    f"{scene}[{key1}] No best reconstructions found. "
                    f"Set the zero matrix to R and t"
                )
                metadata = scene.data_schema.get_output_metadata(
                    scene.dataset,
                    scene.scene,
                    Path(path).name,
                )
                outputs[key1] = {
                    "R": np.eye(3),
                    "t": np.zeros(3),
                    "metadata": metadata,
                }
                infos["localization_by"][key1] = "fill_zero"
    elif fill_nan_Rt:
        for path in scene.image_paths:
            key1 = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, Path(path).name
            )
            if key1 not in outputs:
                print(
                    f"{scene}[{key1}] No best reconstructions found. "
                    f"Set the zero matrix containing np.nan to R and t"
                )
                metadata = scene.data_schema.get_output_metadata(
                    scene.dataset,
                    scene.scene,
                    Path(path).name,
                )
                R = np.eye(3)
                R[-1, -1] = np.nan
                outputs[key1] = {
                    "R": R,
                    "t": np.zeros(3),
                    "metadata": metadata,
                }
                infos["localization_by"][key1] = "fill_nan"

    if failures_to_outliers:
        for path in scene.image_paths:
            key1 = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, Path(path).name
            )
            if key1 not in outputs:
                print(
                    f"Reconstruction failed: "
                    f"{scene}[{key1}] -> {DEFAULT_OUTLIER_SCENE_NAME}"
                )
                metadata = scene.data_schema.get_output_metadata(
                    scene.dataset,
                    scene.scene,
                    Path(path).name,
                )
                R = np.eye(3) * np.nan
                t = np.zeros(3) * np.nan
                outputs[key1] = {
                    "R": R,
                    "t": t,
                    "cluster_name": DEFAULT_OUTLIER_SCENE_NAME,
                    "metadata": metadata,
                }
                infos["localization_by"][key1] = "fill_nan"

    return outputs, infos


def get_reconstructions(
    maps: dict,
    scene: Scene,
    keypoint_storage: InMemoryKeypointStorage,
    matching_storage: InMemoryMatchingStorage,
    post_localizer: Optional[PostLocalizer] = None,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> tuple[dict, dict]:
    outputs = {}
    infos = {"localization_by": {}}

    print(f"# of reconstruction models = {len(maps)}")

    registered_img_counts = {}
    for map_index, rec in maps.items():
        print(map_index, rec.summary())
        registered_img_counts[map_index] = len(rec.images)

    added_key = set()
    for map_index, count in sorted(
        registered_img_counts.items(), key=lambda d: d[1], reverse=True
    ):
        rec = maps[map_index]

        for k, im in rec.images.items():
            key1 = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, im.name
            )
            if key1 in added_key:
                continue

            metadata = scene.data_schema.get_output_metadata(
                scene.dataset,
                scene.scene,
                im.name,
            )
            pose = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            outputs[key1] = {
                "R": copy.deepcopy(pose.rotation.matrix()),
                "t": copy.deepcopy(np.array(pose.translation)),
                "cluster_name": f"cluster{map_index}",
                "metadata": metadata,
            }
            infos["localization_by"][key1] = "colmap"
            added_key.add(key1)

    for path in scene.image_paths:
        key1 = scene.data_schema.format_output_key(
            scene.dataset, scene.scene, Path(path).name
        )
        metadata = scene.data_schema.get_output_metadata(
            scene.dataset,
            scene.scene,
            Path(path).name,
        )
        if key1 not in outputs:
            print(
                f"{scene}[{key1}] No reconstructions found. "
                f"Set the nan matrix to R and t"
            )
            R = np.eye(3) * np.nan
            t = np.zeros(3) * np.nan
            outputs[key1] = {
                "R": R,
                "t": t,
                "cluster_name": DEFAULT_OUTLIER_SCENE_NAME,
                "metadata": metadata,
            }
            infos["localization_by"][key1] = "fill_nan"

    return outputs, infos


def get_outlier_reconstructions(scene: Scene) -> tuple[dict, dict]:
    outputs = {}
    infos = {"localization_by": {}}

    for path in scene.image_paths:
        key1 = scene.data_schema.format_output_key(
            scene.dataset, scene.scene, Path(path).name
        )
        metadata = scene.data_schema.get_output_metadata(
            scene.dataset,
            scene.scene,
            Path(path).name,
        )
        print(f"{scene}[{key1}] Scene based on outliers. Set the nan matrix to R and t")
        R = np.eye(3) * np.nan
        t = np.zeros(3) * np.nan
        outputs[key1] = {
            "R": R,
            "t": t,
            "cluster_name": DEFAULT_OUTLIER_SCENE_NAME,
            "metadata": metadata,
        }
        infos["localization_by"][key1] = "fill_nan"

    return outputs, infos
