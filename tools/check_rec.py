from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pycolmap


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-r", "--rec")
    return parser.parse_args()


def main():
    args = parse_args()

    rec = pycolmap.Reconstruction(args.rec)

    track_length_counts = defaultdict(int)
    track_lengths = []
    errors = []
    for point3D_id, point3D in rec.points3D.items():
        assert isinstance(point3D, pycolmap.Point3D)
        track_lengths.append(point3D.track.length())
        track_length_counts[point3D.track.length()] += 1
        errors.append(point3D.error)
        for element in point3D.track.elements:
            image_id = element.image_id
            point2D_idx = element.point2D_idx
            img = rec.images[image_id]
            print(img.points2D[point2D_idx], point3D.track.length())
        # print(point3D_id, point3D.track.length(), point3D.error, point3D.xyz)
        # print(point3D.track)

    print(f"# of images: {rec.num_images()}")
    print(f"# of reg images: {rec.num_reg_images()}")
    print(f"# of cameras: {rec.num_cameras()}")
    print(f"# of points3D: {len(rec.points3D)}")
    print(f"Sum of track lengths: {np.sum(track_lengths)}")
    print(f"Avg of track lengths: {np.mean(track_lengths)}")
    print(f"Median of track lengths: {np.median(track_lengths)}")
    print(f"Track length: {track_length_counts}")
    print(f"Mean point3D error: {np.mean(errors)}")


if __name__ == "__main__":
    main()
