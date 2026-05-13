import json

import numpy as np

from prune import back_project, dynamic_static_split, make_transforms, pcd


def test_back_project_projects_identity_camera_points():
    depth = np.array([[[2.0, 4.0]]], dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    cam_c2w = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 1, axis=0)

    world_points = back_project(depth, intrinsic, cam_c2w)

    expected = np.array([[[1.0, 1.0, 2.0], [6.0, 2.0, 4.0]]], dtype=np.float32)
    np.testing.assert_allclose(world_points, expected)


def test_dynamic_static_split_partitions_points_by_threshold():
    cloud = pcd(
        xyz=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        rgb=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        prob_motion=np.array([[0.2], [0.7], [0.5]], dtype=np.float32),
        time_stamp=np.array([[0.0], [0.5], [1.0]], dtype=np.float32),
    )

    dynamic_cloud, static_cloud = dynamic_static_split(cloud, threshold=0.5)

    np.testing.assert_array_equal(dynamic_cloud.xyz, np.array([[1.0, 1.0, 1.0]], dtype=np.float32))
    np.testing.assert_array_equal(
        static_cloud.xyz,
        np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float32),
    )


def test_make_transforms_writes_expected_camera_metadata(tmp_path):
    intrinsic = np.array(
        [[1000.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    cam_c2w = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()

    make_transforms(
        intrinsic=intrinsic,
        cam_c2w=cam_c2w,
        save_dir=str(tmp_path),
        frame_dir=str(frame_dir),
        image_width=960,
        image_extension=".jpg",
        time_span=6.0,
    )

    train = json.loads((tmp_path / "transforms_train.json").read_text())
    test = json.loads((tmp_path / "transforms_test.json").read_text())

    assert train["w"] == 480
    assert train["h"] == 720
    assert train["fl_x"] == 500.0
    assert train["fl_y"] == 450.0
    assert train["cx"] == 160.0
    assert train["cy"] == 120.0
    assert [frame["time"] for frame in train["frames"]] == [0.0, 3.0, 6.0]
    assert train["frames"][0]["file_path"].endswith("frames/00001")
    assert train == test
