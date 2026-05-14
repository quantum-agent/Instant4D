import argparse
import json
import os
from typing import NamedTuple

import numpy as np


class pcd(NamedTuple):
    xyz: np.ndarray
    rgb: np.ndarray
    prob_motion: np.ndarray
    time_stamp: np.ndarray


def back_project(depth, intrinsic, cam_c2w):
    """
    Vectorized back-projection of depth maps to 3D points in world coordinates.

    Args:
        depth: B, H, W numpy array
        intrinsic: 3, 3 numpy array
        cam_c2w: B, 4, 4 numpy array

    Returns:
        xyz: B, H*W, 3 numpy array of 3D points in world coordinates
    """
    B, H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.reshape(-1) + 0.5
    y = y.reshape(-1) + 0.5

    homogeneous_coords = np.vstack((x, y, np.ones_like(x)))
    cam_points = np.linalg.inv(intrinsic) @ homogeneous_coords
    depth_flat = depth.reshape(B, -1)
    cam_points_expanded = np.tile(cam_points[None, :, :], (B, 1, 1))
    cam_points_scaled = cam_points_expanded * depth_flat[:, None, :]

    world_points = np.zeros((B, H * W, 3))
    for b in range(B):
        world_points[b] = (cam_points_scaled[b].T @ cam_c2w[b, :3, :3].T) + cam_c2w[b, :3, 3]

    return world_points


def read_droid_data(droid_path, motion_path, save_dir):
    import cv2

    droid_data = np.load(droid_path)
    print(droid_data.keys())

    print(droid_data['images'].shape)
    print(droid_data['depths'].shape)
    print(droid_data['intrinsic'].shape)
    print(droid_data['cam_c2w'].shape)

    color = droid_data['images']
    depth = droid_data['depths']
    intrinsic = droid_data['intrinsic']
    cam_c2w = droid_data['cam_c2w']
    motion_prob = np.load(motion_path)

    B = color.shape[0]
    H_new = color.shape[1]
    W_new = color.shape[2]

    resized_motion = np.empty((B, H_new, W_new), dtype=np.float32)
    for i in range(motion_prob.shape[0]):
        resized_motion[i] = cv2.resize(motion_prob[i], (W_new, H_new), interpolation=cv2.INTER_LINEAR)

    print(f"motion_prob shape: {resized_motion.shape}")
    print(f"color shape: {color.shape}")
    print(f"depth shape: {depth.shape}")
    print(f"cam_c2w shape: {cam_c2w.shape}")
    print(f"intrinsic shape: {intrinsic.shape}")

    return depth, color, resized_motion.reshape(-1, 1), intrinsic, cam_c2w


def process_data(depth, color, motion_prob, intrinsic, cam_c2w, time_span=3.0):
    B, _, _ = depth.shape

    xyz = back_project(depth, intrinsic, cam_c2w).reshape(-1, 3)
    rgb = color.reshape(-1, 3).astype(np.float32) / 255.0
    time_stamp = np.repeat(np.arange(B).astype(np.float32) / B * time_span, xyz.shape[0] // B).reshape(-1, 1)
    prob_motion = motion_prob

    print(f"prob_motion range from {np.min(prob_motion)} to {np.max(prob_motion)}")
    print(f"prob_motion shape: {prob_motion.shape}")

    return pcd(xyz=xyz, rgb=rgb, prob_motion=prob_motion, time_stamp=time_stamp)


def dynamic_static_split(pc, threshold=0.5):
    dynamic_region = (pc.prob_motion > threshold).reshape(-1)
    static_region = ~dynamic_region

    print(f"shape of dynamic region: {dynamic_region.shape}")
    print(f"shape of static region: {static_region.shape}")
    print(f"shape of pc xyz: {pc.xyz.shape}")

    dynamic_pcd = pcd(
        xyz=pc.xyz[dynamic_region],
        rgb=pc.rgb[dynamic_region],
        prob_motion=pc.prob_motion[dynamic_region],
        time_stamp=pc.time_stamp[dynamic_region],
    )
    static_pcd = pcd(
        xyz=pc.xyz[static_region],
        rgb=pc.rgb[static_region],
        prob_motion=pc.prob_motion[static_region],
        time_stamp=pc.time_stamp[static_region],
    )

    return dynamic_pcd, static_pcd


def make_transforms(intrinsic, cam_c2w, save_dir, frame_dir, image_width, image_extension=".png", time_span=3.0):
    scale_factor = 480 / image_width
    B = cam_c2w.shape[0]
    frame_dir = os.path.abspath(frame_dir)

    dict_to_save = {
        "w": 480,
        "h": 720,
        "fl_x": (intrinsic[0, 0] * scale_factor).item(),
        "fl_y": (intrinsic[1, 1] * scale_factor).item(),
        "cx": (intrinsic[0, 2] * scale_factor).item(),
        "cy": (intrinsic[1, 2] * scale_factor).item(),
    }

    selected = range(B)
    print(f"selected_len: {len(list(selected))}")

    frames = []
    ext_without_dot = image_extension[1:] if image_extension.startswith(".") else image_extension
    for i in range(B):
        frame_stem = os.path.join(frame_dir, f"{i + 1:05d}")
        if ext_without_dot:
            # transforms readers append extension, so store the stem only
            frame_stem = frame_stem
        frames.append(
            {
                "file_path": frame_stem,
                "transform_matrix": cam_c2w[i].tolist(),
                "time": i / (B - 1) * time_span if B > 1 else 0.0,
            }
        )

    dict_to_save["frames"] = frames
    with open(f"{save_dir}/transforms_train.json", "w") as f:
        json.dump(dict_to_save, f, indent=4)
    with open(f"{save_dir}/transforms_test.json", "w") as f:
        json.dump(dict_to_save, f, indent=4)


def _downsample_with_rgb_channels(pcu, voxel_size, xyz, rgb, *extra_attributes):
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    rgb = np.ascontiguousarray(rgb, dtype=np.float64)
    extras = [np.ascontiguousarray(attr, dtype=np.float64) for attr in extra_attributes]

    downsampled = pcu.downsample_point_cloud_on_voxel_grid(
        voxel_size,
        xyz,
        rgb[:, 0],
        rgb[:, 1],
        rgb[:, 2],
        *extras,
    )
    xyz_out = downsampled[0]
    rgb_out = np.stack(downsampled[1:4], axis=1)
    extra_out = downsampled[4:]
    return (xyz_out, rgb_out, *extra_out)


def voxel_filter(
    droid_path,
    motion_path,
    save_dir,
    frame_dir,
    image_extension=".png",
    frame_stride=3,
    time_span=3.0,
    dynamic_threshold=0.5,
):
    import point_cloud_utils as pcu

    depth, color, motion_prob, intrinsic, cam_c2w = read_droid_data(droid_path, motion_path, save_dir)

    B, _, _ = depth.shape
    print(f"depth shape: {depth.shape}")
    _, _, image_width = depth.shape
    make_transforms(intrinsic, cam_c2w, save_dir, frame_dir, image_width=image_width, image_extension=image_extension, time_span=time_span)

    color = color[::frame_stride]
    depth = depth[::frame_stride]
    cam_c2w = cam_c2w[::frame_stride]
    motion_prob = motion_prob[::frame_stride]

    motion_prob = np.concatenate(motion_prob, axis=0).astype(np.float32)

    print(f"motion_prob shape: {motion_prob.shape}")
    print(f"color shape: {color.shape}")
    print(f"depth shape: {depth.shape}")
    print(f"cam_c2w shape: {cam_c2w.shape}")

    pc = process_data(depth, color, motion_prob, intrinsic, cam_c2w, time_span=time_span)
    pcd_dynamic, pcd_static = dynamic_static_split(pc, threshold=dynamic_threshold)

    mean_depth = np.mean(depth[0])
    focal = intrinsic[0, 0]

    voxel_size_dynamic = mean_depth / focal * 0.5
    voxel_size_static = mean_depth / focal * 2

    xyz_static, rgb_static, prob_motion_static = _downsample_with_rgb_channels(
        pcu,
        voxel_size_static,
        pcd_static.xyz,
        pcd_static.rgb,
        np.asarray(pcd_static.prob_motion).reshape(-1),
    )

    xyz_dynamic, rgb_dynamic, prob_motion_dynamic, time_stamp_dynamic = _downsample_with_rgb_channels(
        pcu,
        voxel_size_dynamic,
        pcd_dynamic.xyz,
        pcd_dynamic.rgb,
        np.asarray(pcd_dynamic.prob_motion).reshape(-1),
        np.asarray(pcd_dynamic.time_stamp).reshape(-1),
    )

    time_stamp_static = np.repeat(1, xyz_static.shape[0])
    scale_time_static = np.repeat(time_span, xyz_static.shape[0])
    scale_time_dynamic = np.repeat(time_span / ((B - 1) * 10), xyz_dynamic.shape[0]) if B > 1 else np.repeat(1.0, xyz_dynamic.shape[0])

    xyz_sampled = np.concatenate([xyz_static, xyz_dynamic], axis=0)
    rgb_sampled = np.concatenate([rgb_static, rgb_dynamic], axis=0)
    prob_motion_sampled = np.concatenate([np.asarray(prob_motion_static).squeeze(), np.asarray(prob_motion_dynamic).squeeze()], axis=0)
    time_stamp_sampled = np.concatenate([time_stamp_static.squeeze(), np.asarray(time_stamp_dynamic).squeeze()], axis=0)
    scale_time_sampled = np.concatenate([scale_time_static, scale_time_dynamic], axis=0)

    print("--------------------------------")
    print(f"xyz_static: {xyz_static.shape}")
    print(f"xyz_dynamic: {pcd_dynamic.xyz.shape}")
    print(f"xyz_sampled: {xyz_sampled.shape}")
    print(f"time_stamp: {time_stamp_sampled.shape}")
    print(f"prob_motion: {prob_motion_sampled.shape}")
    print(f"scale_time: {scale_time_sampled.shape}")

    np.savez(
        f"{save_dir}/filtered_cvd.npz",
        xyz=xyz_sampled,
        rgb=rgb_sampled,
        prob_motion=prob_motion_sampled,
        time_stamp=time_stamp_sampled,
        scale_time=scale_time_sampled,
        intrinsic=intrinsic,
        cam_c2w=cam_c2w,
    )


def main():
    parser = argparse.ArgumentParser(description="Create filtered_cvd.npz scene bundles from Mega-SAM outputs.")
    parser.add_argument("--scene", dest="scenes", action="append", default=[], help="Scene name; can be repeated")
    parser.add_argument("--droid-dir", required=True, help="Directory containing CVD npz outputs")
    parser.add_argument("--motion-dir", required=True, help="Directory containing motion_prob.npy outputs")
    parser.add_argument("--save-dir", required=True, help="Directory where per-scene filtered outputs will be written")
    parser.add_argument("--image-root", required=True, help="Root directory containing per-scene frame folders")
    parser.add_argument("--droid-file-template", default="{scene}_sgd_cvd_hr.npz")
    parser.add_argument("--motion-file-template", default="{scene}/motion_prob.npy")
    parser.add_argument("--frame-dir-template", default="{scene}")
    parser.add_argument("--image-extension", default=".png")
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--time-span", type=float, default=3.0)
    parser.add_argument("--dynamic-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.scenes:
        parser.error("At least one --scene is required")

    for scene in args.scenes:
        droid_path = os.path.join(args.droid_dir, args.droid_file_template.format(scene=scene))
        motion_path = os.path.join(args.motion_dir, args.motion_file_template.format(scene=scene))
        save_path = os.path.join(args.save_dir, scene)
        frame_dir = os.path.join(args.image_root, args.frame_dir_template.format(scene=scene))

        os.makedirs(save_path, exist_ok=True)
        voxel_filter(
            droid_path=droid_path,
            motion_path=motion_path,
            save_dir=save_path,
            frame_dir=frame_dir,
            image_extension=args.image_extension,
            frame_stride=args.frame_stride,
            time_span=args.time_span,
            dynamic_threshold=args.dynamic_threshold,
        )


if __name__ == "__main__":
    main()
