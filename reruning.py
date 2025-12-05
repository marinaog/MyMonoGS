import os
import cv2
import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- Configuration ---
dataset_folder = r"Record\recordings\objects_from_front"
dataset_folder = r"Record\recordings\dark_kitchen "
rgb_folder = os.path.join(dataset_folder, "sRGB")
depth_folder = os.path.join(dataset_folder, "depth")
trajectory_file = os.path.join(dataset_folder, "groundtruth.txt")

# Initialize rerun
rr.init("slam_visualization", spawn=True)


# Read trajectory file
poses = []
with open(trajectory_file, "r") as f:
    first = True
    for line in f:
        if first:
            first = False
            continue
        parts = line.strip().split()
        if len(parts) != 8:
            print("Length is not the expected one", len(parts), line)
            break
        frame, timestamp, x, y, z, rx, ry, rz = map(float, parts)
        poses.append(
            {
                "frame": int(frame),
                "timestamp": timestamp,
                "position": np.array([z, x, y]),
                "rotation": np.array([rz, rx, ry]),
            }
        )

# Log world origin for reference
rr.log("world/origin", rr.Transform3D(translation=[0, 0, 0]))
rr.log(
    "trajectory/images",
    rr.Pinhole(
        resolution=(1920, 1080),
        focal_length=[1380, 1379],
        principal_point=[961, 568],
        camera_xyz=rr.components.ViewCoordinates.LBU
    ),
    static=True
)
trajectory_so_far = []
# Iterate through poses and visualize
for ind, pose in enumerate(poses):
    frame = pose["frame"]
    time = pose["timestamp"]
    rotation = R.from_euler("xyz", pose["rotation"], degrees=True)
    trajectory_so_far.append(pose["position"])

    # Load RGB and depth images
    rgb_path = os.path.join(rgb_folder, f"{frame}.png")
    depth_path = os.path.join(depth_folder, f"{frame}.png")

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f"Missing RGB or depth image for frame {frame}")
        continue

    if not os.path.exists(rgb_path):
        print(f"Missing RGB image at {rgb_path}")
        continue
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    quat_xyzw = rotation.as_quat()  # [x, y, z, w]

    # Log images and depth
    rr.set_time("frame", sequence=frame)
    rr.log(
        "trajectory/images",
        rr.Transform3D(
            translation=pose["position"],
            quaternion=rr.Quaternion(
                xyzw=quat_xyzw,
            ),
        ),
    )
    rr.log(
        f"trajectory/images/rgb", rr.Image(rgb).compress(jpeg_quality=90)
    ) 
    rr.log(f"trajectory/images/depth", rr.DepthImage(depth, meter=1000))  # depth scale

    # Log growing trajectory line
    rr.log("trajectory", rr.LineStrips3D([trajectory_so_far]))