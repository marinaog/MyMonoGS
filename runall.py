import subprocess

args_slam_srgb = [
    "--config", "configs/rgbd/rawslam/candles.yaml",
    "--eval"]
args_slam_raw = [
    "--config", "configs/rgbd/rawslam/candles_raw.yaml",
    "--eval"]
args_posteval = [
    "--scene", "candles",
    "--runnum", "5",
    "--data_type", "rawslam",
    "--both", "True",
    "--raw", "True"]

args = [args_slam_srgb, args_slam_raw, args_posteval]

scripts = [
    "slam.py",
    "slam.py",
    "post_evaluation.py",
]

for script, args in zip(scripts, args):
    print(f"Running {script}...")
    subprocess.run(["python3", script] + args, check=True)
    print(f"Finished {script}.")
    print("")
