import subprocess

args_slam_srgb = [
    "--config", "configs/rgbd/rawslam/boxes/boxes.yaml",
    "--eval"]
args_slam_raw = [
    "--config", "configs/rgbd/rawslam/boxes/boxes_raw_l1_mlp.yaml",
    "--eval"]
args_posteval = [
    "--scene", "boxes"]

args = [,args_slam_raw, args_slam_srgb]

scripts = [
    "slam.py",
    "slam.py",
    #"post_evaluation.py",
]

#script = "extract.py"
#print(f"Running {script}...")
#subprocess.run(["python3", script], check=True)
#print(f"Finished {script}.")
#print("")

for script, args in zip(scripts, args):
    print(f"Running {script}...")
    subprocess.run(["python", script] + args, check=True)
    print(f"Finished {script}.")
    print("")
