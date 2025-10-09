import os
import cv2
import rawpy
import cv2
from tqdm import tqdm
import numpy as np

def load_raw_image(file_path, width, height):
    """
    Load a 16-bit raw grayscale image and return it as a NumPy array.
    """
    with open(file_path, "rb") as f:
        raw_data = np.fromfile(f, dtype=np.uint16)
    if raw_data.shape[0] != height * width:
        print("Raw data shape:", raw_data.shape, "Expected shape:", height * width, "Frame", file_path)
    else:
        image = raw_data.reshape((height, width))
        return image


def process_RAW(frame, raw_file, srgb_folder, raw_folder, green, linear, width=1920, height=1080, bayer_pattern=cv2.COLOR_BAYER_GR2BGR):
    output_raw_path = os.path.join(raw_folder, str(frame) + '.png')
    output_ldr_path = os.path.join(srgb_folder, str(frame) + '.png')

    # Save theº LDR sRGB ones:
    with rawpy.imread(raw_file) as raw:
        rgb_srgb = raw.postprocess(
                    use_camera_wb=True,  # Use white balance from metadata
                    output_color=rawpy.ColorSpace.sRGB,  # Convert to sRGB color space
                    gamma=(2.2, 0.0),  # Apply sRGB gamma (standard 2.2)
                    no_auto_bright=True,  # Do not apply automatic brightness adjustments
                    output_bps=8,  # Output 8-bit per channel 
                )
        cv2.imwrite(output_ldr_path, rgb_srgb)

        # Save the HDR raw image
        # If save RAW image as not green...
        if not green:
            # If it doesnt exist already
            if not os.path.isfile(output_raw_path): 
                if linear:
                    gamma_curve = (1.0, 0.0)
                else:
                    gamma_curve = (2.2, 0.0)
                raw_srgb = raw.postprocess(
                                    use_camera_wb=True,  # Use white balance from metadata
                                    output_color=rawpy.ColorSpace.sRGB,  # Convert to sRGB color space
                                    gamma=gamma_curve,  # gamma curve
                                    no_auto_bright=True,  # Do not apply automatic brightness adjustments
                                    output_bps=16,  # Output 16-bit per channel
                                )
                cv2.imwrite(output_raw_path, raw_srgb)
    
    # If save RAW image as green...
    if green:
        # If it doesnt exist already
        if not os.path.isfile(output_raw_path): 
            # 1. Process .dng RAW image
            raw_image = load_raw_image(raw_file, width, height)
            bgr_image = cv2.cvtColor(raw_image, bayer_pattern)

            # 2. Demosaic RAW image
            green_raw_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            # 3. Save it with 16 bits
            cv2.imwrite(output_raw_path, green_raw_image)

        
def testing_record(gt_lines, raw_folder, srgb_folder):
    # First frame
    first_frame = gt_lines[0].strip().split()[0]

    # The last line might be empty
    i = 0
    while True:
        line = gt_lines[-i-1]
        parts = line.strip().split()
        if len(parts) == 8:
            break
        else:
            print(f"#{i+1} last line of groundtruth.txt is empty")
            i += 1

    num_frames = int(parts[0]) - int(first_frame) + 1

    # Test RAW images    
    raw_count = sum(1 for entry in os.listdir(raw_folder))

    raw_sample = cv2.imread(
        os.path.join(raw_folder, os.listdir(raw_folder)[0]), cv2.IMREAD_UNCHANGED
    )
    rgb_bit_depth = raw_sample.dtype.itemsize * 8

    if raw_count != num_frames:
        print(
            f"❌ Error:  Some RAW images haven't been converted. RAW images:({raw_count}) vs. recorded images in gt.txt: ({num_frames})"
        )
        print("Missing RGB images:")
        for i in range(int(first_frame), int(parts[0])+1):
            if not os.path.exists(os.path.join(raw_folder, f"{i}.png")):
                print(f"{i}.png missing")
    else:
        if rgb_bit_depth != 16:
            print(f"❌ Error: RGB images bit depth ({rgb_bit_depth}) is not 16 bits.")
        else:
            print(f"✅ RGB images are correct and 16 bits.")
        

    # Test sRGB images
    srgb_count = sum(1 for entry in os.listdir(srgb_folder))

    srgb_sample = cv2.imread(
        os.path.join(srgb_folder, os.listdir(srgb_folder)[0]), cv2.IMREAD_UNCHANGED
    )
    srgb_bit_depth = srgb_sample.dtype.itemsize * 8

    if srgb_count != num_frames:
        print(
            f"❌ Error: Some sRGB images haven't been converted. sRGB images: ({srgb_count}) vs. RAW images: ({num_frames})"
        )
        print("Missing sRGB images:")
        for i in range(int(first_frame), int(parts[0])+1):
            if not os.path.exists(os.path.join(srgb_folder, f"{i}.png")):
                print(f"{i}.png missing")
    else:
        if srgb_bit_depth != 8:
            print(f"❌ Error: sRGB images bit depth ({srgb_bit_depth}) is not 8 bits.")
        else:
            print(f"✅ sRGB images are correct and 8 bits.")


def main():  
    # Select dataset   
    main_folder = r"datasets/rawslam/coat_rack"  # Change this to your dataset folder
    green = False # True for green RAW, False for color sRGB RAW
    linear = True # This should be always in linear

    # Create and load directories 
    dng_folder = os.path.join(main_folder, "raw")
    print(dng_folder)
    
    srgb_folder = os.path.join(main_folder, "sRGB")
    if green:
        raw_folder = os.path.join(main_folder, "raw_green")
        print("🟢 Green RAW")
    else:
        if linear:
            raw_folder = os.path.join(main_folder, "raw_linear_sRGB")
            print("🌈 Non green RAW + linear")
        else:            
            raw_folder = os.path.join(main_folder, "raw_sRGB")
            print("🌈 Non green RAW + linear")
    
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(srgb_folder, exist_ok=True)

    groundtruth = os.path.join(main_folder, "groundtruth.txt")

    # Loop over groundtruth frames to extract images
    with open(groundtruth, "r") as f:
        gt_lines = f.readlines()
    gt_lines = gt_lines[1:] 

    print("🔴 Extracting RAW and LDR images")
    for line in tqdm(gt_lines):
        parts = line.strip().split()
        if len(parts) != 8:
            continue # Skip empty row (last ones)
        frame = int(parts[0])

        raw_file = os.path.join(dng_folder, str(frame) + '.dng')
        
        process_RAW(frame, raw_file, srgb_folder, raw_folder, green, linear)

    
    print("🔎 Inspecting files")
    testing_record(gt_lines, raw_folder, srgb_folder) 

    
if __name__ == "__main__":
    main()