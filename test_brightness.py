import os
import cv2
import rawpy
from tqdm import tqdm
import numpy as np

# ... (función load_raw_image se mantiene igual) ...

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

def process_RAW_comparison(frame, raw_file, srgb_folder, raw_folder, green, linear, bright_value):
    """
    Versión modificada que acepta un valor de 'bright' específico.
    """
    output_raw_path = os.path.join(raw_folder, str(frame) + '.png')
    output_ldr_path = os.path.join(srgb_folder, str(frame) + '.png')

    with rawpy.imread(raw_file) as raw:
        # 1. Guardar sRGB estándar (solo la primera vez o si no existe)
        if not os.path.isfile(output_ldr_path):
            rgb_srgb = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                gamma=(2.2, 0.0),
                no_auto_bright=True,
                output_bps=8,
            )
            cv2.imwrite(output_ldr_path, rgb_srgb)

        # 2. Guardar RAW con la exposición específica
        if not green and not os.path.isfile(output_raw_path):
            gamma_curve = (1.0, 0.0) if linear else (2.2, 0.0)

            raw_srgb = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                gamma=gamma_curve,
                no_auto_bright=True,
                output_bps=16,
                bright=bright_value, # <--- Valor variable
            )
            cv2.imwrite(output_raw_path, raw_srgb.astype(np.uint16))

def main():
    main_folder = r"datasets/rawslam/kitchen"
    green = False
    linear = True

    # --- LISTA DE EXPOSICIONES A PROBAR ---
    # Puedes probar valores como 1.0 (original), 2.0, 4.0, 8.0...
    exposures_to_test = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

    dng_folder = os.path.join(main_folder, "raw")
    srgb_folder = os.path.join(main_folder, "sRGB")
    os.makedirs(srgb_folder, exist_ok=True)

    with open(os.path.join(main_folder, "groundtruth.txt"), "r") as f:
        gt_lines = f.readlines()[1:4] # Solo 3 frames para el test rápido

    for exp in exposures_to_test:
        # Creamos una carpeta por cada exposición
        # Ejemplo: raw_linear_sRGB_bright_4.0
        folder_name = f"raw_linear_sRGB_bright_{exp}"
        raw_folder = os.path.join(main_folder, folder_name)
        os.makedirs(raw_folder, exist_ok=True)

        print(f"\n--- Procesando exposición: {exp} ---")
        for line in tqdm(gt_lines):
            parts = line.strip().split()
            if len(parts) != 8: continue
            frame = int(parts[0])
            raw_file = os.path.join(dng_folder, str(frame) + '.dng')

            process_RAW_comparison(frame, raw_file, srgb_folder, raw_folder, green, linear, exp)

    print("\n✅ ¡Comparativa lista! Revisa las carpetas en tu directorio de dataset.")

if __name__ == "__main__":
    main()