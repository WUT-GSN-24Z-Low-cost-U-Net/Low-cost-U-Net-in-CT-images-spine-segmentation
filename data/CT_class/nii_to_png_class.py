import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def nii_to_middle_slice_png(input_file, output_dir):
    try:
        nii_image = nib.load(input_file)
        data = nii_image.get_fdata()
        middle_index = data.shape[0] // 2
        middle_slice = data[middle_index, :, :]
        colors = [
            "#000000",
            "#00FF00",
            "#0000FF",
            "#FFFF00",
            "#FF00FF",
            "#00FFFF",
            "#800000",
            "#808000",
            "#008000",
            "#800080",
            "#008080",
            "#000080",
            "#FFA500",
            "#A52A2A",
            "#5F9EA0",
            "#7FFF00",
            "#D2691E",
            "#FF7F50",
            "#6495ED",
            "#DC143C",
            "#00FA9A",
            "#8A2BE2",
            "#A9A9A9",
            "#2F4F4F",
        ]
        cmap = ListedColormap(colors)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir, os.path.basename(input_file).replace(".nii", ".png")
        )
        plt.figure()
        plt.axis("off")
        plt.imshow(middle_slice, cmap=cmap, interpolation="nearest")
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"Saved: {output_file}")

    except Exception as e:
        print(f"Error processing {input_file}: {e}")


def process_all_nii_files(input_dir, output_dir):
    nii_files = [f for f in os.listdir(input_dir) if f.endswith(".nii")]
    if not nii_files:
        print("No .nii files found in the input directory.")
        return
    for nii_file in nii_files:
        input_file_path = os.path.join(input_dir, nii_file)
        nii_to_middle_slice_png(input_file_path, output_dir)


input_directory = "/Data/CT/nii_data/segmentation"
output_directory = "/Data/CT_class/data_ct_class/train_dir/mask"
process_all_nii_files(input_directory, output_directory)
