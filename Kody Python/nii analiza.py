import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

file_path = "/Data/CT/nii_data/segmentation/case_0000.nii"
nii_data = nib.load(file_path)
mask_data = nii_data.get_fdata()
unique_classes = np.unique(mask_data).astype(int)
print("Unikalne wartości w masce segmentacyjnej (klasy):")
print(unique_classes)
custom_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#393b79",
    "#5254a3",
    "#6b6ecf",
    "#9c9ede",
]
cmap = ListedColormap(custom_colors[: len(unique_classes)])
class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
indexed_data = np.vectorize(class_to_index.get)(mask_data)
slice_index = mask_data.shape[0] // 2
plt.imshow(indexed_data[slice_index, :, :], cmap=cmap)
plt.title("Środkowy wycinek maski segmentacyjnej")
plt.colorbar(ticks=range(len(unique_classes)), label="Klasy")
plt.clim(-0.5, len(unique_classes) - 0.5)
plt.show()
