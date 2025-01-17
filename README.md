# Low-cost-U-Net-in-CT-images-spine-segmentation
This project was created for the GSN course during the 24Z semester at the Warsaw University of Technology.
The project aims to create a low-cost U-Net with quantization, pruning, and attention mechanisms for the purpose of CT image spine segmentation.

## Creating dataset
1. Download this repository.
2. Download the dataset from [Kaggle: Spine Segmentation from CT Scans](https://www.kaggle.com/datasets/pycadmk/spine-segmentation-from-ct-scans).
3. Put files in `data/CT/nii_data/data` for volumes and `data/CT/nii_data/mask` for segmentation.
4. (Optional) modify parameters in `data/CT/nii_to_png.py`.
5. Run `python data/CT/nii_to_png.py`.
6. Data will be in `data` directory.
7. To use it in training put it in `/content` while while running one of the notebooks and skip the line downloading different dataset.
## Hydra (only training model)
1. Download this repository.
2. (Optional) modify parameters in files in `config` directory.
3. Put Hydra.ipynb on google colab.
4. Run all cells.
5. When prompted about Weights & Biases, enter a key in the console.
## Project notebook
1. Download this repository.
2. Put Projekt_GSN.ipynb on google colab.
3. Run all cells.
4. When prompted about Weights & Biases, enter a key in the console.