from torch.utils.data import Dataset
from PIL import Image
import os


class SegmentationImageFolder(Dataset):
    def __init__(self, dataset_path, transform):
        self.image_dir = dataset_path + "/data"
        self.mask_dir = dataset_path + "/mask"
        self.transform = transform
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        return self.transform(image), self.transform(mask)
