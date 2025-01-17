from ct_segmentation.data_modules.segmentationimagefolder import SegmentationImageFolder
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class CTDataset(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        data_dir="/content/data/CT/png",
        train_dir="/train_dir",
        test_dir="/test_dir",
        num_classes=1,
        padding=True,
        image_small=True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.train_dataset_path = data_dir + train_dir
        self.test_dataset_path = data_dir + test_dir
        self.num_classes = num_classes
        if image_small:
            self.image_size = (256, 256)
        else:
            self.image_size = (512, 512)
        if padding:
            self.transform = transforms.Compose(
                [transforms.Pad((61, 61, 62, 62)), Resize(self.image_size), ToTensor()]
            )  # padding standardowych zdjęć 389x389 do 512x512
        else:
            self.transform_resize = transforms.Compose(
                [Resize(self.image_size), ToTensor()]
            )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = SegmentationImageFolder(
                self.train_dataset_path, transform=self.transform
            )
            train_dataset_size = int(len(dataset) * 0.8)
            self.train_dataset, self.val_dataset = random_split(
                dataset, [train_dataset_size, len(dataset) - train_dataset_size]
            )
        if stage == "test" or stage is None:
            self.test_dataset = SegmentationImageFolder(
                self.test_dataset_path, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
