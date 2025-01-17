import torch
from torch import nn, optim
import pytorch_lightning as pl
from torcheval.metrics.functional import binary_f1_score
from rich.console import Console
from ct_segmentation.u_net_model.unetWithAttention import UNetWithAttention

console = Console()


class UNetModel(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        learning_rate=1e-3,
        scheduler_step_size=8,
        scheduler_gamma=0.5,
        pos_weight=2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.out_channels = out_channels
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.pos_weight = pos_weight

        self.model = UNetWithAttention(in_channels, out_channels)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y):
        pos_weight = torch.tensor([self.pos_weight]).cuda()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return criterion(x, y)

    def common_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.compute_loss(outputs, y)
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        acc = binary_f1_score(outputs.view(-1), y.view(-1), threshold=0.5)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        self.log(
            "train_acc", acc, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        console.print(
            f"Validation [bold cyan]Loss (batch {batch_idx}): {loss:.4f}[bold cyan]"
        )
        console.print(
            f"Validation [bold green]Accuracy (batch {batch_idx}): {acc:.4f}[/bold green]"
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        console.print(
            f"Test [bold cyan]Loss (batch {batch_idx}): {loss:.4f}[bold cyan]"
        )
        console.print(
            f"Test [bold green]Accuracy (batch {batch_idx}): {acc:.4f}[/bold green]"
        )
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
        )
        return [optimizer], [lr_scheduler]
