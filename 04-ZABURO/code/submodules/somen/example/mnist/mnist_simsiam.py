from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from lightly.loss import NegativeCosineSimilarity
from lightly.models import ResNetGenerator
from lightly.models.modules import heads
from lightly.utils.benchmarking import knn_predict
from pytorch_pfn_extras.handler import Logic, torch_autocast
from torch import Tensor, nn
from torchvision import datasets, transforms

from somen.pytorch_utility import TrainingConfig, train


@torch.no_grad()
def patch_first_conv(model: nn.Module) -> None:
    # get first conv
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = 1
    module.weight = torch.nn.parameter.Parameter(module.weight.detach().sum(dim=1, keepdim=True))


class SimSiam(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        patch_first_conv(resnet)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1))
        self.prediction_head = heads.SimSiamPredictionHead(2048, 512, 2048)
        # use a 2-layer projection head for cifar10 as described in the paper
        self.projection_head = heads.ProjectionHead(
            [(512, 2048, nn.BatchNorm1d(2048), nn.ReLU(inplace=True)), (2048, 2048, nn.BatchNorm1d(2048), None)]
        )

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p


class MNISTSimSiamDataset(torch.utils.data.Dataset):
    def __init__(self, train: bool, transform: transforms.Compose) -> None:
        super().__init__()
        self.base_dataset = datasets.MNIST("../data", train=train, download=True, transform=transform)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        x0 = self.base_dataset[index][0]
        x1 = self.base_dataset[index][0]
        return {"x0": x0, "x1": x1}


class SimSiamLogic(Logic):
    def __init__(self, knn_train_loader: torch.utils.data.DataLoader, num_classes: int) -> None:
        super().__init__()
        self.criterion = NegativeCosineSimilarity()
        self.knn_train_loader = knn_train_loader
        self.feature_bank: Optional[Tensor] = None
        self.targets_bank: Optional[Tensor] = None
        self.num_classes = num_classes
        self.knn_k = 200
        self.knn_t = 0.1

    def train_step(
        self,
        models: Mapping[str, torch.nn.Module],
        optimizers: Mapping[str, torch.optim.Optimizer],
        batch_idx: int,
        batch: Any,
    ) -> Any:
        with torch_autocast(enabled=self._autocast):
            optimizers[self.model_name].zero_grad()

            model = models[self.model_name]
            assert isinstance(model, SimSiam)

            x0, x1 = batch["x0"], batch["x1"]
            z0, p0 = model(x0)
            z1, p1 = model(x1)

            loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))

            with torch.no_grad():
                output = p0.detach()
                output = torch.nn.functional.normalize(output, dim=1)

                output_std = torch.std(output, 0)
                output_std = output_std.mean()

                collapse_level = max(0.0, 1 - np.sqrt(output.shape[1]) * output_std)
        self._backward({"loss": loss})
        return {"loss": loss, "collapse_level": collapse_level}

    def train_validation_begin(self, models: Mapping[str, torch.nn.Module]) -> None:
        model = models[self.model_name]
        assert isinstance(model, SimSiam)
        model.eval()

        feature_bank, targets_bank = [], []
        with torch.no_grad(), torch_autocast(enabled=self._autocast):
            for x, target in self.knn_train_loader:
                # TODO: runtime を取ってくる方法を考える
                x = x.to("cuda")
                target = target.to("cuda")
                feature = model.backbone(x).squeeze()
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
                targets_bank.append(target)

        self.feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        self.targets_bank = torch.cat(targets_bank, dim=0).t().contiguous()
        return super().train_validation_begin(models)

    def eval_step(self, models: Mapping[str, torch.nn.Module], batch_idx: int, batch: Any) -> Any:
        if self.feature_bank is not None and self.targets_bank is not None:
            model = models[self.model_name]
            assert isinstance(model, SimSiam)

            x, target = batch
            feature = model.backbone(x).squeeze()
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(
                feature, self.feature_bank, self.targets_bank, self.num_classes, self.knn_k, self.knn_t
            )
            top1_accuracy = (pred_labels[:, 0] == target).float().mean()
            return {"top1_acc": top1_accuracy}
        else:
            return {}


if __name__ == "__main__":
    batch_size = 32
    num_workers = 4
    lr_factor = batch_size / 128

    transform = transforms.Compose(
        [
            transforms.RandomAffine(degrees=(-20, 20), translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_set = MNISTSimSiamDataset(True, transform)

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    knn_train_set = datasets.MNIST("../data", train=True, download=True, transform=test_transform)
    knn_train_loader = torch.utils.data.DataLoader(
        knn_train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    knn_test_set = datasets.MNIST("../data", train=False, download=True, transform=test_transform)

    model = SimSiam()

    config = TrainingConfig(
        optimizer="SGD",
        optimizer_params={"lr": 6e-2 * lr_factor, "momentum": 0.9, "weight_decay": 5e-4},
        lr_scheduler="cos",
        logic=SimSiamLogic(knn_train_loader=knn_train_loader, num_classes=10),
        objective=None,
        nb_epoch=20,
        batch_size=batch_size,
        device="cuda",
        progress_bar=True,
        resume=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    model = train(config, model, train_set, knn_test_set, "working_32_6e-2_20/")
