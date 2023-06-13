import pytorch_pfn_extras as ppe
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import datasets, transforms

from somen.pytorch_utility import TrainingConfig, train
from somen.pytorch_utility.metrics import ScikitLearnProbMetrics


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST("../data", train=True, download=True, transform=transform)
valid_set = datasets.MNIST("../data", train=False, download=True, transform=transform)

debug_mode = False
collate_fn = ppe.dataloaders.utils.CollateAsDict(names=["x", "target"])
macro_metrics = [ScikitLearnProbMetrics({"roc_auc_score": {"multi_class": "ovr"}})]
config = TrainingConfig(
    objective="nll",
    nb_epoch=10,
    device="cuda",
    progress_bar=True,
    macro_metrics=macro_metrics,
    debug_mode=debug_mode,
    lr_scheduler="cos",
    resume=False,
    distributed=False,
    num_workers=4,
    pin_memory=False,
)

model = train(config, CNN(), train_set, valid_set, "working/", collate_fn)
