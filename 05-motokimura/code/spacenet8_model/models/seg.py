import segmentation_models_pytorch as smp
import torch

# isort: off
from spacenet8_model.utils.misc import get_flatten_classes
# isort: on


class SegmentationModel(torch.nn.Module):
    def __init__(self, config, **kwargs):

        super().__init__()

        self.backbone = smp.create_model(
            config.Model.arch,
            encoder_name=config.Model.encoder,
            in_channels=(1 + config.Model.n_input_post_images) * 3,
            classes=len(get_flatten_classes(config)),
            encoder_weights="imagenet",
            **kwargs)

        self.head = torch.nn.Identity()  # TODO: implement head if needed

    def forward(self, image):
        x = self.backbone(image)
        return self.head(x)
