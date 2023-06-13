import torch

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class ChangeDetector(EncoderDecoder):
    """Segmentors for change detectection.

    ChangeDetector typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def extract_feat(self, img):
        """Extract features from images."""
        pair = []
        for x in torch.chunk(img, 2, dim=1):
            x = super(ChangeDetector, self).extract_feat(x)
            pair.append(x)
        return [torch.cat((x1, x2), dim=1) for x1, x2 in zip(*pair)]
