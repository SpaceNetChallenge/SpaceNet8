import torch
import torch.nn as nn

from .. import builder
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SiameseHead(BaseDecodeHead):

    def __init__(self, decode_head, **kwargs):
        super(SiameseHead, self).__init__(
            in_channels=-1, in_index=-1, input_transform=None, **kwargs)

        decode_head = builder.build_head(decode_head)
        if decode_head.dropout is not None:
            decode_head.dropout = nn.Identity()
        decode_head.conv_seg = nn.Identity()
        self.decode_head = decode_head

    def forward(self, inputs):
        """Placeholder of forward function."""
        x1, x2 = zip(*(torch.chunk(x, 2, dim=1) for x in inputs))
        output = torch.cat(
            (self.decode_head.forward(x1), self.decode_head.forward(x2)),
            dim=1)
        output = self.cls_seg(output)
        return output
