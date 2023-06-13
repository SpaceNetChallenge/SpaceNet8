from pytorch_pfn_extras.training.metrics import Batch as DictBatch
from typing_extensions import Protocol


class Metric(Protocol):
    def __cal__(self, ins: DictBatch, outs: DictBatch) -> DictBatch:
        ...
