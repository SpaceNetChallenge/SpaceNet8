import segmentation_models_pytorch as smp
import torch
from torch.nn.modules.loss import _Loss


class Loss(_Loss):
    def __init__(self, config):
        super().__init__()

        self.loss_fns = []
        for group in config.Class.groups:
            self.loss_fns.append(
                _CombinedLoss(
                    group,
                    config.Class.classes[group],
                    config.Class.class_weights[group],
                    config.Class.losses[group],
                    config.Class.loss_weights[group]
                )
            )
        
        self.channel_indices = []
        first = 0
        for v in config.Class.classes.values():
            self.channel_indices.append(list(range(first, first + len(v))))
            first = first + len(v)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        loss = 0
        losses = {}
        for loss_fn, indices in zip(self.loss_fns, self.channel_indices):
            tmp_loss, tmp_losses = loss_fn(y_pred[:, indices], y_true[:, indices])
            loss += tmp_loss
            losses.update(tmp_losses)
        return loss, losses


class _CombinedLoss(_Loss):
    def __init__(
        self,
        group,
        classes,
        class_weights,
        losses,
        loss_weights
    ):
        assert len(classes) == len(class_weights), (classes, class_weights)
        assert len(losses) == len(loss_weights), (losses, loss_weights)

        super().__init__()

        self.loss_fns = []
        for loss in losses:
            if loss == 'dice':
                self.loss_fns.append(_DiceLoss(class_weights))
            elif loss == 'bce':
                self.loss_fns.append(_BCELoss(class_weights))
            elif loss == 'ce':
                self.loss_fns.append(
                    torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights)).cuda()  # TODO: cuda()?
                )
            else:
                raise ValueError(loss)

        self.group = group
        self.losses = losses
        self.loss_weights = loss_weights

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        loss = 0
        losses = {}
        for loss_fn, loss_weight, loss_name in zip(self.loss_fns, self.loss_weights, self.losses):
            tmp_loss = loss_weight * loss_fn(y_pred, y_true)
            loss += tmp_loss
            losses[f'{self.group}-{loss_name}'] = tmp_loss.detach().cpu()
        return loss, losses


class _BinaryLossBase(_Loss):
    def __init__(self):
        super().__init__()
        self.loss_fn = None
        self.class_weights = None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.loss_fn is None:
            raise NotImplementedError()
        if self.class_weights is None:
            raise NotImplementedError()

        assert y_pred.shape[1] == len(self.class_weights)
        assert y_true.shape[1] == len(self.class_weights)
        loss = 0
        for i, weight in enumerate(self.class_weights):
            loss += weight * self.loss_fn(y_pred[:, i], y_true[:, i])
        return loss


class _DiceLoss(_BinaryLossBase):
    def __init__(self, class_weights):
        super().__init__()
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        self.class_weights = class_weights


class _BCELoss(_BinaryLossBase):
    def __init__(self, class_weights):
        super().__init__()
        self.loss_fn = smp.losses.SoftBCEWithLogitsLoss()
        self.class_weights = class_weights
