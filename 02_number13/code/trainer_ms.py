import sys
import torch
from tqdm import tqdm as tqdm
import numpy as np
import torch.nn.functional as F
scaler = torch.cuda.amp.GradScaler()


class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y, l):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
                dataloader,
                desc=self.stage_name,
                file=sys.stdout,
                disable=not (self.verbose),
        ) as iterator:
            for x, yb, yr in iterator:
                x, yb, yr = x.to(self.device), yb.to(self.device), yr.to(self.device)

                loss, y_pred_b, y_pred_r = self.batch_update(x, yb, yr)
                #y = torch.stack([yb, yr], dim=1)
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred_b, yb).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", task='buildings',
                 verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.task = task

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, yb, yr):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            prediction_buildings, prediction_roads = self.model.forward(x)
            loss_b = self.loss(prediction_buildings, yb)
            loss_r = self.loss(prediction_roads, yr)
            loss = loss_b + loss_r
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        #prediction = torch.stack([yb, yr], dim=1)
        return loss, prediction_buildings, prediction_roads


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True, task='buildings'):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )
        self.task = task

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, yb, yr):
        with torch.no_grad():
            prediction_buildings, prediction_roads = self.model.forward(x)
            loss_b = self.loss(prediction_buildings, yb)
            loss_r = self.loss(prediction_roads, yr)
            loss = loss_b + loss_r
        #scaler.scale(loss)
        #prediction = torch.stack([yb, yr], dim=1)
        #prediction_buildings, prediction_roads = torch.sigmoid(prediction_buildings), torch.sigmoid(prediction_roads)
        return loss, prediction_buildings, prediction_roads


class EpochSiam:
    def __init__(self, model, loss, metrics, stage_name, device="cpu",
                 verbose=True, task='both'):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.task = task
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        if isinstance(self.loss, list):
            for item in self.loss:
                item.to(self.device)
        else:
            self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x_pre, x_post, flood, lab):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
                dataloader,
                desc=self.stage_name,
                file=sys.stdout,
                disable=not (self.verbose),
        ) as iterator:
            for x_pre, x_post, flood, lab in iterator:
                x_pre, x_post = x_pre.to(self.device), x_post.to(self.device)
                lab = lab.to(self.device)
                flood = flood.to(self.device)
                loss, y_pred = self.batch_update(x_pre, x_post, flood[:, [0], :, :],
                                                 flood[:, [1], :, :], lab)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                li = self.loss[0] if isinstance(self.loss, list) else self.loss
                loss_logs = {li.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, torch.stack([flood[:, [0], :, :],  flood[:, [1], :, :]], dim=1).squeeze()).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpochSiamese(EpochSiam):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", task='both',
                 verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x_pre, x_post, flood_b, flood_r, lab):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            seg_b, seg_r,  cls = self.model.forward(x_pre, x_post)
            seg_loss = self.loss[0](seg_b, flood_b) + self.loss[0](seg_r, flood_r)
            loss = 0.75 * seg_loss + 0.25 * self.loss[1](cls, lab)
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        return loss, torch.stack([seg_b, seg_r], dim=1).squeeze()


class ValidEpochSiamese(EpochSiam):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True, task='both'):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,

        )
        self.task = task

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x_pre, x_post, flood_b, flood_r, lab):
        with torch.no_grad(), torch.cuda.amp.autocast():
            seg_b, seg_r, cls = self.model.forward(x_pre, x_post)
            seg_loss = self.loss[0](seg_b, flood_b) + self.loss[0](seg_r, flood_r)
            loss = 0.75 * seg_loss + 0.25 * self.loss[1](cls, lab)
        scaler.scale(loss)
        return loss, torch.stack([seg_b, seg_r], dim=1).squeeze()
