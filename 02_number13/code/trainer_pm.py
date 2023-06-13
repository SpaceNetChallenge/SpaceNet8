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
            for x, y, l in iterator:
                x, y, l = x.to(self.device), y.to(self.device), l.to(self.device)
                loss, y_pred = self.batch_update(x, y, l)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
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
        self.device = device

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, l):
        self.optimizer.zero_grad(set_to_none=True)
        # We pass buildings images and road images and get corresponding output from the two heads
        prediction_buildings, prediction_roads = self.model.forward(x[l==0], x[l==1])
        # We backpropagate once for buildings
        loss_b = self.loss(prediction_buildings, y[l==0])
        loss_b.backward()
        # We backpropagate once again for roads
        loss_r = self.loss(prediction_roads, y[l==1])
        loss_r.backward()
        self.optimizer.step()
        loss = loss_b + loss_r
        prediction = torch.zeros(y.shape, dtype=prediction_buildings.dtype, device=self.device)
        prediction[l==0] = prediction_buildings
        prediction[l==1] = prediction_roads
        return loss, prediction


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
        self.device = device

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, l):
        with torch.no_grad():
            prediction_buildings, prediction_roads = self.model.forward(x[l==0], x[l==1])
            loss_b = self.loss(prediction_buildings, y[l==0])
            loss_r = self.loss(prediction_roads, y[l==1])
            loss = loss_b + loss_r
            prediction = torch.zeros(y.shape, dtype=prediction_buildings.dtype, device= self.device)
            prediction[l==0] = prediction_buildings
            prediction[l==1] = prediction_roads
        return loss, prediction