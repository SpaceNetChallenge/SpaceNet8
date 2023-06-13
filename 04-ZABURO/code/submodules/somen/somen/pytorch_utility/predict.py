from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from pytorch_pfn_extras.runtime import runtime_registry
from tqdm import tqdm


def predict(
    model: Union[torch.nn.Module, Callable],
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 0,
    device: str = "cuda",
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
    progress: bool = True,
) -> Union[Mapping[str, np.ndarray], Sequence[np.ndarray]]:

    if isinstance(model, torch.nn.Module):
        model.to(device)
        model.eval()

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    if progress:
        data_loader = tqdm(data_loader)

    runtime = runtime_registry.get_runtime_class_for_device_spec(device)(device, {})

    with torch.no_grad():

        def _get_non_ref_array(x):
            return x.detach().cpu().numpy().copy()

        y_preds: Union[Dict[str, List[np.ndarray]], List[List[np.ndarray]], None] = None
        for batch in data_loader:
            batch = runtime.convert_batch(batch)

            if isinstance(batch, dict):
                y_pred = model(**batch)
            elif isinstance(batch, (list, tuple)):
                y_pred = model(*batch)
            else:
                y_pred = model(batch)

            if isinstance(y_pred, dict):
                y_pred = {key: _get_non_ref_array(value) for key, value in y_pred.items()}
            if isinstance(y_pred, (list, tuple)):
                y_pred = [_get_non_ref_array(e) for e in y_pred]
            else:
                y_pred = [_get_non_ref_array(y_pred)]

            if isinstance(y_pred, dict):
                if y_preds is None:
                    y_preds = {key: [] for key in y_pred}

                assert isinstance(y_preds, dict)
                for key, value in y_pred.items():
                    y_preds[key].append(value)
            else:
                assert isinstance(y_pred, list)
                if y_preds is None:
                    y_preds = [[] for _ in range(len(y_pred))]

                assert isinstance(y_preds, list)
                for k in range(len(y_pred)):
                    y_preds[k].append(y_pred[k])

        assert y_preds is not None
        if isinstance(y_preds, dict):
            return {key: np.concatenate(value, axis=0) for key, value in y_preds.items()}
        else:
            return [np.concatenate(value, axis=0) for value in y_preds]
