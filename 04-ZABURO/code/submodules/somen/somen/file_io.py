import contextlib
import dataclasses
import hashlib
import json
import os
import pickle
from distutils.util import strtobool
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import dacite
import feather
import numpy as np
import pandas as pd
import tables as tb
import yaml

from somen.types import PathLike

hash_dict = None


def _md5(fname: PathLike) -> str:
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@contextlib.contextmanager
def recording_hash(use_dict):
    global hash_dict
    old_hash_dict = hash_dict
    try:
        hash_dict = use_dict
        yield
    finally:
        hash_dict = old_hash_dict


def recorded(filepath: PathLike) -> PathLike:
    global hash_dict

    if hash_dict is None:
        raise ValueError("`recorded` can be used in `recording_hash` context")

    hash_dict[str(filepath)] = _md5(filepath)
    return filepath


def _maybe_record_hash(filepath: PathLike) -> None:
    if hash_dict is not None:
        recorded(filepath)


def _check_path_exist(filepath: PathLike) -> None:
    if not os.path.exists(filepath):
        raise IOError("{} not found".format(filepath))


def _check_dir(filepath: PathLike) -> None:
    dir_path = Path(filepath).parent
    if not dir_path.exists():
        dir_path.mkdir(parents=True)


def save_array(X: np.ndarray, filepath: PathLike) -> None:
    _check_dir(filepath)
    X = np.asarray(X)
    with tb.open_file(str(filepath), "w") as f:
        atom = tb.Atom.from_dtype(X.dtype)
        filters = tb.Filters(complib="blosc", complevel=9)
        ds = f.create_carray(f.root, "X", atom, X.shape, filters=filters)
        ds[:] = X


def load_array(filepath: PathLike) -> np.ndarray:
    _maybe_record_hash(filepath)
    with tb.open_file(str(filepath), "r") as f:
        return f.root.X.read()


def save_npy(X: np.ndarray, filepath: PathLike) -> None:
    _check_dir(filepath)
    X = np.asarray(X)
    np.save(str(filepath), X)


def load_npy(filepath: PathLike) -> np.ndarray:
    _maybe_record_hash(filepath)
    return np.load(str(filepath))


def save_npz(datasets: Union[np.ndarray, Sequence[np.ndarray]], filepath: PathLike) -> None:
    _check_dir(filepath)
    if isinstance(datasets, np.ndarray):
        datasets = (datasets,)
    np.savez_compressed(filepath, *datasets)


def load_npz(filepath: PathLike, allow_pickle: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
    _maybe_record_hash(filepath)
    load_data = np.load(filepath, allow_pickle=allow_pickle)
    result = []
    i = 0
    while True:
        key = "arr_{}".format(i)
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    if len(result) == 1:
        result = result[0]
    return result


def save_df(X: pd.DataFrame, filepath: PathLike) -> None:
    _check_dir(filepath)
    pd.DataFrame(X).to_feather(str(filepath))


def load_df(filepath: PathLike) -> pd.DataFrame:
    _maybe_record_hash(filepath)
    try:
        df = pd.read_feather(str(filepath))
    except Exception:
        df = feather.read_dataframe(str(filepath))
    return df


def save_df_as_hdf5(X: pd.DataFrame, filepath: PathLike) -> None:
    _check_dir(filepath)
    pd.DataFrame(X).to_hdf(str(filepath), complevel=9, complib="blosc", key="table")


def load_df_as_hdf5(filepath: PathLike) -> pd.DataFrame:
    _maybe_record_hash(filepath)
    return pd.read_hdf(str(filepath), key="table")


def save_series(X: pd.Series, filepath: PathLike) -> None:
    _check_dir(filepath)
    pd.Series(X).to_hdf(str(filepath), complevel=9, complib="blosc", key="table")


def load_series(filepath: PathLike) -> pd.Series:
    _maybe_record_hash(filepath)
    return pd.read_hdf(str(filepath), key="table")


def save_pickle(X: Any, filepath: PathLike) -> None:
    _check_dir(filepath)
    with open(str(filepath), "wb") as fp:
        pickle.dump(X, fp, pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath: PathLike) -> Any:
    _maybe_record_hash(filepath)
    with open(str(filepath), "rb") as fp:
        ret = pickle.load(fp)
    return ret


def save_json(X: dict, filepath: PathLike, *args, **kwargs) -> None:  # type: ignore
    _check_dir(filepath)
    # TODO: make X jsonable
    with open(str(filepath), "w") as fp:
        json.dump(X, fp, *args, **kwargs)


def load_json(filepath: PathLike) -> dict:
    _maybe_record_hash(filepath)
    with open(str(filepath)) as fp:
        return json.load(fp)


T = TypeVar("T", bound=Any)
DACITE_INV_TYPE_HOOKS = {Path: lambda x: str(x)}


def _apply_inv_type_hooks(data: Any) -> Any:
    for type_, hook in DACITE_INV_TYPE_HOOKS.items():
        if isinstance(data, type_):
            return hook(data)
    if isinstance(data, dict):
        return {key: _apply_inv_type_hooks(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_apply_inv_type_hooks(value) for value in data]
    if isinstance(data, tuple):
        return tuple(_apply_inv_type_hooks(value) for value in data)
    return data


def save_yaml_from_dataclass(X: Any, filepath: PathLike) -> None:
    _check_dir(filepath)
    with open(str(filepath), "w") as fp:
        yaml.safe_dump(_apply_inv_type_hooks(dataclasses.asdict(X)), fp)


def _as_annotated_type(value: str, annotated_type: Type[T]) -> T:
    if hasattr(annotated_type, "__origin__") and annotated_type.__origin__ is Union:  # type: ignore
        for t in annotated_type.__args__:  # type: ignore
            if isinstance(None, t):  # check if t is NoneType
                continue
            try:
                return _as_annotated_type(value, t)
            except ValueError:
                pass
        raise ValueError(f"`{value}` could not be interpreted as `{annotated_type}`")
    if annotated_type is bool:
        return bool(strtobool(value))  # type: ignore
    else:
        return annotated_type(value)


def load_yaml_as_dataclass(
    target_class: Type[T], filepath: Optional[PathLike], overrides: Optional[Sequence[str]] = None
) -> T:
    if filepath is not None:
        _maybe_record_hash(filepath)
        with open(str(filepath), "r") as fp:
            data = yaml.safe_load(fp)
        result = dacite.from_dict(
            data_class=target_class,
            data=data,
            config=dacite.Config(
                cast=[Enum, Path, Tuple],  # type: ignore
            ),
        )
    else:
        result = target_class()

    if overrides is not None:
        for keys_value in overrides:
            cat_keys, value = keys_value.split("=")
            lhs = result
            keys = cat_keys.split(".")
            for key in keys[:-1]:
                lhs = getattr(lhs, key)
                assert lhs is not None
            key = keys[-1]

            annotated_type: Optional[Type] = None
            if hasattr(type(lhs), "__annotations__"):
                annotations = type(lhs).__annotations__
                if key in annotations:
                    annotated_type = annotations[key]

            if annotated_type is None:
                annotated_type = type(getattr(lhs, key))

            setattr(lhs, key, _as_annotated_type(value, annotated_type))

    return result
