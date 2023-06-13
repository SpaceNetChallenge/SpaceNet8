# https://github.com/konradhalas/dacite/issues/100
import typing

from typing_extensions import Literal

typing.Literal = Literal  # type: ignore
