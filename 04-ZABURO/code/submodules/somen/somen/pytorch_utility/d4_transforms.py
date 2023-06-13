from typing import Callable, Sequence

from torch import Tensor

D4: Sequence[Callable[[Tensor], Tensor]] = [
    lambda x: x,  # (x, y)
    lambda x: x.transpose(2, 3).flip(3),  # (y, -x)
    lambda x: x.flip(2).flip(3),  # (-x, -y)
    lambda x: x.transpose(2, 3).flip(2),  # (-y, x)
    lambda x: x.flip(3),  # (x, -y)
    lambda x: x.transpose(2, 3),  # (y, x)
    lambda x: x.flip(2),  # (-x, y)
    lambda x: x.transpose(2, 3).flip(2).flip(3),  # (-y, -x)
]
D4_inv = [D4[k] for k in [0, 3, 2, 1, 4, 5, 6, 7]]
