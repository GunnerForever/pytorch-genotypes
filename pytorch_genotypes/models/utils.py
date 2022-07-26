import torch.nn as nn
from typing import Tuple, Optional, Iterable, List


def build_mlp(
    _in: int,
    hidden: Tuple[int],
    out: Optional[int] = None,
    add_batchnorm: bool = False,
    activations: Iterable[nn.Module] = [nn.LeakyReLU()]
):
    layers: List[nn.Module] = []
    h_prev = _in
    for h in hidden:
        layers.extend([
            nn.Linear(in_features=h_prev, out_features=h),
        ])
        layers.extend(activations)
        if add_batchnorm:
            layers.append(nn.BatchNorm1d(h))

        h_prev = h

    if out is not None:
        layers.append(nn.Linear(in_features=h_prev, out_features=out))

    return layers
