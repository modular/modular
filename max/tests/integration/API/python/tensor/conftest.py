# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from max.experimental.tensor import Tensor


def assert_all_close(
    t1: Tensor | np.ndarray | Sequence[int | float],
    t2: Tensor | np.ndarray,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    if not isinstance(t1, Tensor):
        t1 = Tensor.constant(t1, dtype=t2.dtype, device=t2.device)

    absolute_difference = abs(t1 - t2)
    # TODO: div0
    left_relative_difference = abs(absolute_difference / t1)
    right_relative_difference = abs(absolute_difference / t2)

    if (d := absolute_difference.max()) > atol:
        idx = absolute_difference.argmax().item()
        raise AssertionError(
            f"atol: tensors not close at index {idx}, {d.item()} > {atol}: \n"
            f"   left[{idx}] = {t1[idx].item()}\n"
            f"  right[{idx}] = {t2[idx].item()}\n"
        )
    elif (d := left_relative_difference.max()) > rtol:
        idx = left_relative_difference.argmax().item()
        raise AssertionError(
            f"rtol: tensors not close at index {idx}, {d.item()} > {rtol}: \n"
            f"   left[{idx}] = {t1[idx].item()}\n"
            f"  right[{idx}] = {t2[idx].item()}\n"
        )
    elif (d := right_relative_difference.max()) > rtol:
        idx = right_relative_difference.argmax().item()
        raise AssertionError(
            f"rtol: tensors not close at index {idx}, {d.item()} > {rtol}: \n"
            f"   left[{idx}] = {t1[idx].item()}\n"
            f"  right[{idx}] = {t2[idx].item()}\n"
        )
