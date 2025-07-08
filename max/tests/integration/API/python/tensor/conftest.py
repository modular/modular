# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from max.experimental.tensor import Tensor


def assert_all_close(t1, t2: Tensor, atol=1e-6, rtol=1e-6):
    if not isinstance(t1, Tensor):
        t1 = Tensor.constant(t1, dtype=t2.dtype, device=t2.device_)

    absolute_difference = abs(t1 - t2)
    # TODO: div0
    left_relative_difference = abs(absolute_difference / t1)
    right_relative_difference = abs(absolute_difference / t2)

    if (d := absolute_difference.max()) > atol:
        idx = absolute_difference.argmax()
        raise AssertionError(
            f"atol: tensors not close at index {idx.item()}, {d.item()} > {atol}: \n"
            f"   left[{idx.item()}] = {t1[idx].item()}\n"
            f"  right[{idx.item()}] = {t2[idx].item()}\n"
        )
    elif (d := left_relative_difference.max()) > rtol:
        idx = left_relative_difference.argmax()
        raise AssertionError(
            f"rtol: tensors not close at index {idx.item()}, {d.item()} > {rtol}: \n"
            f"   left[{idx.item()}] = {t1[idx].item()}\n"
            f"  right[{idx.item()}] = {t2[idx].item()}\n"
        )
    elif (d := right_relative_difference.max()) > rtol:
        idx = right_relative_difference.argmax()
        raise AssertionError(
            f"rtol: tensors not close at index {idx.item()}, {d.item()} > {rtol}: \n"
            f"   left[{idx.item()}] = {t1[idx].item()}\n"
            f"  right[{idx.item()}] = {t2[idx].item()}\n"
        )
