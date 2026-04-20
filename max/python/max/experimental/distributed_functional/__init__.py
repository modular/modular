# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Distributed functional API — PyTree-centric rule-based SPMD dispatch.

Explicit per-op SPMD dispatch via ``spmd_dispatch``.
Creation and random ops are standalone (no tensor inputs).

Usage::

    from max.experimental import functional as F

    y = F.matmul(a, b)
    z = F.add(x, y)
    w = F.transfer_to(z, new_mapping)

All logic lives in submodules:
    ``utils.py``          — shared helpers
    ``collective_ops.py`` — collectives + transfer_to
    ``spmd_ops.py``       — spmd_dispatch engine + explicit op functions
    ``creation_ops.py``   — full/ones/zeros/uniform/gaussian

Custom ops (``custom``, ``inplace_custom``) are defined here because
they combine graph ops with extension loading and don't fit a submodule.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from max._mlir_context import MLIRThreadPoolExecutor
from max.driver import Device
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Graph, TensorType, Type, ops
from max.graph.value import Value

# ── Collective ops ───────────────────────────────────────────────────────
from .collective_ops import (
    allgather,
    allreduce_sum,
    reduce_scatter,
    transfer_to,
)

# ── Creation + random ops ────────────────────────────────────────────────
from .creation_ops import (
    arange,
    constant,
    constant_external,
    full,
    full_like,
    gaussian,
    gaussian_like,
    hann_window,
    normal,
    normal_like,
    ones,
    ones_like,
    range,
    uniform,
    uniform_like,
    zeros,
    zeros_like,
)

# ── SPMD op engine + all registered ops ──────────────────────────────────
from .spmd_ops import (
    abs,
    acos,
    add,
    argmax,
    argmin,
    argsort,
    as_interleaved_complex,
    atanh,
    avg_pool2d,
    band_part,
    bottom_k,
    broadcast_to,
    buffer_store,
    buffer_store_slice,
    cast,
    chunk,
    clamp,
    complex_mul,
    concat,
    cond,
    conv2d,
    conv2d_transpose,
    conv3d,
    cos,
    cumsum,
    dequantize,
    div,
    elementwise_max,
    elementwise_min,
    equal,
    erf,
    exp,
    flatten,
    floor,
    fold,
    functional,
    gather,
    gather_nd,
    gelu,
    greater,
    greater_equal,
    group_norm,
    irfft,
    is_inf,
    is_nan,
    layer_norm,
    log,
    log1p,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    logsoftmax,
    masked_scatter,
    matmul,
    max,
    max_pool2d,
    mean,
    min,
    mod,
    mul,
    negate,
    non_maximum_suppression,
    nonzero,
    not_equal,
    outer,
    pad,
    permute,
    pow,
    prod,
    qmatmul,
    relu,
    repeat_interleave,
    reshape,
    resize,
    resize_bicubic,
    resize_linear,
    resize_nearest,
    rms_norm,
    roi_align,
    round,
    rsqrt,
    scatter_add,
    scatter_max,
    scatter_min,
    scatter_mul,
    scatter_nd,
    scatter_nd_add,
    scatter_nd_max,
    scatter_nd_min,
    scatter_nd_mul,
    sigmoid,
    silu,
    sin,
    slice_tensor,
    softmax,
    split,
    spmd_dispatch,
    sqrt,
    squeeze,
    stack,
    sub,
    sum,
    tanh,
    tile,
    top_k,
    transpose,
    trunc,
    unsqueeze,
    where,
    while_loop,
)

# Also re-export scatter from spmd_ops (the per-element scatter op,
# NOT the collective scatter).  The spmd_ops version shadows the
# collective one in this namespace — matching prior behavior.
from .spmd_ops import scatter as scatter

# ── Utils ────────────────────────────────────────────────────────────────
from .utils import (
    ensure_context,
    in_graph_context,
    lazy,
    tensor_to_layout,
)

_Result = TypeVar("_Result")


# ── Async helpers (used by tensor._sync_realize, realization_context) ────


def _in_running_loop() -> bool:
    """Check whether the caller is inside a running asyncio event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _run(coro: Coroutine[Any, Any, _Result]) -> _Result:
    """Run a coroutine synchronously, handling nested event loops.

    If not inside an event loop, uses ``asyncio.run()``. If already inside
    an event loop (e.g., in Jupyter), runs the coroutine in a separate
    thread to avoid blocking.
    """
    if not _in_running_loop():
        return asyncio.run(coro)

    loop = asyncio.new_event_loop()
    with MLIRThreadPoolExecutor() as pool:
        fut = pool.submit(loop.run_until_complete, coro)
    return fut.result()


# ── Custom ops ──────────────────────────────────────────────────────────


def _load_custom_extensions(
    custom_extensions: str | Path | Sequence[str | Path] | None,
) -> None:
    """Load custom extensions into the current graph if not already loaded."""
    if custom_extensions is None:
        return
    graph = Graph.current
    if isinstance(custom_extensions, (str, Path)):
        custom_extensions = [custom_extensions]
    paths = [Path(p) for p in custom_extensions]
    graph._import_kernels(paths)


def custom(
    name: str,
    device: Device | DeviceRef,
    values: Sequence[Value[Any]],
    out_types: Sequence[Type[Any]],
    parameters: Mapping[str, bool | int | str | DType] | None = None,
    custom_extensions: str | Path | Sequence[str | Path] | None = None,
) -> list[Tensor]:
    """Apply a custom op with optional custom extension loading."""
    with ensure_context():
        _load_custom_extensions(custom_extensions)
        return [
            Tensor.from_graph_value(v)
            for v in ops.custom(
                name=name,
                device=device,
                values=values,
                out_types=out_types,
                parameters=parameters,
            )
        ]


def inplace_custom(
    name: str,
    device: Device | DeviceRef,
    values: Sequence[Value[Any]],
    out_types: Sequence[Type[Any]] | None = None,
    parameters: dict[str, bool | int | str | DType] | None = None,
    custom_extensions: str | Path | Sequence[str | Path] | None = None,
) -> list[Tensor]:
    """Apply an in-place custom op with optional custom extension loading."""
    with ensure_context():
        _load_custom_extensions(custom_extensions)
        result = ops.inplace_custom(
            name=name,
            device=device,
            values=values,
            out_types=out_types,
            parameters=parameters,
        )
        return [Tensor.from_graph_value(v) for v in result] if result else []
