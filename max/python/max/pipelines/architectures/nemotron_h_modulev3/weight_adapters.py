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
"""Weight loading for Nemotron-H checkpoints.

The module tree uses the checkpoint's names, so tensors pass through as
stored, quantized ones included. :func:`stack_nvfp4_experts` and
:func:`stack_bf16_experts` stack each mixer's routed experts for the grouped
matmul, and :func:`prepare_nvfp4_linears` lays out the dense quantized
modules' scales. Under tensor parallelism, both stackers stack each expert's
channels as one zero-padded block per device,
:func:`permute_mamba_for_tp` regroups the Mamba mixers' fused rows by device
and :func:`repeat_kv_heads_for_tp` gives each device a whole KV head.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.graph.weights.weights import Shape
from max.support.math import ceildiv

from .model_config import LayerKind, NemotronHConfig, moe_channels_per_device
from .quantization import (
    NVFP4_GROUP_SIZE,
    ModuleFormat,
    Parallelism,
    linear_parallelism,
)


def convert_nemotron_h_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs: object
) -> dict[str, WeightData]:
    """Returns the tensors Nemotron-H reads, under their own names.

    Upcasts the router weight to float32; the router matmul runs in float32.
    Stores each depthwise conv weight ``[dim, 1, kernel]`` as the
    ``[dim, kernel]`` the conv kernel reads.
    """
    weights: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        # The MTP head is not built, and the FP8 KV-cache scales are not
        # applied.
        if name.startswith("mtp.") or name.endswith(
            (".k_proj.k_scale", ".v_proj.v_scale")
        ):
            continue
        data = value.data()
        if name.endswith(".mixer.gate.weight") and data.dtype != DType.float32:
            data = data.astype(DType.float32)
        elif name.endswith(".mixer.conv1d.weight"):
            dim, _, kernel = (int(d) for d in data.shape)
            data = WeightData(
                data=data.to_buffer().view(data.dtype, (dim, kernel)),
                name=name,
                dtype=data.dtype,
                shape=Shape([dim, kernel]),
            )
        weights[name] = data
    return weights


def _bytes(weight: WeightData) -> npt.NDArray[np.uint8]:
    return np.from_dlpack(weight.to_buffer().view(DType.uint8)).view(np.uint8)


def _scalar(weight: WeightData) -> np.float32:
    return np.from_dlpack(weight.to_buffer()).astype(np.float32).reshape(-1)[0]


def _pool() -> ThreadPoolExecutor:
    # numpy releases the GIL inside these array operations, so the tasks run
    # in parallel on threads.
    return ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1))


def _weight(
    name: str,
    array: npt.NDArray[np.generic],
    dtype: DType = DType.bfloat16,
    shape: tuple[int, ...] | None = None,
) -> WeightData:
    """Wraps host bytes as a weight, since numpy has no BF16 or FP8 dtype."""
    shape = array.shape if shape is None else shape
    return WeightData(
        data=Buffer.from_numpy(array).view(dtype, shape),
        name=name,
        dtype=dtype,
        shape=Shape(shape),
    )


# The SM100 block-scaled matmul reads its scales in granules of 128 rows by 4
# scale columns.
_SF_GRANULE_ROWS = 128
_SF_ATOM_ROWS = 32
_SF_ATOM_COLS = 4


def interleave_nvfp4_scales(
    scales: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    """Permutes one expert's E4M3 block scales into the kernel layout.

    Element ``(r, c)`` lands at ``[r // 128, c // 4, r % 32, (r % 128) // 32,
    c % 4]``, where ``set_scale_factor`` in
    ``max/kernels/src/linalg/fp4_utils.mojo`` stores it. Rows are zero-padded
    to a whole granule.

    Args:
        scales: The ``[N, K / 16]`` scales as bytes. ``K / 16`` must be a
            multiple of 4.

    Returns:
        The ``[ceil(N / 128), K / 64, 32, 4, 4]`` scales.
    """
    rows, cols = scales.shape
    if cols % _SF_ATOM_COLS:
        raise ValueError(f"cannot interleave NVFP4 scales {scales.shape}")
    granules = ceildiv(rows, _SF_GRANULE_ROWS)
    padded = granules * _SF_GRANULE_ROWS
    if padded != rows:
        scales = np.pad(scales, ((0, padded - rows), (0, 0)))
    atoms = scales.reshape(
        granules,
        _SF_GRANULE_ROWS // _SF_ATOM_ROWS,
        _SF_ATOM_ROWS,
        cols // _SF_ATOM_COLS,
        _SF_ATOM_COLS,
    )
    return np.ascontiguousarray(atoms.transpose(0, 3, 2, 1, 4))


def _stack_device_blocks(
    array: npt.NDArray[np.uint8],
    axis: int,
    n: int,
    size: int,
    each: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]] = (
        lambda block: block
    ),
) -> npt.NDArray[np.uint8]:
    """Splits ``axis`` into ``n`` equal blocks and stacks them back padded.

    Each block is zero-padded to ``size`` along ``axis`` and passed through
    ``each`` before the blocks concatenate along ``axis``.
    """
    if n == 1:
        return each(array)
    pad = [(0, 0)] * array.ndim
    pad[axis] = (0, size - array.shape[axis] // n)
    return np.concatenate(
        [each(np.pad(block, pad)) for block in np.split(array, n, axis=axis)],
        axis=axis,
    )


def _channel_axis(proj: str) -> int:
    """Returns the weight axis of a routed projection's expert channels."""
    return 0 if proj == "up_proj" else 1


def stack_nvfp4_experts(
    state_dict: Mapping[str, WeightData],
    num_experts: int,
    shared_slices: Mapping[str, int],
    num_devices: int = 1,
) -> dict[str, WeightData]:
    """Stacks the NVFP4 routed experts of each mixer for the W4A4 matmul.

    Each mixer's per-expert ``up_proj`` and ``down_proj`` tensors become
    ``{mixer}.up_weight`` (packed E2M1, ``[experts, N, K / 2]``),
    ``{mixer}.up_block_scale`` (E4M3, ``[experts, N, K / 16]``, which the
    graph interleaves at init), ``{mixer}.up_scale`` (float32
    ``weight_scale_2``, ``[experts]``) and the three ``down_`` tensors.

    A mixer's shared expert splits into slices after its routed experts,
    ``up_proj`` by output rows and ``down_proj`` by input columns, each slice
    with the shared expert's ``weight_scale_2`` (see
    :meth:`NemotronHConfig.shared_expert_slices`).

    Under tensor parallelism the expert channels, ``N`` of ``up`` and ``K``
    of ``down``, stack one block per device. Each block holds the device's
    share of the channels zero-padded to
    :func:`~.model_config.moe_channels_per_device`, so an even split of the
    stack gives each device its share of every expert. Zero codes and zero
    scales add nothing to the matmul.

    Args:
        state_dict: The checkpoint's tensors.
        num_experts: The number of routed experts per mixer.
        shared_slices: The MoE mixers to stack (see
            :meth:`NemotronHConfig.w4a4_mixers`), each with the number of
            slices its shared expert runs as, or 0.
        num_devices: The number of devices the expert channels split across.
            Each device's share must be a whole number of NVFP4 blocks.

    Returns:
        A new state dict.
    """
    weights = dict(state_dict)

    def pop(
        module: str,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], np.float32]:
        codes = _bytes(weights.pop(f"{module}.weight"))
        scales = _bytes(weights.pop(f"{module}.weight_scale"))
        global_scale = _scalar(weights.pop(f"{module}.weight_scale_2"))
        return codes, scales, global_scale

    def device_blocks(
        codes: npt.NDArray[np.uint8],
        scales: npt.NDArray[np.uint8],
        global_scale: np.float32,
        axis: int,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], np.float32]:
        # The packed codes hold two channels per byte along K.
        channels = codes.shape[axis] * (2 if axis else 1)
        size = moe_channels_per_device(channels, num_devices)
        codes = _stack_device_blocks(
            codes, axis, num_devices, size // 2 if axis else size
        )
        scales = _stack_device_blocks(
            scales,
            axis,
            num_devices,
            size // NVFP4_GROUP_SIZE if axis else size,
        )
        return codes, scales, global_scale

    def load(
        module: str, axis: int
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], np.float32]:
        return device_blocks(*pop(module), axis)

    def load_slices(
        module: str, slices: int, axis: int
    ) -> list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], np.float32]]:
        codes, scales, global_scale = pop(module)
        return [
            device_blocks(
                np.ascontiguousarray(c),
                np.ascontiguousarray(sc),
                global_scale,
                axis,
            )
            for c, sc in zip(
                np.split(codes, slices, axis=axis),
                np.split(scales, slices, axis=axis),
                strict=True,
            )
        ]

    with _pool() as pool:
        for mixer, slices in sorted(shared_slices.items()):
            for proj in ("up_proj", "down_proj"):
                axis = _channel_axis(proj)
                loaded = list(
                    pool.map(
                        functools.partial(load, axis=axis),
                        (
                            f"{mixer}.experts.{e}.{proj}"
                            for e in range(num_experts)
                        ),
                    )
                )
                if slices:
                    loaded += load_slices(
                        f"{mixer}.shared_experts.{proj}", slices, axis
                    )
                prefix = f"{mixer}.{proj.removesuffix('_proj')}"
                for suffix, array, dtype in (
                    (
                        "weight",
                        np.stack([c for c, _, _ in loaded]),
                        DType.uint8,
                    ),
                    (
                        "block_scale",
                        np.stack([s for _, s, _ in loaded]),
                        DType.float8_e4m3fn,
                    ),
                    (
                        "scale",
                        np.array([g for _, _, g in loaded], dtype=np.float32),
                        DType.float32,
                    ),
                ):
                    name = f"{prefix}_{suffix}"
                    weights[name] = _weight(name, array, dtype)
    return weights


def _device_slots(
    stack: npt.NDArray[np.uint16], axis: int, n: int, channels: int
) -> list[npt.NDArray[np.uint16]]:
    """Returns the views of one expert's padded slot its channels fill.

    The expert's rows split evenly between the views, in order, and each
    row fills one row of its view.

    Args:
        stack: The expert's ``[N, K]`` slot, holding ``n`` padded blocks of
            channels along ``axis``.
        axis: The axis of the expert channels.
        n: The number of devices.
        channels: The expert's channels without padding.
    """
    rows, cols = stack.shape
    if axis == 0:
        # Each device's rows are one contiguous run of the expert's rows.
        return list(stack.reshape(n, rows // n, cols)[:, : channels // n])
    # Every row of the expert fills a run of each device's block.
    return [stack.reshape(rows, n, cols // n)[:, :, : channels // n]]


def stack_bf16_experts(
    state_dict: Mapping[str, WeightData],
    num_experts: int,
    mixers: Collection[str],
    num_devices: int = 1,
) -> dict[str, WeightData]:
    """Stacks the BF16 routed experts of ``mixers`` for the grouped matmul.

    Each mixer's per-expert ``up_proj`` and ``down_proj`` weights become
    ``{mixer}.up_weight`` and ``{mixer}.down_weight``, ``[experts, N, K]``.
    Under tensor parallelism the expert channels stack one zero-padded block
    per device, as in :func:`stack_nvfp4_experts`.

    Args:
        state_dict: The checkpoint's tensors.
        num_experts: The number of routed experts per mixer.
        mixers: The MoE mixers to stack.
        num_devices: The number of devices the expert channels split across.

    Returns:
        A new state dict.
    """
    weights = dict(state_dict)
    stacks: dict[str, npt.NDArray[np.uint16]] = {}
    with _pool() as pool:
        tasks: list[Future[None]] = []
        for mixer in sorted(mixers):
            for proj in ("up_proj", "down_proj"):
                modules = [
                    f"{mixer}.experts.{e}.{proj}" for e in range(num_experts)
                ]
                shape = [int(d) for d in weights[f"{modules[0]}.weight"].shape]
                axis = _channel_axis(proj)
                channels = shape[axis]
                padded = list(shape)
                padded[axis] = num_devices * moe_channels_per_device(
                    channels, num_devices
                )
                # Zeros, for the padding.
                stack = np.zeros((num_experts, *padded), dtype=np.uint16)
                for e, module in enumerate(modules):
                    weight = weights.pop(f"{module}.weight")
                    if weight.dtype != DType.bfloat16:
                        raise ValueError(
                            f"'{module}' is {weight.dtype}; routed experts "
                            "outside the W4A4 path are read as BF16"
                        )
                    slots = _device_slots(stack[e], axis, num_devices, channels)
                    bits = _bytes(weight).view(np.uint16).reshape(shape)
                    for slot, rows in zip(
                        slots, np.split(bits, len(slots)), strict=True
                    ):
                        tasks.append(
                            pool.submit(
                                np.copyto, slot, rows.reshape(slot.shape)
                            )
                        )
                stacks[f"{mixer}.{proj.removesuffix('_proj')}_weight"] = stack
        for task in tasks:
            task.result()
    for name, stack in stacks.items():
        weights[name] = _weight(name, stack)
    return weights


def prepare_nvfp4_linears(
    state_dict: Mapping[str, WeightData], config: NemotronHConfig, n: int
) -> dict[str, WeightData]:
    """Lays out the dense NVFP4 modules' scales as their layers read them.

    Each weight's block scales are split into one share per device and each
    share is interleaved for the block-scaled matmul (see
    :func:`~.layers.quantized.nvfp4_block_scale_shape`). Its
    ``weight_scale_2`` becomes a one-element float32 vector.

    Args:
        state_dict: The checkpoint's tensors.
        config: The model config.
        n: The number of devices.

    Returns:
        A new state dict.
    """
    weights = dict(state_dict)
    for module, fmt in config.quant_scheme.quantized.items():
        if (
            fmt is not ModuleFormat.NVFP4_WEIGHT_ONLY
            or ".experts." in module
            or config.runs_as_routed_experts(module)
        ):
            continue
        parallelism = linear_parallelism(module)
        scales = _bytes(weights[f"{module}.weight_scale"])
        if parallelism is Parallelism.COLUMN:
            shares = np.split(scales, n, axis=0)
        elif parallelism is Parallelism.ROW:
            shares = np.split(scales, n, axis=1)
        else:
            shares = [scales]
        block = np.stack([interleave_nvfp4_scales(s) for s in shares])
        name = f"{module}.weight_scale"
        weights[name] = _weight(name, block, DType.float8_e4m3fn)
        name = f"{module}.weight_scale_2"
        global_scale = np.array([_scalar(weights[name])], dtype=np.float32)
        weights[name] = _weight(name, global_scale, DType.float32)
    return weights


def _group_rows_by_device(
    rows: npt.NDArray[np.uint8], segments: Sequence[int], n: int
) -> npt.NDArray[np.uint8]:
    """Regroups rows that stack ``segments`` into one block per device.

    Device ``d``'s block holds the ``d``-th ``1 / n`` of every segment, in
    segment order, so splitting the result into ``n`` equal blocks gives each
    device its share of every segment.
    """
    starts = np.cumsum([0, *segments[:-1]])
    return np.concatenate(
        [
            rows[start + d * size // n : start + (d + 1) * size // n]
            for d in range(n)
            for start, size in zip(starts, segments, strict=True)
        ]
    )


def _relayout_rows(
    weights: dict[str, WeightData],
    name: str,
    relayout: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]],
    rows: int,
) -> None:
    """Replaces a weight with ``relayout`` of its rows, as raw bytes."""
    weight = weights[name]
    shape = tuple(int(d) for d in weight.shape)
    out = relayout(_bytes(weight).reshape(shape[0], -1))
    weights[name] = _weight(name, out, weight.dtype, (rows, *shape[1:]))


def permute_mamba_for_tp(
    state_dict: Mapping[str, WeightData], config: NemotronHConfig, n: int
) -> dict[str, WeightData]:
    """Lays out each Mamba mixer's fused rows for ``n``-way sharding.

    ``in_proj`` stacks the gate, x, B, C and dt projections and the conv
    stacks x, B and C. Sharding them on rows must give each device the rows
    of its own heads and groups in every part. An NVFP4 ``in_proj``'s block
    scales move with their rows, so this runs before
    :func:`prepare_nvfp4_linears` splits them into per-device shares.

    Args:
        state_dict: The checkpoint's tensors.
        config: The model config.
        n: The number of devices.

    Returns:
        A new state dict.
    """
    weights = dict(state_dict)
    if n == 1:
        return weights
    inner = config.mamba_intermediate_size
    group = config.n_groups * config.ssm_state_size
    conv = [inner, group, group]
    for mixer in config.mixers(LayerKind.MAMBA):
        in_proj = [inner, *conv, config.mamba_num_heads]
        relayouts = [
            (f"{mixer}.in_proj.weight", in_proj),
            (f"{mixer}.conv1d.weight", conv),
            (f"{mixer}.conv1d.bias", conv),
        ]
        if (
            config.quant_scheme.format_of(f"{mixer}.in_proj")
            is ModuleFormat.NVFP4_WEIGHT_ONLY
        ):
            relayouts.append((f"{mixer}.in_proj.weight_scale", in_proj))
        for name, segments in relayouts:
            _relayout_rows(
                weights,
                name,
                functools.partial(
                    _group_rows_by_device, segments=segments, n=n
                ),
                sum(segments),
            )
    return weights


def repeat_kv_heads_for_tp(
    state_dict: Mapping[str, WeightData], config: NemotronHConfig, n: int
) -> dict[str, WeightData]:
    """Repeats each KV head once per device in its group.

    With more devices than KV heads, column parallelism alone would split a
    head. Repeating each head ``n // num_key_value_heads`` times in place
    gives device ``d`` head ``d // (n // num_key_value_heads)``.

    Args:
        state_dict: The checkpoint's tensors, with ``k_proj`` and ``v_proj``
            in BF16.
        config: The model config.
        n: The number of devices.

    Returns:
        A new state dict.
    """
    weights = dict(state_dict)
    kv_heads = config.num_key_value_heads
    if n <= kv_heads:
        return weights
    head_dim = config.head_dim
    for mixer in config.mixers(LayerKind.ATTENTION):
        for proj in ("k_proj", "v_proj"):
            _relayout_rows(
                weights,
                f"{mixer}.{proj}.weight",
                lambda rows: np.repeat(
                    rows.reshape(kv_heads, head_dim, -1), n // kv_heads, axis=0
                ).reshape(n * head_dim, -1),
                n * head_dim,
            )
    return weights
