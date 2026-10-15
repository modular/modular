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
matmul. Everything else a kernel or tensor parallelism needs, the graph does
at init.
"""

from __future__ import annotations

import math
import os
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.graph.weights.weights import Shape


def convert_nemotron_h_state_dict(
    state_dict: dict[str, Weights], **unused_kwargs: object
) -> dict[str, WeightData]:
    """Returns the tensors Nemotron-H reads, under their own names.

    The router gate weight keeps its stored dtype, and
    :func:`cast_router_gates` casts it to the router's at prepare time.
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
        if name.endswith(".mixer.conv1d.weight"):
            dim, _, kernel = (int(d) for d in data.shape)
            data = WeightData(
                data=data.to_buffer().view(data.dtype, (dim, kernel)),
                name=name,
                dtype=data.dtype,
                shape=Shape([dim, kernel]),
            )
        weights[name] = data
    return weights


def cast_router_gates(
    state_dict: dict[str, WeightData], dtype: DType
) -> dict[str, WeightData]:
    """Returns ``state_dict`` with every MoE router gate weight in ``dtype``,
    the router module's (:attr:`NemotronHConfig.router_gate_dtype`)."""
    return {
        name: (
            data.astype(dtype)
            if name.endswith(".mixer.gate.weight") and data.dtype != dtype
            else data
        )
        for name, data in state_dict.items()
    }


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
    dtype: DType,
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


def _stack(
    pool: ThreadPoolExecutor, name: str, parts: Sequence[WeightData]
) -> WeightData:
    """Stacks ``parts`` on a new first axis, copying them in parallel."""
    shape = tuple(int(d) for d in parts[0].shape)
    size = math.prod(shape) * parts[0].dtype.size_in_bytes
    out = np.empty((len(parts), size), np.uint8)
    tasks = [
        pool.submit(np.copyto, out[i], _bytes(part).reshape(-1))
        for i, part in enumerate(parts)
    ]
    for task in tasks:
        task.result()
    return _weight(name, out, parts[0].dtype, (len(parts), *shape))


def stack_nvfp4_experts(
    state_dict: Mapping[str, WeightData],
    num_experts: int,
    mixers: Collection[str],
) -> dict[str, WeightData]:
    """Stacks the NVFP4 routed experts of ``mixers`` for the W4A4 matmul.

    Each mixer's per-expert ``up_proj`` and ``down_proj`` tensors become
    ``{mixer}.up_weight`` (packed E2M1, ``[experts, N, K / 2]``),
    ``{mixer}.up_block_scale`` (E4M3, ``[experts, N, K / 16]``),
    ``{mixer}.up_scale`` (float32 ``weight_scale_2``, ``[experts]``) and the
    three ``down_`` tensors.

    Args:
        state_dict: The checkpoint's tensors.
        num_experts: The number of routed experts per mixer.
        mixers: The MoE mixers to stack (see
            :meth:`NemotronHConfig.w4a4_mixers`).

    Returns:
        A new state dict.
    """
    weights = dict(state_dict)
    with _pool() as pool:
        for mixer in sorted(mixers):
            for proj in ("up_proj", "down_proj"):
                modules = [
                    f"{mixer}.experts.{e}.{proj}" for e in range(num_experts)
                ]
                prefix = f"{mixer}.{proj.removesuffix('_proj')}"
                for suffix, tensor in (
                    ("weight", "weight"),
                    ("block_scale", "weight_scale"),
                ):
                    name = f"{prefix}_{suffix}"
                    weights[name] = _stack(
                        pool,
                        name,
                        [weights.pop(f"{m}.{tensor}") for m in modules],
                    )
                name = f"{prefix}_scale"
                global_scales = np.array(
                    [
                        _scalar(weights.pop(f"{m}.weight_scale_2"))
                        for m in modules
                    ],
                    dtype=np.float32,
                )
                weights[name] = _weight(name, global_scales, DType.float32)
    return weights


def stack_bf16_experts(
    state_dict: Mapping[str, WeightData],
    num_experts: int,
    mixers: Collection[str],
) -> dict[str, WeightData]:
    """Stacks the BF16 routed experts of ``mixers`` for the grouped matmul.

    Each mixer's per-expert ``up_proj`` and ``down_proj`` weights become
    ``{mixer}.up_weight`` and ``{mixer}.down_weight``, ``[experts, N, K]``.

    Args:
        state_dict: The checkpoint's tensors.
        num_experts: The number of routed experts per mixer.
        mixers: The MoE mixers to stack.

    Returns:
        A new state dict.
    """
    weights = dict(state_dict)
    with _pool() as pool:
        for mixer in sorted(mixers):
            for proj in ("up_proj", "down_proj"):
                parts = []
                for e in range(num_experts):
                    module = f"{mixer}.experts.{e}.{proj}"
                    weight = weights.pop(f"{module}.weight")
                    if weight.dtype != DType.bfloat16:
                        raise ValueError(
                            f"'{module}' is {weight.dtype}; routed experts "
                            "outside the W4A4 path are read as BF16"
                        )
                    parts.append(weight)
                name = f"{mixer}.{proj.removesuffix('_proj')}_weight"
                weights[name] = _stack(pool, name, parts)
    return weights
