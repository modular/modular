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

"""Device memory the blockwise MegaFFN's experts leave contiguous after init.

A stack of MegaFFN layers initializes in its own process, in a pool capped a
little above its experts, and then allocates one buffer of most of what is
left, as a KV cache does after the model loads.

NOTE: AMD gfx950 only.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import queue
from multiprocessing.queues import Queue

import numpy as np
import pytest
from max import functional as F
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, TensorType
from max.nn import Module, as_subgraph
from max.nn.common_layers import quant_ops
from max.nn.common_layers.quant_linear import QuantizedMLP
from max.nn.common_layers.quant_tensor import (
    FP8BlockActivation,
    QuantAwareTensor,
)
from max.nn.legacy.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.sequential import ModuleList
from max.tensor import Tensor, default_device

# Device-side stacking fails this down to two layers of nine experts; three
# layers leave two holes, and nine experts span two permute groups.
LAYERS, EXPERTS, H, I, ROWS = 3, 9, 6144, 2048, 64
EXPERT_BYTES = LAYERS * EXPERTS * 3 * I * H
POOL_BYTES = EXPERT_BYTES * 8 // 5
KV_BYTES = (POOL_BYTES - EXPERT_BYTES) * 19 // 20

_QUANT = QuantConfig(
    input_scale=InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.DYNAMIC,
        dtype=DType.float32,
        block_size=(1, 128),
    ),
    weight_scale=WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float32,
        block_size=(128, 128),
    ),
    mlp_quantized_layers={0},
    attn_quantized_layers={0},
    format=QuantFormat.BLOCKSCALED_FP8,
)


class _Layer(Module[..., Tensor]):
    def __init__(self) -> None:
        self.experts = ModuleList(
            [
                QuantizedMLP(
                    hidden_dim=H, feed_forward_length=I, quant_config=_QUANT
                )
                for _ in range(EXPERTS)
            ]
        )

    def forward(self, *args: Tensor) -> Tensor:
        x, sx, offsets, expert_ids, usage, estimated_total_m = args
        gate_up: list[list[QuantAwareTensor]] = [
            [e.gate_proj.weight, e.up_proj.weight] for e in self.experts
        ]
        down: list[list[QuantAwareTensor]] = [
            [e.down_proj.weight] for e in self.experts
        ]
        device = x.device
        return quant_ops.grouped_ffn_blockwise_fp8_amd_ep(
            FP8BlockActivation(data=x, scales=sx),
            quant_ops.stack_blockwise_mega_ffn(gate_up, device),
            quant_ops.stack_blockwise_mega_ffn(down, device),
            offsets,
            expert_ids,
            usage,
            estimated_total_m,
            256,
            ROWS,
        )


class _Model(Module[..., Tensor]):
    def __init__(self) -> None:
        self.layers = ModuleList([_Layer() for _ in range(LAYERS)])

    def forward(self, *args: Tensor) -> Tensor:
        out = as_subgraph(self.layers[0], name="layer")(*args)
        for layer in self.layers[1:]:
            out = out + as_subgraph(layer, name="layer")(*args)
        return out


def _allocate_kv_after_init(out: Queue[str]) -> None:
    device = Accelerator()
    with F.lazy(), default_device(device):
        model = _Model()
    rng = np.random.default_rng(0)
    values: dict[tuple[DType, tuple[int, ...]], Buffer] = {}
    weights: dict[str, Buffer] = {}
    for name, p in model.parameters:
        shape = tuple(int(d) for d in p.shape)
        if (p.dtype, shape) not in values:
            raw = rng.integers(0, 120, size=shape, dtype=np.uint8)
            values[p.dtype, shape] = (
                Buffer.from_numpy(raw).view(p.dtype)
                if p.dtype == DType.float8_e4m3fn
                else Buffer.from_numpy(np.ones(shape, np.float32))
            )
        weights[name] = values[p.dtype, shape]
    gpu, cpu = DeviceRef.from_device(device), DeviceRef.CPU()
    compiled = model.compile(
        TensorType(DType.float8_e4m3fn, [ROWS, H], gpu),
        TensorType(DType.float32, [H // 128, ROWS], gpu),
        TensorType(DType.uint32, [EXPERTS + 1], gpu),
        TensorType(DType.int32, [EXPERTS], gpu),
        TensorType(DType.uint32, [2], cpu),
        TensorType(DType.uint32, [], cpu),
        weights=weights,
        session=InferenceSession(devices=[device]),
    )
    try:
        Buffer(DType.uint8, [KV_BYTES], device=device)
    except Exception as e:
        out.put(str(e))
        return
    try:
        Buffer(DType.uint8, [POOL_BYTES], device=device)
        out.put("the pool is not capped, so the test proves nothing")
    except Exception:
        out.put("")
    del compiled


@pytest.mark.skipif(
    accelerator_api() != "hip", reason="the AMD MoE FFN is gfx950-only"
)
def test_experts_leave_room_for_a_contiguous_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The experts leave a 0.95-utilization KV cache one contiguous block.

    Joining, stacking or permuting them on the device left a hole between
    every two layers' weights. GLM-5.3's expert shapes.
    """
    # The child reads these as it starts, before anything it imports can.
    monkeypatch.setenv(
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE", str(POOL_BYTES)
    )
    monkeypatch.setenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY", "1")
    ctx = mp.get_context("spawn")
    out: Queue[str] = ctx.Queue()
    proc = ctx.Process(target=_allocate_kv_after_init, args=(out,))
    proc.start()
    # Read before joining: a child exits only once its result is flushed.
    error: str | None = None
    while error is None and (proc.is_alive() or not out.empty()):
        with contextlib.suppress(queue.Empty):
            error = out.get(timeout=5)
    proc.join()
    assert proc.exitcode == 0, f"init failed with exit code {proc.exitcode}"
    assert not error, error
