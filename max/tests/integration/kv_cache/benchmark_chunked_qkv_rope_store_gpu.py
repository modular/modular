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
"""Times MiMo full QKV preparation graphs, excluding projection and attention.

TP labels describe one rank's published layout; this runs on one B200.
Decode uses captured replay and prefill uses eager execution. Fixed cache
lengths make each repetition overwrite the same slots without allocation.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
import torch
from max import tree
from max.driver import Accelerator, Buffer, DeviceEvent
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from test_chunked_qkv_rope_store_gpu import (
    _bits,
    _CompiledFixture,
    _graph,
    _Layout,
    _projection,
)
from test_common.simple_kv_cache import paged_kv_cache_inputs


def _measure(execute: Callable[[], object], device: Accelerator) -> list[float]:
    samples = []
    queue = device.default_queue
    for iteration in range(110):
        start = DeviceEvent(device, enable_timing=True)
        end = DeviceEvent(device, enable_timing=True)
        queue.record_event(start)
        execute()
        queue.record_event(end)
        end.synchronize()
        if iteration >= 10:
            samples.append(start.elapsed_time(end))
    return samples


@pytest.fixture(scope="module", params=[2, 4], ids=["tp2", "tp4"])
def compiled(request: pytest.FixtureRequest) -> _CompiledFixture:
    tp = request.param
    layout = _Layout(
        f"full-tp{tp}", 4 // tp, 64 // tp, 4 // tp, chunk_padding=64
    )
    device = Accelerator()
    session = InferenceSession(devices=[device])
    params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=layout.num_kv_heads,
        head_dim=256,
        num_layers=2,
        page_size=128,
        devices=[DeviceRef.GPU()],
    )
    models = tuple(
        session.load(
            _graph(
                params,
                0.707,
                fused=fused,
                layout=layout,
                observe_cache=False,
                frequency_positions=1024,
            )
        )
        for fused in (False, True)
    )
    return _CompiledFixture(
        device, session, params, layout, (models[0], models[1])
    )


@pytest.mark.parametrize(
    "batch,length",
    [(1, 1), (8, 1), (1, 512)],
    ids=["decode-c1", "decode-c8", "prefill-512"],
)
def test_benchmark_full_qkv_preparation(
    compiled: _CompiledFixture, batch: int, length: int
) -> None:
    device, layout = compiled.device, compiled.layout
    prompt_lens = [length] * batch
    caches = [
        paged_kv_cache_inputs(
            compiled.params, prompt_lens, cache_lengths=[128] * batch
        )
        for _ in compiled.models
    ]
    qkv = Buffer.from_dlpack(_projection(batch * length, 0, layout)).to(device)
    offsets = Buffer.from_numpy(
        np.arange(batch + 1, dtype=np.uint32) * length
    ).to(device)
    position = torch.arange(1024, dtype=torch.float32)[:, None]
    inv_freq = 10000000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    angle = position * inv_freq
    freqs = Buffer.from_dlpack(
        torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        .reshape(1024, 64)
        .to(torch.bfloat16)
    ).to(device)
    inputs = [[qkv, offsets, freqs, *tree.leaves(cache)] for cache in caches]
    outputs = [
        model.execute(*values)
        for model, values in zip(compiled.models, inputs, strict=True)
    ]
    assert torch.equal(_bits(outputs[0][0]), _bits(outputs[1][0]))
    assert torch.equal(_bits(caches[0].kv_blocks), _bits(caches[1].kv_blocks))
    device.synchronize()
    medians = []
    for fused, model, values in zip(
        (False, True), compiled.models, inputs, strict=True
    ):
        if length == 1:
            try:
                model.capture(batch, *values)
                device.synchronize()
                samples = _measure(
                    partial(model.replay, batch, *values), device
                )
            finally:
                model.release_captured_graph(batch)
        else:
            samples = _measure(partial(model.execute, *values), device)
        median_ms = float(np.median(samples))
        p95_ms = float(np.percentile(samples, 95))
        medians.append(median_ms)
        print(
            f"{layout.name} batch={batch} tokens/request={length} fused={fused} "
            f"median_us={median_ms * 1000:.3f} "
            f"p95_us={p95_ms * 1000:.3f} n={len(samples)}"
        )
    print(
        f"{layout.name} batch={batch} tokens/request={length} "
        f"preparation_speedup={medians[0] / medians[1]:.3f}"
    )
