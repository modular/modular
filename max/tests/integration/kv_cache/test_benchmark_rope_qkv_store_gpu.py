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
"""Times each fused K/V write against the launches it replaces.

One rank of the published TP4 drafter on one B200, at c1 and c8 decode: the
cache's own 128-page geometry. Two pairs are timed, the drafter's block
(``rope-qkv`` / ``fused-qkv``) and its context writer (``rope-kv`` /
``fused-kv``); the reference arm of each is the sequence of graph ops the
fused op replaces -- a K store, a V store, the fused QK rope, and the
cast/mul/cast chain that scaled V -- recorded as one frame, so the reported
delta is the extra launches plus the elementwise work the fusion absorbs.

Each repetition overwrites the same slots, so the interval holds one launch's
cost and no allocation. Each fused op runs once on each of the drafter's five
layers a step, so a pair's per-step saving is five times its reported delta.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, DeviceEvent
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.nn.kv_cache import MHAKVCacheParams
from rope_qkv_store_testlib import (
    FUSED_KV,
    FUSED_QKV,
    ROPE_KV,
    ROPE_QKV,
    Shape,
    bits,
    build_graph,
    graph_inputs,
    params_of,
    projection,
    rope_freqs,
)
from test_common.simple_kv_cache import paged_kv_cache_inputs

_SHAPE = Shape("drafter-tp4-fp32", 16, 2, 128, 64, DType.float32)
_PAIRS = ((ROPE_QKV, FUSED_QKV), (ROPE_KV, FUSED_KV))
_SCALE = 0.707


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


@pytest.fixture(scope="module")
def loaded() -> tuple[Accelerator, MHAKVCacheParams, dict[str, Model]]:
    device = Accelerator()
    session = InferenceSession(devices=[device])
    params = params_of(_SHAPE)
    models = {
        mode: session.load(
            build_graph(params, _SHAPE, _SCALE, mode=mode, observe_cache=False)
        )
        for mode in (ROPE_QKV, FUSED_QKV, ROPE_KV, FUSED_KV)
    }
    return device, params, models


@pytest.mark.parametrize("batch", [1, 8], ids=["c1", "c8"])
def test_benchmark_rope_qkv_store(
    loaded: tuple[Accelerator, MHAKVCacheParams, dict[str, Model]], batch: int
) -> None:
    device, params, models = loaded
    prompt_lens = [1] * batch
    cache_lens = [128] * batch
    offsets = Buffer.from_numpy(np.arange(batch + 1, dtype=np.uint32)).to(
        device
    )
    for reference, fused in _PAIRS:
        # K and V get their own epochs, so a swapped read cannot pass parity.
        values = (
            Buffer.from_dlpack(
                projection(batch, 0, _SHAPE.num_q_heads, _SHAPE.head_dim)
            ).to(device),
            Buffer.from_dlpack(
                projection(batch, 0, _SHAPE.num_kv_heads, _SHAPE.head_dim)
            ).to(device),
            Buffer.from_dlpack(
                projection(batch, 5, _SHAPE.num_kv_heads, _SHAPE.head_dim)
            ).to(device),
            offsets,
            Buffer.from_dlpack(rope_freqs(_SHAPE)).to(device),
        )
        caches = [
            paged_kv_cache_inputs(params, prompt_lens, cache_lengths=cache_lens)
            for _ in range(2)
        ]
        inputs = [
            graph_inputs(mode, values, cache)
            for mode, cache in zip((reference, fused), caches, strict=True)
        ]
        outputs = [
            model.execute(*operands)
            for model, operands in zip(
                (models[reference], models[fused]), inputs, strict=True
            )
        ]
        if fused == FUSED_QKV:
            assert torch.equal(bits(outputs[0][0]), bits(outputs[1][0]))
        assert torch.equal(bits(caches[0].kv_blocks), bits(caches[1].kv_blocks))
        device.synchronize()
        medians = []
        for mode, operands in zip((reference, fused), inputs, strict=True):
            try:
                models[mode].capture(batch, *operands)
                device.synchronize()
                samples = _measure(
                    partial(models[mode].replay, batch, *operands), device
                )
            finally:
                models[mode].release_captured_graph(batch)
            median_us = float(np.median(samples)) * 1000
            p95_us = float(np.percentile(samples, 95)) * 1000
            medians.append(median_us)
            print(
                f"c{batch} arm={mode} median_us={median_us:.3f} "
                f"p95_us={p95_us:.3f} n={len(samples)}"
            )
        print(
            f"c{batch} arm={reference} saved_us_per_step="
            f"{(medians[0] - medians[1]) * 5:.2f} (five drafter layers)"
        )
