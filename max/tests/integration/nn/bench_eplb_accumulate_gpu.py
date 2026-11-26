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
"""Device-time cost of the graph-level EPLB routing histogram.

Static-EPLB routing profiling counts each MoE layer's logical expert ids with
`_accumulate`: a broadcast equality against every expert id, a column sum and
a read-modify-write of a per-layer counter buffer. This times one graph that
runs it over every MiniMax-M3 MoE layer against the same graph without it,
eagerly and as a replayed device graph, the way decode serves. The difference
divided by the layer count is the per-layer cost. Not a correctness test.
"""

from __future__ import annotations

import os
import time

import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.legacy.moe.expert_parallel import _accumulate

# MiniMax-M3 has 57 MoE layers, 128 routed experts, top-4. The default layer
# count is lower because loading the 57-layer graph did not finish within an
# hour on the remote pool; the per-layer cost does not depend on it.
_LAYERS = int(os.environ.get("EPLB_ACC_LAYERS", "8"))
_EXPERTS = 128
_TOP_K = 4
# Per-GPU tokens: decode at a few batch sizes, then a prefill chunk.
_TOKENS = [16, 256, 2048]
_ITERS = 200
_WARMUP = 20
_KEY = 7


def _build(with_histogram: bool, tokens: int) -> Graph:
    device_ref = DeviceRef.GPU()
    input_types: list[TensorType | BufferType] = [
        TensorType(DType.int32, [_LAYERS, tokens, _TOP_K], device=device_ref)
    ]
    input_types += [
        BufferType(DType.int64, [_EXPERTS], device=device_ref)
        for _ in range(_LAYERS)
    ]
    with Graph("bench", input_types=input_types) as graph:
        ids = graph.inputs[0].tensor
        if with_histogram:
            for layer in range(_LAYERS):
                _accumulate(
                    graph.inputs[1 + layer].buffer,
                    ops.reshape(ids[layer], [-1]),
                    device_ref,
                    _EXPERTS,
                )
        graph.output(ids[0])
    return graph


def _time(run: object, *inputs: Buffer) -> float:
    assert callable(run)
    for _ in range(_WARMUP):
        run(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(_ITERS):
        run(*inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / _ITERS * 1e3  # us per step


def _log(msg: str) -> None:
    print(msg, flush=True)


def _eager_and_replay(
    name: str, model: Model, inputs: list[Buffer]
) -> tuple[float, float]:
    eager = _time(model.execute, *inputs)
    _log(f"  {name}: eager {eager:.2f} us; capturing")
    model.capture(_KEY, *inputs)
    replay = _time(lambda *xs: model.replay(_KEY, *xs), *inputs)
    _log(f"  {name}: replay {replay:.2f} us")
    return eager, replay


def main() -> None:
    device = Accelerator(0)
    session = InferenceSession(devices=[device])
    torch.manual_seed(0)
    _log(
        f"layers={_LAYERS} experts={_EXPERTS} top_k={_TOP_K}; us per step"
        " (all layers) and per layer"
    )
    for tokens in _TOKENS:
        ids = torch.randint(
            0, _EXPERTS, (_LAYERS, tokens, _TOP_K), dtype=torch.int32
        )
        inputs = [Buffer.from_dlpack(ids).to(device)]
        inputs += [
            Buffer.from_dlpack(torch.zeros(_EXPERTS, dtype=torch.int64)).to(
                device
            )
            for _ in range(_LAYERS)
        ]
        start = time.perf_counter()
        _log(f"tokens={tokens}: compiling")
        base = session.load(_build(False, tokens))
        _log(f"  without: compiled in {time.perf_counter() - start:.0f} s")
        start = time.perf_counter()
        hist = session.load(_build(True, tokens))
        _log(f"  with: compiled in {time.perf_counter() - start:.0f} s")
        base_eager, base_replay = _eager_and_replay("without", base, inputs)
        hist_eager, hist_replay = _eager_and_replay("with", hist, inputs)
        for mode, b, h in (
            ("eager", base_eager, hist_eager),
            ("replay", base_replay, hist_replay),
        ):
            _log(
                f"tokens={tokens:5d} {mode:6s} without={b:8.2f} with={h:8.2f}"
                f" cost/step={h - b:8.2f} cost/layer={(h - b) / _LAYERS:6.2f}"
            )


def test_bench_eplb_accumulate() -> None:
    main()


if __name__ == "__main__":
    main()
