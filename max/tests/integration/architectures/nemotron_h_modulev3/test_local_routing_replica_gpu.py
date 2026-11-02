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
"""Manual TP4 bitwise check of Nemotron-H's replicated residual stream on B200.

The model is the schedule ``LAYERS`` of :mod:`replica_probe` at the trace's
width with nonzero weights: four MoE layers, each with one allreduce on its
mixer's output. The reduction produces the residual the next layer's router
reads, so every router input inherits each earlier step's. Every reduction
takes the plain ``mo.distributed.allreduce.sum``, as production's do: the
trace stages each block as a subgraph, and ``test_local_routing_ir`` pins the
form in a post-fusion dump.

Every row count runs as its own step, one sequence of ``rows`` independent
random rows. At 8 KiB per row the kernel each op takes on four B200s is:

- 1-128 rows: Lamport;
- 129-512 rows: one-stage;
- 513 rows and up: two-stage.

The test plants an operand spread by scaling the four devices' shares by
``(1, 2^-24, 1, 2^-24)`` just before a reduction. A sum of bf16 shares of
similar size is exact in f32, so the two large shares often sum to a bf16
rounding tie, and the order in which the two tiny shares are absorbed then
decides the rounding. A rank-rotated sum seeds each rank's f32 accumulation
with its own share, so under a rotated order a spread op makes almost every
row it touches differ across devices. The factors are exact in bf16 and live
only in the test's trace; the production graph is unchanged.

A divergence at op ``k`` leaves every later row incomparable, so plan ``k``
spreads ops ``k`` through ``ALLREDUCES - 1`` and leaves the earlier ops as the
model computes them: op ``k`` sees identical inputs on every device. Every
plan must match bitwise on every device at every row count. The 513-row count
is the control rather than a second check: there the plain path is the exact
two-stage kernel too, so the ranks agree even without rank ordering and a
mismatch means the harness is wrong, not the kernel.

With ``NEMOTRON_H_REPLICA_TP=2`` the same steps run as a smoke check on two
GPUs, with one row count from each regime above. A two-way sum does not depend
on its order, so the smoke run checks the step path, not the cross-rank
agreement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import ml_dtypes
import numpy as np
import pytest
from max import functional as F
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.pipelines.architectures.nemotron_h_modulev3.nemotron_h import (
    NemotronH,
)
from max.tensor import Tensor, compilation, default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from replica_probe import (
    ALLREDUCES_PER_LAYER,
    HIDDEN,
    LAYERS,
    NemotronHReplicaTrace,
    replica_config,
    stage_replica_trace,
)

NUM_GPUS = int(os.environ.get("NEMOTRON_H_REPLICA_TP", "4"))
SMOKE = NUM_GPUS == 2
ALLREDUCES = ALLREDUCES_PER_LAYER * len(LAYERS)
SPREAD = tuple(2.0 ** (-24 * (device % 2)) for device in range(NUM_GPUS))
if SMOKE:
    ROWS: tuple[int, ...] = (1, 129, 513)
else:
    ROWS = (1, 49, 128, 129, 300, 512, 513)


@dataclass
class _Layer:
    router: list[np.ndarray]
    experts: list[np.ndarray]
    weights: list[np.ndarray]
    out: list[np.ndarray]

    def fields(self) -> dict[str, list[np.ndarray]]:
        return {
            "router": self.router,
            "experts": self.experts,
            "weights": self.weights,
            "out": self.out,
        }


def _plant(first_spread_op: int) -> np.ndarray:
    plant = np.ones((ALLREDUCES, NUM_GPUS), np.float32)
    plant[first_spread_op:] = SPREAD
    return plant


def _h0(rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng([rows, seed])
    values = rng.standard_normal((rows, HIDDEN), dtype=np.float32)
    return values.astype(ml_dtypes.bfloat16).view(np.uint16)


def _bf16_buffer(bits: np.ndarray, device: Accelerator) -> Buffer:
    # Allocating and copying both need real hardware, which is where the
    # precompile record run of this test ends: after the compile above it, so
    # its artifact is exported, and before the step it cannot run.
    return (
        Buffer.from_numpy(bits)
        .view(DType.bfloat16, list(bits.shape))
        .copy(device=device)
    )


def _bits(value: Tensor) -> np.ndarray:
    buffer = value.driver_tensor.to(CPU())
    if buffer.dtype != DType.bfloat16:
        return buffer.to_numpy().copy()
    # DLPack has no bf16, so the bits cross as uint16.
    bits = buffer.view(DType.uint16, list(buffer.shape)).to_numpy().copy()
    assert np.isfinite(bits.view(ml_dtypes.bfloat16).astype(np.float32)).all()
    return bits


def _layers(outputs: list[Tensor]) -> list[_Layer]:
    shards = iter(outputs)

    def take() -> list[np.ndarray]:
        return [_bits(next(shards)) for _ in range(NUM_GPUS)]

    layers = [_Layer(take(), take(), take(), take()) for _ in LAYERS]
    assert next(shards, None) is None
    return layers


def _mismatches(shards: list[np.ndarray]) -> int:
    assert len(shards) == NUM_GPUS, shards
    return sum(int(np.count_nonzero(x != shards[0])) for x in shards[1:])


def _assert_replicas_equal(
    rows: int, plan: int, h0: np.ndarray, layers: list[_Layer]
) -> None:
    previous = h0
    for index, layer in enumerate(layers):
        for field, shards in layer.fields().items():
            count = _mismatches(shards)
            assert count == 0, (rows, plan, index, field, count)
        # The reduction must move the residual, or its op checked nothing.
        assert np.any(layer.out[0] != previous), (rows, plan, index, "mixer")
        previous = layer.out[0]


def _compile_trace(
    model: NemotronH,
    refs: list[DeviceRef],
    session: InferenceSession,
) -> compilation.CompiledCallable[..., list[Tensor]]:
    """Compiles the trace, then initializes it.

    Through ``compile_reusing_mefs`` so a precompiling target answers the
    compile from its record. The allreduce form needs the dump only a real
    compile writes, so it is pinned by ``test_local_routing_ir`` instead.
    """
    staged, registry = stage_replica_trace(NemotronHReplicaTrace(model), refs)
    compiled = session.compile_reusing_mefs(staged.graph)
    engine_model = session.init(compiled, weights_registry=registry)
    return compilation.CompiledCallable.from_engine_model(
        staged, compiled, engine_model, registry
    )


def _run_step(
    compiled: compilation.CompiledCallable[..., list[Tensor]],
    device: Accelerator,
    h0: np.ndarray,
    plant: np.ndarray,
) -> list[_Layer]:
    plant_bits = plant.astype(ml_dtypes.bfloat16).view(np.uint16)
    outputs = compiled(
        _bf16_buffer(h0, device),
        _bf16_buffer(plant_bits, device),
    )
    return _layers(outputs)


def run_replica_check() -> None:
    """Runs every step in the production configuration."""
    assert accelerator_count() >= NUM_GPUS, f"requires {NUM_GPUS} GPUs"
    refs = [DeviceRef.GPU(i) for i in range(NUM_GPUS)]
    devices = [Accelerator(i) for i in range(NUM_GPUS)]
    mesh = DeviceMesh(tuple(devices), (NUM_GPUS,), ("tp",))
    config = replica_config(refs)
    session = InferenceSession(devices=[CPU(), *devices])
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = NemotronH(config)
    compiled = _compile_trace(model, refs, session)

    for rows in ROWS:
        for plan in range(ALLREDUCES):
            h0 = _h0(rows, plan)
            layers = _run_step(compiled, devices[0], h0, _plant(plan))
            _assert_replicas_equal(rows, plan, h0, layers)
            print(f"rows={rows} plan={plan}: bitwise equal across ranks")


# NEMOTRON_H_REPLICA_TP picks the arm, so each target collects both names and
# runs the one its GPU count selects.
@pytest.mark.skipif(SMOKE, reason="the TP4 target runs the bitwise check")
def test_nemotron_h_replicas_are_bitwise_equal() -> None:
    run_replica_check()


@pytest.mark.skipif(not SMOKE, reason="the smoke target runs on two GPUs")
def test_nemotron_h_replica_steps_run_on_two_gpus() -> None:
    run_replica_check()
