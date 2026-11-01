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
"""Manual TP4 bitwise check of MiMo's replicated residual stream on B200.

The model is the first ``LAYERS`` layers of MiMo-V2.6 at full width with
nonzero weights (:func:`replica_probe.replica_weights`): a dense layer, then
three MoE layers. Each layer has two allreduces. On the MoE layers the
attention output's reduction, residual add and ``post_attention_layernorm``
fuse into the allreduce composite that produces the router input; layer 0's
attention reduction and every MLP or MoE output's reduction stay plain. Ops
``0`` through ``ALLREDUCES - 1`` number them in execution order, where
``2 * layer`` is the attention's, ``2 * layer + 1`` the MLP's or MoE's, and
ops 2, 4 and 6 are the composites.

Every row count runs as its own step, one sequence of ``rows`` independent
random rows. The kernel each op takes on four B200s at 8 KiB per row:

- 1-128 rows: plain Lamport and fused Lamport;
- 129-512 rows: plain one-stage and the exact fused two-stage;
- 513 rows and up: two-stage everywhere.

The test plants an operand spread by scaling the four devices' shares by
``(1, 2^-24, 1, 2^-24)`` just before a reduction. A sum of bf16 shares of
similar size is exact in f32, so the two large shares often sum to a bf16
rounding tie, and the order in which the two tiny shares are absorbed then
decides the rounding. A rank-rotated sum seeds each rank's f32 accumulation
with its own share, so under a rotated order a spread op makes almost every
row it touches differ across devices, as the reverted-kernel probe below
records. The factors are exact in bf16 and live only in the test's trace;
the production graph is unchanged.

A divergence at op ``k`` leaves every later row incomparable, so plan ``k``
spreads ops ``k`` through ``ALLREDUCES - 1`` and leaves the earlier ops as
the model computes them: op ``k`` sees identical inputs on every device.
Every plan must match bitwise on every device at every row count. The
513-row count is the control rather than a second check: there the plain
path is the exact two-stage kernel too, so the ranks agree even without rank
ordering and a mismatch means the harness is wrong, not the kernel.

The bitwise check is the can-fail evidence for the rank-ordered kernels: a
validation-lane probe that reverts #104227's kernel hunks sees mismatches at
1, 49, 128, 129, 300 and 512 rows, and the kernel-level
``test_allreduce_determinism`` covers the same property at the kernel. Which
allreduces fuse is pinned by the post-fusion IR tests, which compile this same
trace at full width.

With ``MIMO_REPLICA_TP=2`` the same steps run as a smoke check on two GPUs,
with 16 routed experts and one row count from each regime above. A two-way
sum does not depend on its order and MiMo routes locally at TP2 anyway, so
the smoke run checks the step path, not the cross-rank agreement.

There is no verify-graph arm: verify runs the same layers, and the
post-fusion IR test checks that its staging reduces on every device.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import ml_dtypes
import numpy as np
import pytest
from max import functional as F
from max import tree
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import MiMoV2
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    MiMoV2Config,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.paged_kv_cache.jenga_cache_manager import (
    JengaKVCacheManager,
)
from max.pipelines.modeling.types import RequestID
from max.tensor import Tensor, compilation, default_device, default_dtype
from max.tensor.sharding import DeviceMesh
from mimo_v3_runner import kv_params
from replica_probe import (
    ALLREDUCES_PER_LAYER,
    MiMoReplicaTrace,
    first_four_layer_config,
    stage_replica_trace,
)

NUM_GPUS = int(os.environ.get("MIMO_REPLICA_TP", "4"))
SMOKE = NUM_GPUS == 2
HIDDEN = 4096
LAYERS = 4
MOE_LAYERS = {1, 2, 3}
ALLREDUCES = ALLREDUCES_PER_LAYER * LAYERS
SPREAD = tuple(2.0 ** (-24 * (device % 2)) for device in range(NUM_GPUS))
if SMOKE:
    LAMPORT_ROWS: tuple[int, ...] = (1,)
    ONE_STAGE_ROWS: tuple[int, ...] = (129,)
    TWO_STAGE_ROWS: tuple[int, ...] = (513,)
else:
    LAMPORT_ROWS = (1, 49, 128)
    ONE_STAGE_ROWS = (129, 300, 512)
    TWO_STAGE_ROWS = (513,)
ROWS = LAMPORT_ROWS + ONE_STAGE_ROWS + TWO_STAGE_ROWS


@dataclass
class _Layer:
    router: list[np.ndarray]
    experts: list[np.ndarray]
    mid: list[np.ndarray]
    out: list[np.ndarray]

    def fields(self) -> dict[str, list[np.ndarray]]:
        return {
            "router": self.router,
            "experts": self.experts,
            "mid": self.mid,
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
    return (
        Buffer.from_numpy(bits)
        .view(DType.bfloat16, list(bits.shape))
        .to(device)
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

    layers: list[_Layer] = []
    for layer in range(LAYERS):
        router = take()
        experts = take() if layer in MOE_LAYERS else []
        layers.append(_Layer(router, experts, take(), take()))
    assert next(shards, None) is None
    return layers


def _mismatches(shards: list[np.ndarray]) -> int:
    return sum(int(np.count_nonzero(x != shards[0])) for x in shards[1:])


def _assert_replicas_equal(
    rows: int, plan: int, h0: np.ndarray, layers: list[_Layer]
) -> None:
    previous = h0
    for index, layer in enumerate(layers):
        for field, shards in layer.fields().items():
            if shards:
                count = _mismatches(shards)
                assert count == 0, (rows, plan, index, field, count)
        # Each reduction must move the residual, or its op checked nothing.
        assert np.any(layer.mid[0] != previous), (rows, plan, index, "attn")
        assert np.any(layer.out[0] != layer.mid[0]), (rows, plan, index, "mlp")
        previous = layer.out[0]


def _compile_trace(
    model: MiMoV2,
    refs: list[DeviceRef],
    session: InferenceSession,
) -> compilation.CompiledCallable[..., list[Tensor]]:
    """Compiles the trace, then initializes it.

    Through ``compile_reusing_mefs`` so a precompiling target answers the
    compile from its record. The fused-op check needs the dump only a real
    compile writes, so it lives in the IR test instead.
    """
    staged, registry = stage_replica_trace(MiMoReplicaTrace(model), refs)
    compiled = session.compile_reusing_mefs(staged.graph)
    engine_model = session.init(compiled, weights_registry=registry)
    return compilation.CompiledCallable.from_engine_model(
        staged, compiled, engine_model, registry
    )


def _run_step(
    compiled: compilation.CompiledCallable[..., list[Tensor]],
    manager: JengaKVCacheManager,
    device: Accelerator,
    h0: np.ndarray,
    plant: np.ndarray,
) -> list[_Layer]:
    rows = h0.shape[0]
    context = TextContext(
        request_id=RequestID(),
        max_length=rows + 1,
        tokens=TokenBuffer(np.zeros(rows, dtype=np.int64)),
    )
    manager.claim(context)
    try:
        manager.alloc(context)
        kv_inputs = manager.runtime_inputs([[context]])
        offsets = Buffer.from_numpy(np.array([0, rows], dtype=np.uint32)).to(
            device
        )
        plant_bits = plant.astype(ml_dtypes.bfloat16).view(np.uint16)
        outputs = compiled(
            _bf16_buffer(h0, device),
            _bf16_buffer(plant_bits, device),
            offsets,
            *tree.leaves(kv_inputs),
        )
        return _layers(outputs)
    finally:
        manager.release(context)


def run_replica_check() -> None:
    """Runs every step in the production configuration."""
    assert accelerator_count() >= NUM_GPUS, f"requires {NUM_GPUS} GPUs"
    refs = [DeviceRef.GPU(i) for i in range(NUM_GPUS)]
    devices = [Accelerator(i) for i in range(NUM_GPUS)]
    mesh = DeviceMesh(tuple(devices), (NUM_GPUS,), ("tp",))
    hf = first_four_layer_config()
    if SMOKE:
        # Every expert is a separate weight stack, and the smoke run checks
        # the step path, not the full expert count.
        hf.n_routed_experts = 16
    params = kv_params(hf, refs, 128)
    config = MiMoV2Config.from_huggingface_config(
        hf, devices=refs, kv_params=params, max_seq_len=max(ROWS) + 1
    )
    session = InferenceSession(devices=[CPU(), *devices])
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = MiMoV2(config)
    # Both arms route on every device: the gate follows the mode, not the
    # device count.
    assert model.route_on_every_device
    compiled = _compile_trace(model, refs, session)
    # The pool is 1 GB over the mesh, so a device holds one 256 MB page at TP4
    # and two at TP2; a 514-token sequence needs a few MB of it.
    manager = JengaKVCacheManager.create(
        params=config.kv_params,
        available_bytes=1024**3,
        max_batch_size=1,
        max_seq_len=max(ROWS) + 1,
    )

    for rows in ROWS:
        for plan in range(ALLREDUCES):
            h0 = _h0(rows, plan)
            layers = _run_step(compiled, manager, devices[0], h0, _plant(plan))
            _assert_replicas_equal(rows, plan, h0, layers)
            print(f"rows={rows} plan={plan}: bitwise equal across ranks")


# MIMO_REPLICA_TP picks the arm, so each target collects both names and runs
# the one its GPU count selects.
@pytest.mark.skipif(SMOKE, reason="the TP4 target runs the bitwise check")
def test_mimo_replicas_are_bitwise_equal() -> None:
    run_replica_check()


@pytest.mark.skipif(not SMOKE, reason="the smoke target runs on two GPUs")
def test_mimo_replica_steps_run_on_two_gpus() -> None:
    run_replica_check()
