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
"""Tests that a verify with no drafts lands the linear state on its forward.

Each case runs one Gated DeltaNet block through the speculative snapshot,
verify and rollback, and the same block through the base forward on its own
copy of the pools. A verify width of zero is the committed path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Dim, Graph, TensorType, ops
from max.nn.comm import Signals
from max.nn.kv_cache import (
    MHAKVCacheParams,
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
)
from max.nn.state_space import verify_width_operand
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.qwen3_5 import (
    Qwen3_5,
    Qwen3_5LinearAttentionBlock,
)
from max.pipelines.architectures.qwen3_5.state_cache import (
    layer_state_access,
    linear_state_regions,
    ring_len_for_window,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.spec_state import (
    LIVE_CONV_POOLS,
    LIVE_CONV_ROW_IDS,
    LIVE_RECURRENT_POOLS,
    LIVE_RECURRENT_ROW_IDS,
    RING_POOLS,
    RING_ROW_IDS,
    SHADOW_RECURRENT_POOLS,
    Qwen3_5RecurrentState,
)
from max.pipelines.architectures.unified_mtp_qwen3_5.state_rollback import (
    shadow_row_ids,
)

NUM_DRAFTS = 3
HIDDEN = 32
HEAD_DIM = 16
# The only head shape the gated-delta kernels are compiled for.
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
NUM_K_HEADS = 1
NUM_V_HEADS = 2
CONV_KERNEL = 4
POOL_ROWS = 7
SCRATCH_ROWS = 4


def _config() -> Qwen3_5Config:
    devices = [DeviceRef.GPU()]
    kv_params = MHAKVCacheParams(
        dtype=DType.float32,
        devices=devices,
        n_kv_heads=2,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=HEAD_DIM,
    )
    return Qwen3_5Config(
        hidden_size=HIDDEN,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=1024,
        intermediate_size=HIDDEN * 2,
        interleaved_rope_weights=True,
        vocab_size=64,
        dtype=DType.float32,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=kv_params,
        norm_dtype=DType.float32,
        rms_norm_eps=1e-6,
        attention_multiplier=float(HEAD_DIM) ** -0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=devices,
        clip_qkv=None,
        layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=KEY_HEAD_DIM,
        linear_value_head_dim=VALUE_HEAD_DIM,
        linear_num_key_heads=NUM_K_HEADS,
        linear_num_value_heads=NUM_V_HEADS,
        linear_conv_kernel_dim=CONV_KERNEL,
        partial_rotary_factor=0.25,
        use_subgraphs=False,
    )


class _Harness:
    """One compiled graph for an arm and a pair of verify widths."""

    def __init__(
        self, ring: bool, forward_width: int, rollback_width: int
    ) -> None:
        config = _config()
        target = Qwen3_5(config)
        block = target.layers[target.linear_layer_indices[0]]
        assert isinstance(block, Qwen3_5LinearAttentionBlock)
        ring_len = ring_len_for_window(1 + NUM_DRAFTS) if ring else 0
        regions = linear_state_regions(
            num_linear_layers=1,
            key_head_dim=KEY_HEAD_DIM,
            num_key_heads=NUM_K_HEADS,
            value_head_dim=VALUE_HEAD_DIM,
            num_value_heads=NUM_V_HEADS,
            conv_kernel_dim=CONV_KERNEL,
            dtype=config.state_dtype,
            num_devices=1,
            ring_len=ring_len,
        )
        self.pool_shapes = {
            "conv": (POOL_ROWS, *regions[0].row_shape),
            "recurrent": (POOL_ROWS, *regions[1].row_shape),
        }
        self.scratch_shapes = (
            [(SCRATCH_ROWS, *region.row_shape) for region in regions[2:]]
            if ring
            else [(SCRATCH_ROWS, *regions[1].row_shape)]
        )

        rng = np.random.default_rng(3)
        raw = target.raw_state_dict()
        target.load_state_dict(
            {
                name: Buffer.from_numpy(
                    (0.2 * rng.standard_normal(w.shape.static_dims)).astype(
                        np.float32
                    )
                )
                for name, w in raw.items()
            },
            weight_alignment=1,
            strict=False,
        )

        gpu = DeviceRef.GPU()
        dtype = config.state_dtype
        rows_type = TensorType(DType.uint32, [1, "batch_size"], device=gpu)
        types: list[TensorType | BufferType] = [
            TensorType(DType.float32, ["total_seq_len", HIDDEN], device=gpu),
            TensorType(DType.uint32, ["offsets_len"], device=gpu),
            TensorType(DType.int64, ["batch_size"], device=gpu),
            rows_type,
            rows_type,
            *Signals([gpu]).input_types(),
            *(
                BufferType(dtype, list(shape), device=gpu)
                for shape in (
                    self.pool_shapes["conv"],
                    self.pool_shapes["recurrent"],
                    self.pool_shapes["conv"],
                    self.pool_shapes["recurrent"],
                )
            ),
            *(
                BufferType(
                    DType.float32 if ring else dtype, list(shape), device=gpu
                )
                for shape in self.scratch_shapes
            ),
        ]
        with Graph("commit_on_forward", input_types=types) as graph:
            x, offsets, accepted, conv_rows, rec_rows = (
                v.tensor for v in graph.inputs[:5]
            )
            signals, conv, rec, base_conv, base_rec, *scratch = (
                v.buffer for v in graph.inputs[5:]
            )
            state = Qwen3_5RecurrentState(target, regions, ring_len)
            extra: dict[str, Any] = {
                LIVE_CONV_POOLS: [conv],
                LIVE_RECURRENT_POOLS: [rec],
                LIVE_CONV_ROW_IDS: [conv_rows],
                LIVE_RECURRENT_ROW_IDS: [rec_rows],
                SHADOW_RECURRENT_POOLS: None if ring else scratch,
                RING_POOLS: scratch if ring else None,
                # Any rows the pool holds work, and the dense layout is one.
                RING_ROW_IDS: [shadow_row_ids(1, gpu)] if ring else None,
            }
            leaves = state.snapshot(extra, [gpu])
            with state.capturing(1, Dim(forward_width)):
                spec_out = block(
                    [x], [signals], layer_state_access(leaves, 0), [offsets]
                )
            if rollback_width != forward_width:
                state.num_draft_tokens = Dim(rollback_width)
                state.verify_width = verify_width_operand(rollback_width)
            state.roll_forward(
                extra,
                merged_offsets=offsets,
                num_accepted=accepted,
                num_draft_tokens=ops.constant(0, DType.int64, device=gpu),
                total_rows=x.shape[0],
                signal_buffers=[signals],
                device=gpu,
            )
            base = [
                RecurrentStateInputsPerDevice(
                    leaves=(
                        RecurrentLeafInputs(
                            pool=base_conv, live_row_ids=conv_rows
                        ),
                        RecurrentLeafInputs(
                            pool=base_rec, live_row_ids=rec_rows
                        ),
                    )
                )
            ]
            base_out = block(
                [x], [signals], layer_state_access(base, 0), [offsets]
            )
            graph.output(spec_out[0], base_out[0])

        self.device = Accelerator()
        session = InferenceSession(devices=[self.device])
        self.model = session.load(graph, weights_registry=target.state_dict())
        self.signal_buffers = Signals.allocate([self.device])

    def pools(self, seed: int) -> dict[str, Buffer]:
        """Returns fresh device pools, the spec and base copies equal."""
        rng = np.random.default_rng(seed)
        conv = rng.standard_normal(self.pool_shapes["conv"]).astype(np.float32)
        rec = (0.1 * rng.standard_normal(self.pool_shapes["recurrent"])).astype(
            np.float32
        )
        scratch = [
            rng.standard_normal(shape).astype(np.float32)
            for shape in self.scratch_shapes
        ]
        return {
            "conv": self._buf(conv),
            "recurrent": self._buf(rec),
            "base_conv": self._buf(conv),
            "base_recurrent": self._buf(rec),
            **{f"scratch{i}": self._buf(s) for i, s in enumerate(scratch)},
        }

    def _buf(self, values: np.ndarray) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(values)).to(self.device)

    def step(
        self, pools: dict[str, Buffer], seq_lengths: list[int], seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Runs one zero-draft step over ``pools`` and returns both readouts."""
        rng = np.random.default_rng(seed)
        batch = len(seq_lengths)
        x = rng.standard_normal((sum(seq_lengths), HIDDEN)).astype(np.float32)
        offsets = np.concatenate([[0], np.cumsum(seq_lengths)]).astype(
            np.uint32
        )
        conv_rows = np.array([[2 * b + 1 for b in range(batch)]], np.uint32)
        rec_rows = np.array(
            [[POOL_ROWS - 1 - b for b in range(batch)]], np.uint32
        )
        spec_out, base_out = self.model.execute(
            self._buf(x),
            self._buf(offsets),
            self._buf(np.zeros(batch, dtype=np.int64)),
            self._buf(conv_rows),
            self._buf(rec_rows),
            *self.signal_buffers,
            *pools.values(),
        )
        return _host(spec_out), _host(base_out)


def _host(value: Any) -> np.ndarray:
    return np.array(value.to(CPU()).to_numpy())


_HARNESSES: dict[tuple[bool, int, int], _Harness] = {}


def _harness(ring: bool, forward_width: int, rollback_width: int) -> _Harness:
    key = (ring, forward_width, rollback_width)
    if key not in _HARNESSES:
        _HARNESSES[key] = _Harness(ring, forward_width, rollback_width)
    return _HARNESSES[key]


def _run(
    ring: bool,
    chunks: list[list[int]],
    forward_width: int = 0,
    rollback_width: int | None = None,
) -> dict[str, np.ndarray]:
    """Runs ``chunks`` as consecutive zero-draft steps on the same pools."""
    harness = _harness(
        ring,
        forward_width,
        forward_width if rollback_width is None else rollback_width,
    )
    pools = harness.pools(seed=5)
    readouts = [
        harness.step(pools, lengths, seed=7 + i)
        for i, lengths in enumerate(chunks)
    ]
    out = {name: _host(b) for name, b in pools.items()}
    out["readout"] = np.concatenate([spec for spec, _ in readouts])
    out["base_readout"] = np.concatenate([base for _, base in readouts])
    return out


_CASES = [
    [[1]],
    [[2]],
    [[3]],
    [[4]],
    [[5]],
    [[64]],
    [[564]],
    # One prompt prefilled in two chunks.
    [[300], [264]],
    # A one-token decode row riding a prompt's launch.
    [[1, 564]],
]


@pytest.mark.parametrize("ring", [True, False], ids=["ring", "snapshot"])
@pytest.mark.parametrize("chunks", _CASES)
def test_a_committed_step_matches_the_base_forward(
    ring: bool, chunks: list[list[int]]
) -> None:
    """Checks the pools and readout equal the base forward's, bit for bit."""
    out = _run(ring, chunks)
    np.testing.assert_array_equal(out["conv"], out["base_conv"])
    np.testing.assert_array_equal(out["recurrent"], out["base_recurrent"])
    np.testing.assert_array_equal(out["readout"], out["base_readout"])


@pytest.mark.parametrize("ring", [True, False], ids=["ring", "snapshot"])
# A width-one plan holds two rows per request, so longer prompts would replay
# past it.
@pytest.mark.parametrize("length", [1, 2])
def test_rolling_back_a_committed_step_is_detectable(
    ring: bool, length: int
) -> None:
    """Checks a rollback left enabled at width zero corrupts the pools."""
    out = _run(ring, [[length]], forward_width=0, rollback_width=1)
    assert not (
        np.array_equal(out["conv"], out["base_conv"])
        and np.array_equal(out["recurrent"], out["base_recurrent"])
    )


@pytest.mark.parametrize("chunks", _CASES)
def test_both_rollbacks_commit_the_same_state(chunks: list[list[int]]) -> None:
    """Checks the ring and snapshot arms leave identical pools and readouts."""
    ring = _run(True, chunks)
    snapshot = _run(False, chunks)
    for key in ("conv", "recurrent", "readout"):
        np.testing.assert_array_equal(ring[key], snapshot[key])
