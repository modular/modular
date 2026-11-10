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
"""A mixed verify runs its prompt rows through the chunked prefill.

One Gated DeltaNet block verifies a batch of decode rows (``1 + K`` tokens)
and prompt rows longer than the verify window. The prompt rows' real tokens must give
the readout and the pools of the base forward over those tokens alone; the
decode rows' readout must equal a verify of the decode rows alone. After the
rollback, the pools must equal those two runs side by side.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    DeviceRef,
    Dim,
    Graph,
    TensorType,
    ops,
)
from max.nn.kv_cache import (
    MHAKVCacheParams,
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
)
from max.nn.legacy.comm import Signals
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.qwen3_5 import (
    Qwen3_5,
    Qwen3_5LinearAttentionBlock,
)
from max.pipelines.architectures.qwen3_5.spec_adapters import (
    Qwen3_5RecurrentState,
)
from max.pipelines.architectures.qwen3_5.state_cache import (
    layer_state_access,
    linear_state_regions,
    ring_len_for_window,
)

K = 3
HIDDEN = 32
HEAD_DIM = 16
# The only head shape the gated-delta kernels are compiled for.
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
NUM_K_HEADS = 1
NUM_V_HEADS = 2
CONV_KERNEL = 4
POOL_ROWS = 7
RING_ROWS = 4
ROLLBACK_POOL_ROWS = 11


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


def _host(value: Buffer) -> np.ndarray:
    return np.array(value.to(CPU()).to_numpy())


@dataclass(frozen=True)
class _StepOutputs:
    """A mixed verify's readout and pools beside the base forward's."""

    readout: np.ndarray
    base_readout: np.ndarray
    conv: np.ndarray
    base_conv: np.ndarray
    recurrent: np.ndarray
    base_recurrent: np.ndarray


def _seeded_block(
    config: Qwen3_5Config,
) -> tuple[Qwen3_5, Qwen3_5LinearAttentionBlock]:
    target = Qwen3_5(config)
    block = target.layers[target.linear_layer_indices[0]]
    assert isinstance(block, Qwen3_5LinearAttentionBlock)
    rng = np.random.default_rng(3)
    target.load_state_dict(
        {
            name: Buffer.from_numpy(
                (0.2 * rng.standard_normal(w.shape.static_dims)).astype(
                    np.float32
                )
            )
            for name, w in target.raw_state_dict().items()
        },
        weight_alignment=1,
        strict=False,
    )
    return target, block


class _Harness:
    """A mixed verify of one block, and the base forward of other rows."""

    def __init__(self) -> None:
        config = _config()
        target, block = _seeded_block(config)
        regions = linear_state_regions(
            num_linear_layers=1,
            key_head_dim=KEY_HEAD_DIM,
            num_key_heads=NUM_K_HEADS,
            value_head_dim=VALUE_HEAD_DIM,
            num_value_heads=NUM_V_HEADS,
            conv_kernel_dim=CONV_KERNEL,
            dtype=config.state_dtype,
            num_devices=1,
            ring_len=ring_len_for_window(1 + K),
        )
        self.conv_shape = (POOL_ROWS, *regions[0].row_shape)
        self.rec_shape = (POOL_ROWS, *regions[1].row_shape)
        self.ring_shape = (RING_ROWS, *regions[2].row_shape)

        gpu, cpu = DeviceRef.GPU(), DeviceRef.CPU()
        dtype = config.state_dtype
        rows = TensorType(DType.uint32, [1, "batch_size"], device=gpu)
        base_rows = TensorType(DType.uint32, [1, "base_batch"], device=gpu)
        types: list[TensorType | BufferType] = [
            TensorType(DType.float32, ["total_seq_len", HIDDEN], device=gpu),
            TensorType(DType.uint32, ["offsets_len"], device=gpu),
            TensorType(DType.uint32, ["offsets_len"], device=cpu),
            rows,
            rows,
            TensorType(DType.float32, ["base_len", HIDDEN], device=gpu),
            TensorType(DType.uint32, ["base_offsets_len"], device=gpu),
            base_rows,
            base_rows,
            *Signals([gpu]).input_types(),
            *(
                BufferType(dtype, list(shape), device=gpu)
                for shape in (
                    self.conv_shape,
                    self.rec_shape,
                    self.conv_shape,
                    self.rec_shape,
                )
            ),
            BufferType(DType.float32, list(self.ring_shape), device=gpu),
        ]
        with Graph("mixed_step_chunk", input_types=types) as graph:
            x, offsets, _host_offsets, conv_rows, rec_rows = (
                v.tensor for v in graph.inputs[:5]
            )
            base_x, base_offsets, base_conv_rows, base_rec_rows = (
                v.tensor for v in graph.inputs[5:9]
            )
            signals, conv, rec, base_conv, base_rec, ring = (
                v.buffer for v in graph.inputs[9:]
            )
            ring_rows = ops.range(
                start=0,
                stop=Dim("batch_size"),
                out_dim="batch_size",
                device=gpu,
                dtype=DType.uint32,
            ).reshape([1, "batch_size"])
            state = Qwen3_5RecurrentState(target, regions)
            leaves = [
                RecurrentStateInputsPerDevice(
                    leaves=(
                        RecurrentLeafInputs(pool=conv, live_row_ids=conv_rows),
                        RecurrentLeafInputs(pool=rec, live_row_ids=rec_rows),
                        RecurrentLeafInputs(pool=ring, live_row_ids=ring_rows),
                    )
                )
            ]
            with state.capturing(1, Dim(K)):
                spec_out = block(
                    [x],
                    [signals],
                    layer_state_access(leaves, 0),
                    [offsets],
                )
            base = [
                RecurrentStateInputsPerDevice(
                    leaves=(
                        RecurrentLeafInputs(
                            pool=base_conv, live_row_ids=base_conv_rows
                        ),
                        RecurrentLeafInputs(
                            pool=base_rec, live_row_ids=base_rec_rows
                        ),
                    )
                )
            ]
            base_out = block(
                [base_x], [signals], layer_state_access(base, 0), [base_offsets]
            )
            graph.output(spec_out[0], base_out[0])

        self.device = Accelerator()
        session = InferenceSession(devices=[CPU(), self.device])
        self.model = session.load(graph, weights_registry=target.state_dict())
        self.signal_buffers = Signals.allocate([self.device])

    def _buf(self, values: np.ndarray) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(values)).to(self.device)

    def run(self, lengths: list[int], long_row: int) -> _StepOutputs:
        """Verifies ``lengths`` at width ``K``; the base forward runs
        ``long_row``'s real tokens alone, on that row's pool rows. Both start
        from equal pools.
        """
        rng = np.random.default_rng(11)
        x = rng.standard_normal((sum(lengths), HIDDEN)).astype(np.float32)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)
        batch = len(lengths)
        conv_rows = np.array([[2 * b + 1 for b in range(batch)]], np.uint32)
        rec_rows = np.array(
            [[POOL_ROWS - 1 - b for b in range(batch)]], np.uint32
        )
        lo = int(offsets[long_row])
        real = lengths[long_row] - K
        conv = rng.standard_normal(self.conv_shape).astype(np.float32)
        rec = (0.1 * rng.standard_normal(self.rec_shape)).astype(np.float32)
        pools = [self._buf(p) for p in (conv, rec, conv, rec)]
        spec_out, base_out = self.model.execute(
            self._buf(x),
            self._buf(offsets),
            Buffer.from_numpy(offsets),
            self._buf(conv_rows),
            self._buf(rec_rows),
            self._buf(x[lo : lo + real]),
            self._buf(np.array([0, real], np.uint32)),
            self._buf(conv_rows[:, [long_row]]),
            self._buf(rec_rows[:, [long_row]]),
            *self.signal_buffers,
            *pools,
            self._buf(np.zeros(self.ring_shape, np.float32)),
        )
        conv_row = int(conv_rows[0, long_row])
        rec_row = int(rec_rows[0, long_row])
        return _StepOutputs(
            readout=_host(spec_out)[lo : lo + real],
            base_readout=_host(base_out),
            conv=_host(pools[0])[conv_row],
            base_conv=_host(pools[2])[conv_row],
            recurrent=_host(pools[1])[rec_row],
            base_recurrent=_host(pools[3])[rec_row],
        )


@functools.cache
def _harness() -> _Harness:
    return _Harness()


@pytest.mark.parametrize(
    ("lengths", "long_row"),
    [
        # A prompt chunk after two decode rows, and before them.
        pytest.param(
            [1 + K, 1 + K, 300 + K], 2, id="prompt-chunk-after-decode-rows"
        ),
        pytest.param(
            [300 + K, 1 + K, 1 + K], 0, id="prompt-chunk-before-decode-rows"
        ),
    ],
)
def test_a_prompt_row_matches_the_base_forward_over_its_real_tokens(
    lengths: list[int], long_row: int
) -> None:
    out = _harness().run(lengths, long_row)
    np.testing.assert_array_equal(out.readout, out.base_readout)
    np.testing.assert_array_equal(out.recurrent, out.base_recurrent)
    np.testing.assert_array_equal(out.conv, out.base_conv)


@dataclass(frozen=True)
class _RollbackOutputs:
    """A rolled-back mixed verify beside the same rows run apart."""

    prompt_rows: int
    readout: np.ndarray
    base_readout: np.ndarray
    conv: np.ndarray
    base_conv: np.ndarray
    recurrent: np.ndarray
    base_recurrent: np.ndarray


class _RollbackHarness:
    """A verify and its rollback at width ``k``, and the base forward."""

    def __init__(self, k: int) -> None:
        config = _config()
        regions = linear_state_regions(
            num_linear_layers=1,
            key_head_dim=KEY_HEAD_DIM,
            num_key_heads=NUM_K_HEADS,
            value_head_dim=VALUE_HEAD_DIM,
            num_value_heads=NUM_V_HEADS,
            conv_kernel_dim=CONV_KERNEL,
            dtype=config.state_dtype,
            num_devices=1,
            ring_len=ring_len_for_window(1 + k),
        )
        self.k = k
        self.conv_shape = (ROLLBACK_POOL_ROWS, *regions[0].row_shape)
        self.rec_shape = (ROLLBACK_POOL_ROWS, *regions[1].row_shape)
        self.ring_shape = (ROLLBACK_POOL_ROWS, *regions[2].row_shape)
        # A layer keeps values from the graph it was first called in, so
        # each graph gets its own copy of the same weights.
        target, block = _seeded_block(config)
        base_target, base_block = _seeded_block(config)

        gpu, cpu = DeviceRef.GPU(), DeviceRef.CPU()
        dtype = config.state_dtype
        x_type = TensorType(
            DType.float32, ["total_seq_len", HIDDEN], device=gpu
        )
        offsets_type = TensorType(DType.uint32, ["offsets_len"], device=gpu)
        rows = TensorType(DType.uint32, [1, "batch_size"], device=gpu)
        pools = [
            BufferType(dtype, list(self.conv_shape), device=gpu),
            BufferType(dtype, list(self.rec_shape), device=gpu),
        ]
        signals_types = Signals([gpu]).input_types()

        # The rollback's plans rebind to ``batch_size``, so the verify of the
        # decode rows alone is a second run of this graph, not a second call.
        verify_types: list[TensorType | BufferType] = [
            x_type,
            offsets_type,
            TensorType(DType.uint32, ["offsets_len"], device=cpu),
            TensorType(DType.int64, ["batch_size"], device=gpu),
            rows,
            rows,
            *signals_types,
            *pools,
            BufferType(DType.float32, list(self.ring_shape), device=gpu),
        ]
        with Graph("mixed_step_rollback", input_types=verify_types) as verify:
            x, offsets, _host_offsets, accepted, conv_rows, rec_rows = (
                v.tensor for v in verify.inputs[:6]
            )
            signals, conv, rec, ring = (v.buffer for v in verify.inputs[6:])
            ring_rows = ops.range(
                start=0,
                stop=Dim("batch_size"),
                out_dim="batch_size",
                device=gpu,
                dtype=DType.uint32,
            ).reshape([1, "batch_size"])
            state = Qwen3_5RecurrentState(target, regions)
            leaves = [
                RecurrentStateInputsPerDevice(
                    leaves=(
                        RecurrentLeafInputs(pool=conv, live_row_ids=conv_rows),
                        RecurrentLeafInputs(pool=rec, live_row_ids=rec_rows),
                        RecurrentLeafInputs(pool=ring, live_row_ids=ring_rows),
                    )
                )
            ]
            with state.capturing(1, Dim(k)):
                out = block(
                    [x],
                    [signals],
                    layer_state_access(leaves, 0),
                    [offsets],
                )
            state.roll_forward(
                leaves,
                merged_offsets=offsets,
                num_accepted=accepted,
                num_draft_tokens=ops.constant(k, DType.int64, device=gpu),
                signal_buffers=[signals],
                device=gpu,
            )
            verify.output(out[0])

        base_types: list[TensorType | BufferType] = [
            x_type,
            offsets_type,
            rows,
            rows,
            *signals_types,
            *pools,
        ]
        with Graph("base_forward", input_types=base_types) as base:
            x, offsets, conv_rows, rec_rows = (
                v.tensor for v in base.inputs[:4]
            )
            signals, conv, rec = (v.buffer for v in base.inputs[4:])
            leaves = [
                RecurrentStateInputsPerDevice(
                    leaves=(
                        RecurrentLeafInputs(pool=conv, live_row_ids=conv_rows),
                        RecurrentLeafInputs(pool=rec, live_row_ids=rec_rows),
                    )
                )
            ]
            base_out = base_block(
                [x], [signals], layer_state_access(leaves, 0), [offsets]
            )
            base.output(base_out[0])

        self.device = Accelerator()
        session = InferenceSession(devices=[CPU(), self.device])
        self.verify = session.load(verify, weights_registry=target.state_dict())
        self.base = session.load(
            base, weights_registry=base_target.state_dict()
        )
        self.signal_buffers = Signals.allocate([self.device])

    def _buf(self, values: np.ndarray) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(values)).to(self.device)

    def _ring(self, rng: np.random.Generator) -> Buffer:
        # Stale records a wrong fold would read.
        return self._buf(
            rng.standard_normal(self.ring_shape).astype(np.float32)
        )

    def run(self, lengths: list[int], accepted: list[int]) -> _RollbackOutputs:
        """Verifies ``lengths`` at width ``k`` and rolls back to ``accepted``.

        On a copy of the pools, the long rows' real tokens then run the base
        forward and the other rows a verify and rollback of their own.
        """
        k = self.k
        rng = np.random.default_rng(13)
        x = rng.standard_normal((sum(lengths), HIDDEN)).astype(np.float32)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)
        batch = len(lengths)
        conv_rows = np.array([[2 * b + 1 for b in range(batch)]], np.uint32)
        rec_rows = np.array(
            [[ROLLBACK_POOL_ROWS - 1 - b for b in range(batch)]], np.uint32
        )
        conv = rng.standard_normal(self.conv_shape).astype(np.float32)
        rec = (0.1 * rng.standard_normal(self.rec_shape)).astype(np.float32)
        spec_pools = [self._buf(conv), self._buf(rec)]
        base_pools = [self._buf(conv), self._buf(rec)]

        (spec_out,) = self.verify.execute(
            self._buf(x),
            self._buf(offsets),
            Buffer.from_numpy(offsets),
            self._buf(np.array(accepted, np.int64)),
            self._buf(conv_rows),
            self._buf(rec_rows),
            *self.signal_buffers,
            *spec_pools,
            self._ring(rng),
        )

        long = [b for b in range(batch) if lengths[b] > k + 1]
        short = [b for b in range(batch) if lengths[b] <= k + 1]
        prompt_tokens = np.concatenate(
            [np.arange(offsets[b], offsets[b + 1] - k) for b in long]
        )
        decode_tokens = np.concatenate(
            [np.arange(offsets[b], offsets[b + 1]) for b in short]
        )
        prompt_offsets = np.concatenate(
            [[0], np.cumsum([lengths[b] - k for b in long])]
        ).astype(np.uint32)
        decode_offsets = np.concatenate(
            [[0], np.cumsum([lengths[b] for b in short])]
        ).astype(np.uint32)
        (prompt_out,) = self.base.execute(
            self._buf(x[prompt_tokens]),
            self._buf(prompt_offsets),
            self._buf(conv_rows[:, long]),
            self._buf(rec_rows[:, long]),
            *self.signal_buffers,
            *base_pools,
        )
        (decode_out,) = self.verify.execute(
            self._buf(x[decode_tokens]),
            self._buf(decode_offsets),
            Buffer.from_numpy(decode_offsets),
            self._buf(np.array([accepted[b] for b in short], np.int64)),
            self._buf(conv_rows[:, short]),
            self._buf(rec_rows[:, short]),
            *self.signal_buffers,
            *base_pools,
            self._ring(rng),
        )
        return _RollbackOutputs(
            prompt_rows=len(prompt_tokens),
            readout=_host(spec_out)[
                np.concatenate([prompt_tokens, decode_tokens])
            ],
            base_readout=np.concatenate([_host(prompt_out), _host(decode_out)]),
            conv=_host(spec_pools[0]),
            base_conv=_host(base_pools[0]),
            recurrent=_host(spec_pools[1]),
            base_recurrent=_host(base_pools[1]),
        )


@functools.cache
def _rollback_harness(k: int) -> _RollbackHarness:
    return _RollbackHarness(k)


@pytest.mark.parametrize(
    ("k", "lengths", "accepted"),
    [
        # MTP: prompt rows first and last, and a 1-token prompt row the
        # window holds; a prompt row's drafts are never accepted.
        pytest.param(
            3,
            [300 + 3, 4, 4, 4, 40 + 3],
            [0, 1, 3, 0, 0],
            id="mtp-prompt-rows-first-and-last",
        ),
        pytest.param(
            3,
            [4, 200 + 3, 4],
            [2, 0, 3],
            id="mtp-prompt-row-in-the-middle",
        ),
        # A row one past the window: the ring skips it, so only the chunk
        # op's own row rule can keep it.
        pytest.param(
            3,
            [5, 200 + 3, 4],
            [2, 0, 3],
            id="row-one-past-the-window",
        ),
        # DFlash2's block width.
        pytest.param(
            7,
            [100 + 7, 8, 8, 8, 30 + 7],
            [0, 3, 7, 0, 0],
            id="dflash2-block-width-prompt-rows-first-and-last",
        ),
        pytest.param(
            7,
            [8, 60 + 7, 8],
            [5, 0, 0],
            id="dflash2-block-width-prompt-row-in-the-middle",
        ),
        # Ring 8 at K = 5: a 6-token row fits the window, 7 and 8 do not.
        pytest.param(
            5,
            [100 + 5, 6, 7, 8, 30 + 5],
            [0, 2, 0, 0, 0],
            id="k5-rows-at-and-past-the-window",
        ),
        # A step verifying fewer drafts than the export's K: a row of 7
        # is long at the step's window of 6.
        pytest.param(
            5,
            [7, 60 + 5, 6],
            [0, 0, 4],
            id="fewer-drafts-than-the-export-k",
        ),
    ],
)
def test_a_rolled_back_mixed_verify_matches_its_rows_run_apart(
    k: int, lengths: list[int], accepted: list[int]
) -> None:
    out = _rollback_harness(k).run(lengths, accepted)
    np.testing.assert_array_equal(out.conv, out.base_conv)
    np.testing.assert_array_equal(out.recurrent, out.base_recurrent)
    n = out.prompt_rows
    np.testing.assert_array_equal(out.readout[:n], out.base_readout[:n])
    np.testing.assert_array_equal(out.readout[n:], out.base_readout[n:])
