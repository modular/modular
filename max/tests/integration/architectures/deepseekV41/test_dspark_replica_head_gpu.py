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
"""The per-replica DSpark head drafts what the device-0 head drafts.

Runs the head on random norm, Markov and LM head weights both ways at DP2
and requires identical draft tokens and distributions, argmax and sampled,
including with an empty replica.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Type, ops
from max.nn.legacy import Signals
from max.pipelines.architectures.deepseekV41.dspark_drafter import (
    DeepseekV41DSparkDrafter,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
)
from max.pipelines.architectures.deepseekV41.spec_adapters import (
    _DSparkProposer,
)
from max.pipelines.speculative.block_driver import (
    Accepted,
    ArgmaxDraftSampler,
    BlockBatch,
    SampledDraftSampler,
)

K = 5
HIDDEN = 128
VOCAB = 256
DP = 2


def _config(devices: list[DeviceRef]) -> DeepseekV41Config:
    # The widths of test_dspark_weight_adapter.py, which build the stages.
    return DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=MagicMock(num_draft_tokens=0),
        devices=devices,
        max_seq_len=64,
        data_parallel_degree=DP,
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
        sliding_window=8,
        n_routed_experts=8,
        moe_intermediate_size=128,
        num_experts_per_tok=2,
        num_dspark_layers=1,
        dspark_target_layer_ids=[0],
        dspark_block_size=K,
        dspark_markov_rank=32,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
        compress_ratios=[1, 0],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        num_attention_heads=4,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=64,
        o_groups=2,
        o_lora_rank=128,
    )


def _random(rng: np.random.Generator, dtype: DType, shape: list[int]) -> Buffer:
    values = rng.standard_normal(shape, dtype=np.float32)
    if dtype == DType.float32:
        return Buffer.from_numpy(values)
    bits = (values.view(np.uint32) >> 16).astype(np.uint16)
    return Buffer.from_numpy(bits).view(DType.bfloat16, shape)


@pytest.mark.skipif(accelerator_count() < DP, reason="needs two GPUs")
def test_replica_head_matches_device0_head() -> None:
    devices = [DeviceRef.GPU(i) for i in range(DP)]
    config = _config(devices)
    drafter = DeepseekV41DSparkDrafter(config, devices[0])
    rng = np.random.default_rng(7)
    drafter.load_state_dict(
        {
            name: _random(rng, w.dtype, [int(d) for d in w.shape])
            for name, w in drafter.raw_state_dict().items()
            if name == "norm.weight" or name.startswith("markov_head.")
        },
        strict=False,
    )
    weights = drafter.state_dict()

    signals = Signals(devices)
    dev0 = devices[0]
    input_types = cast(
        list[Type[Any]],
        [
            *(
                TensorType(DType.bfloat16, [1, f"hs_rows_{i}", HIDDEN], d)
                for i, d in enumerate(devices)
            ),
            *(TensorType(DType.bfloat16, [VOCAB, HIDDEN], d) for d in devices),
            TensorType(DType.int64, ["batch_size"], dev0),
            TensorType(DType.uint64, ["batch_size"], dev0),
            TensorType(DType.float32, ["batch_size"], dev0),
            TensorType(DType.int64, ["batch_size"], dev0),
            TensorType(DType.float32, ["batch_size"], dev0),
            TensorType(DType.int64, [DP + 1], DeviceRef.CPU()),
            *signals.input_types(),
        ],
    )
    with Graph("dspark_replica_head", input_types=input_types) as g:
        hs = [v.tensor for v in g.inputs[:DP]]
        heads = [v.tensor for v in g.inputs[DP : 2 * DP]]
        prev, seed, temperature, top_k, top_p, splits = (
            v.tensor for v in g.inputs[2 * DP : 2 * DP + 6]
        )
        signal_buffers = [v.buffer for v in g.inputs[2 * DP + 6 :]]
        trunks = [
            SimpleNamespace(head=SimpleNamespace(weight=w)) for w in heads
        ]
        verify = SimpleNamespace(config=config, replicas=lambda _: trunks)
        proposer = _DSparkProposer(
            cast(Any, verify),
            drafter,
            SimpleNamespace(block_size=K, mask_token_id=0),
        )
        batch = cast(
            BlockBatch,
            SimpleNamespace(
                devices=devices,
                n_devs=DP,
                signal_buffers=signal_buffers,
                data_parallel_splits=splits,
                data_parallel_degree=DP,
                replica_of=list(range(DP)),
            ),
        )
        accepted = Accepted(
            num_accepted=prev,
            next_tokens=prev,
            commit_lengths=prev,
            is_prefill=prev,
        )
        # Each device takes its blocks' anchors from the block ids.
        mask = ops.broadcast_to(
            ops.constant(0, DType.int64, dev0), ["batch_size", K - 1]
        )
        proposer.embed_block(
            batch, ops.concat([ops.unsqueeze(prev, 1), mask], 1).reshape([-1])
        )
        # What the device-0 head reads: every replica's rows on device 0.
        all_hs = ops.allgather(
            [ops.reshape(h, [-1, HIDDEN]) for h in hs], signal_buffers
        )[0].reshape([1, -1, HIDDEN])

        def sampler() -> SampledDraftSampler:
            return SampledDraftSampler(
                seed=seed,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                vocab_size=VOCAB,
            )

        def drafts(block_hs: TensorValue | list[TensorValue]) -> list[Any]:
            sampled = sampler()
            return [
                proposer.head(batch, block_hs, accepted, ArgmaxDraftSampler()),
                proposer.head(batch, block_hs, accepted, sampled),
                sampled.distributions(),
            ]

        g.output(*drafts(all_hs), *drafts(hs))

    accelerators = [Accelerator(i) for i in range(DP)]
    session = InferenceSession(devices=[CPU(), *accelerators])
    model = session.load(g, weights_registry=weights)
    head = _random(rng, DType.bfloat16, [VOCAB, HIDDEN])
    for split in ([0, 3, 3], [0, 2, 5]):
        batch_size = split[-1]
        block_hs = [
            _random(rng, DType.bfloat16, [1, (hi - lo) * K, HIDDEN]).to(acc)
            for lo, hi, acc in zip(
                split[:-1], split[1:], accelerators, strict=True
            )
        ]
        params: list[npt.NDArray[Any]] = [
            rng.integers(0, VOCAB, batch_size, dtype=np.int64),
            rng.integers(0, 2**63, batch_size, dtype=np.uint64),
            np.full(batch_size, 1.0, np.float32),
            np.full(batch_size, -1, np.int64),
            np.full(batch_size, 0.95, np.float32),
        ]
        outputs = model.execute(
            *block_hs,
            *(head.to(acc) for acc in accelerators),
            *(Buffer.from_numpy(p).to(accelerators[0]) for p in params),
            Buffer.from_numpy(np.array(split, np.int64)),
            *signals.buffers(),
        )
        ref_argmax, ref_tokens, ref_dists, argmax, tokens, dists = (
            cast(Buffer, o).to(CPU()).to_numpy() for o in outputs
        )
        assert ref_argmax.shape == (batch_size, K)
        np.testing.assert_array_equal(argmax, ref_argmax)
        np.testing.assert_array_equal(tokens, ref_tokens)
        assert ref_dists.shape == (batch_size, K, VOCAB)
        np.testing.assert_array_equal(dists, ref_dists)
        # Not vacuous: the sampled chain left the argmax one somewhere.
        assert not np.array_equal(ref_tokens, ref_argmax)
