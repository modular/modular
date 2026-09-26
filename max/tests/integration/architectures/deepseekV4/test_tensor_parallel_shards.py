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
"""DeepSeek-V4 tensor-parallel shard layout, checked on graph-free metadata.

The TP=2 numerics are gated on B200s; this pins the layout those gates
certified (which weights split along which axis, which stay whole, and that
the fp8 linears keep their class) so a change to it fails without a GPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, Weight
from max.nn import Linear
from max.pipelines.architectures.deepseekV4.deepseekV4 import (
    DeepseekV4,
    DeepseekV4Block,
)
from max.pipelines.architectures.deepseekV4.layers.moe import (
    DeepseekV4RoutedExperts,
)
from max.pipelines.architectures.deepseekV4.layers.quantization import (
    DeepseekV4Fp8Linear,
    fp8_block_quant_config,
)
from max.pipelines.architectures.deepseekV4.model_config import (
    DeepseekV4Config,
)

_N_DEV = 2
_HEADS = 4
_O_GROUPS = 2
_INDEX_HEADS = 4
_EXPERTS = 4
_HIDDEN = 256
_HEAD_DIM = 128
# Layer 0 is window-only, layer 1 has the compressor and indexer (ratio 4).
_COMPRESS_RATIOS = [0, 4]


def _config() -> DeepseekV4Config:
    quant_config = fp8_block_quant_config(
        {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
        len(_COMPRESS_RATIOS),
    )
    return DeepseekV4Config(
        dtype=DType.bfloat16,
        # The cache layout plays no part in how weights shard.
        kv_params=Mock(),
        devices=[DeviceRef.CPU()],
        max_seq_len=256,
        vocab_size=256,
        hidden_size=_HIDDEN,
        num_hidden_layers=len(_COMPRESS_RATIOS),
        num_attention_heads=_HEADS,
        head_dim=_HEAD_DIM,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=_O_GROUPS,
        sliding_window=128,
        compress_ratios=_COMPRESS_RATIOS,
        rope_scaling={
            "factor": 16.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 65536,
        },
        index_head_dim=128,
        index_n_heads=_INDEX_HEADS,
        index_topk=16,
        moe_intermediate_size=128,
        n_routed_experts=_EXPERTS,
        num_experts_per_tok=2,
        num_hash_layers=0,
        hc_mult=2,
        quant_config=quant_config,
        dspark_stages=False,
    )


def _shape(weight: Weight) -> tuple[int, ...]:
    return tuple(int(d) for d in weight.shape)


@pytest.fixture(scope="module")
def models() -> Iterator[tuple[DeepseekV4, list[DeepseekV4]]]:
    source = DeepseekV4(_config(), DeviceRef.CPU())
    # Loading names every weight by its path, which the shards' graph values
    # need to stay distinct; the values themselves are never read.
    source.load_state_dict(source.state_dict(auto_initialize=True))
    devices = [DeviceRef.GPU(i) for i in range(_N_DEV)]
    replicas = source.tensor_parallel_replicas(devices)
    # A shard's shape is a slice of its parent's value, which needs a graph;
    # nothing is built into it.
    with Graph("deepseekV4_tp_shards"):
        yield source, replicas


def _blocks(model: DeepseekV4) -> list[DeepseekV4Block]:
    blocks = []
    for layer in model.layers:
        assert isinstance(layer, DeepseekV4Block)
        blocks.append(layer)
    return blocks


def test_attention_splits_heads(
    models: tuple[DeepseekV4, list[DeepseekV4]],
) -> None:
    source, replicas = models
    for rank, replica in enumerate(replicas):
        for src, rep in zip(_blocks(source), _blocks(replica), strict=True):
            s, r = src.attn, rep.attn
            assert r.n_heads == _HEADS // _N_DEV
            assert r.o_groups == _O_GROUPS // _N_DEV
            out, inp = _shape(s.wq_b.weight)
            assert _shape(r.wq_b.weight) == (out // _N_DEV, inp)
            out, inp = _shape(s.wo_a.weight)
            assert _shape(r.wo_a.weight) == (out // _N_DEV, inp)
            out, inp = _shape(s.wo_b.weight)
            assert _shape(r.wo_b.weight) == (out, inp // _N_DEV)
            assert _shape(r.attn_sink) == (_HEADS // _N_DEV,)
            # The latent side is shared by every head.
            assert _shape(r.wq_a.weight) == _shape(s.wq_a.weight)
            assert _shape(r.wkv.weight) == _shape(s.wkv.weight)
            assert r.wq_b.weight.device == DeviceRef.GPU(rank)


def test_indexer_splits_heads(
    models: tuple[DeepseekV4, list[DeepseekV4]],
) -> None:
    source, replicas = models
    for replica in replicas:
        for src, rep in zip(_blocks(source), _blocks(replica), strict=True):
            if src.attn.indexer is None:
                assert rep.attn.indexer is None
                continue
            s, r = src.attn.indexer, rep.attn.indexer
            assert r is not None
            assert r.n_heads == _INDEX_HEADS // _N_DEV
            for name in ("wq_b", "weights_proj"):
                out, inp = _shape(getattr(s, name).weight)
                assert _shape(getattr(r, name).weight) == (out // _N_DEV, inp)


def test_routed_experts_split_whole_experts(
    models: tuple[DeepseekV4, list[DeepseekV4]],
) -> None:
    source, replicas = models
    local = _EXPERTS // _N_DEV
    for rank, replica in enumerate(replicas):
        for src, rep in zip(_blocks(source), _blocks(replica), strict=True):
            assert rep.ffn.local_experts == range(
                rank * local, (rank + 1) * local
            )
            s, r = src.ffn.experts, rep.ffn.experts
            assert isinstance(s, DeepseekV4RoutedExperts)
            assert isinstance(r, DeepseekV4RoutedExperts)
            assert r.n_experts == local
            assert r.expert_offset == rank * local
            for proj in ("gate_up_proj", "down_proj"):
                for name in ("weight", "weight_scale"):
                    full = _shape(getattr(getattr(s, proj), name))
                    part = _shape(getattr(getattr(r, proj), name))
                    assert part == (full[0] // _N_DEV, *full[1:])
            # The shared expert and the gate are replicated.
            for name in ("w1", "w2", "w3"):
                assert _shape(
                    getattr(rep.ffn.shared_experts, name).weight
                ) == _shape(getattr(src.ffn.shared_experts, name).weight)


def test_fp8_linears_keep_their_class(
    models: tuple[DeepseekV4, list[DeepseekV4]],
) -> None:
    source, replicas = models
    for replica in replicas:
        for src, rep in zip(_blocks(source), _blocks(replica), strict=True):
            for name in ("wq_a", "wq_b", "wkv", "wo_b"):
                assert isinstance(getattr(src.attn, name), DeepseekV4Fp8Linear)
                assert isinstance(getattr(rep.attn, name), DeepseekV4Fp8Linear)
                assert getattr(rep.attn, name).weight.dtype == (
                    DType.float8_e4m3fn
                )
            # ``wo_a`` is bf16 in the reference and host-dequantized.
            assert type(rep.attn.wo_a) is Linear


def test_fp8_linear_refuses_linear_shard() -> None:
    quant_config = _config().quant_config
    assert quant_config is not None
    linear = DeepseekV4Fp8Linear(128, 128, DeviceRef.CPU(), quant_config)
    with pytest.raises(TypeError):
        linear.shard([DeviceRef.GPU(0), DeviceRef.GPU(1)])
