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

"""A Qwen3.5 graph hands its KV collections' mixed-step split records to the
ragged MHA of every full-attention layer.

The records reach the attention through the full-attention block's own
arguments (the layers share one subgraph), so a record dropped on the way
leaves every layer on a zeros record: the graph still builds and runs, only
slower. The check reads the record operand of each ragged MHA op.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from max import tree
from max.dtype import DType
from max.graph import DeviceRef, Graph
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    MHAKVCacheParams,
    MultiKVCacheParams,
    RecurrentStateInputsPerDevice,
    RecurrentStateParams,
)
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.qwen3_5.qwen3_5 import Qwen3_5
from max.pipelines.architectures.qwen3_5.state_cache import (
    ATTN_CACHE_KEY,
    STATE_CACHE_KEY,
    attn_cache,
    linear_state_regions,
)
from test_common.mixed_step_split_ops import record_sources

MHA_OP = "attention.mha.ragged.paged"
HIDDEN = 64
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
VOCAB = 128
LAYER_TYPES = ["linear_attention", "full_attention", "full_attention"]
LINEAR_HEAD_DIM = 8
LINEAR_KEY_HEADS = 2
LINEAR_VALUE_HEADS = 4
CONV_KERNEL = 4


def _config(use_subgraphs: bool) -> Qwen3_5Config:
    device = DeviceRef.CPU()
    kv_params = MHAKVCacheParams(
        dtype=DType.float32,
        devices=[device],
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=2,
        page_size=HEAD_DIM,
    )
    return Qwen3_5Config(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_hidden_layers=len(LAYER_TYPES),
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=HIDDEN * 2,
        interleaved_rope_weights=True,
        vocab_size=VOCAB,
        dtype=DType.float32,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=kv_params,
        norm_dtype=DType.float32,
        rms_norm_eps=1e-6,
        attention_multiplier=float(HEAD_DIM) ** -0.5,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        layer_types=list(LAYER_TYPES),
        linear_key_head_dim=LINEAR_HEAD_DIM,
        linear_value_head_dim=LINEAR_HEAD_DIM,
        linear_num_key_heads=LINEAR_KEY_HEADS,
        linear_num_value_heads=LINEAR_VALUE_HEADS,
        linear_conv_kernel_dim=CONV_KERNEL,
        partial_rotary_factor=1.0,
        use_subgraphs=use_subgraphs,
    )


def _record_operands(use_subgraphs: bool, with_records: bool) -> list[str]:
    config = _config(use_subgraphs)
    model = Qwen3_5(config)
    rng = np.random.default_rng(0)
    model.load_state_dict(
        {
            name: rng.standard_normal([int(d) for d in w.shape]).astype(
                np.float32
            )
            for name, w in model.raw_state_dict().items()
        },
        weight_alignment=1,
        strict=False,
        override_quantization_encoding=True,
    )
    attn = attn_cache(config.kv_params)
    state_params = RecurrentStateParams(
        regions=linear_state_regions(
            num_linear_layers=1,
            key_head_dim=LINEAR_HEAD_DIM,
            num_key_heads=LINEAR_KEY_HEADS,
            value_head_dim=LINEAR_HEAD_DIM,
            num_value_heads=LINEAR_VALUE_HEADS,
            conv_kernel_dim=CONV_KERNEL,
            dtype=DType.float32,
            num_devices=1,
        ),
        devices=attn.devices,
        data_parallel_degree=attn.data_parallel_degree,
    )
    cache_params = MultiKVCacheParams.from_params(
        {ATTN_CACHE_KEY: attn, STATE_CACHE_KEY: state_params}
    )
    with Graph(
        "qwen3_5_mixed_step_split", input_types=model.input_types(cache_params)
    ) as graph:
        tokens, row_offsets, return_n_logits, *variadic = graph.inputs
        kv_tree = cache_params.unflatten_kv_inputs(iter(variadic[1:]))
        assert isinstance(kv_tree, dict)
        kv_collections = tree.leaves(
            kv_tree[ATTN_CACHE_KEY], leaf=KVCacheInputsPerDevice
        )
        if not with_records:
            kv_collections = [
                replace(kv, mixed_step_split=None) for kv in kv_collections
            ]
        state = tree.leaves(
            kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
        )
        graph.output(
            model(
                tokens.tensor,
                kv_collections,
                return_n_logits.tensor,
                row_offsets.tensor,
                [variadic[0].buffer],
                list(state),
            )
        )
    # The module holds the subgraph bodies too; the graph holds only its calls.
    return [
        source
        for symbol, source in record_sources(graph._mlir_op.block.owner)
        if symbol == MHA_OP
    ]


@pytest.mark.parametrize("use_subgraphs", [True, False])
def test_records_on_the_kv_collections_reach_every_attention_layer(
    use_subgraphs: bool,
) -> None:
    sources = _record_operands(use_subgraphs, with_records=True)
    # Subgraphs build the shared block once; without them each layer is inlined.
    expected = 1 if use_subgraphs else LAYER_TYPES.count("full_attention")
    assert sources == ["argument"] * expected


@pytest.mark.parametrize("use_subgraphs", [True, False])
def test_without_records_the_attention_gets_zeros(
    use_subgraphs: bool,
) -> None:
    sources = _record_operands(use_subgraphs, with_records=False)
    assert sources
    assert set(sources) == {"mo.constant"}
