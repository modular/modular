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

"""Pins one recurrent state row per linear-attention layer.

Sharing a row is invisible from outside -- the graph's declared inputs are
unchanged, so it loads, runs, and returns fluent nonsense -- so the check is
that each layer asks for a *different* row. The graph is built with subgraphs
on and every ``RecurrentLeafInputs.live_row_id`` call is recorded, which
couples this test to that helper being the one way a row is selected.
"""

from __future__ import annotations

import numpy as np
import pytest
from max import tree
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Graph, TensorValue
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    MHAKVCacheParams,
    MultiKVCacheParams,
    RecurrentLeafInputs,
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

HIDDEN = 64
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
VOCAB = 128

# A full-attention layer among the linear ones, as in the real model: the
# linear layers are one subgraph group and their state rows are numbered
# within it, not by absolute layer index.
LAYER_TYPES = [
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
]
NUM_LINEAR = sum(1 for lt in LAYER_TYPES if lt == "linear_attention")

LINEAR_KEY_HEAD_DIM = 8
LINEAR_VALUE_HEAD_DIM = 8
LINEAR_NUM_KEY_HEADS = 2
LINEAR_NUM_VALUE_HEADS = 4
LINEAR_CONV_KERNEL_DIM = 4


def _config(*, use_subgraphs: bool) -> Qwen3_5Config:
    device = DeviceRef.CPU()
    kv_params = MHAKVCacheParams(
        dtype=DType.float32,
        devices=[device],
        n_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
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
        linear_key_head_dim=LINEAR_KEY_HEAD_DIM,
        linear_value_head_dim=LINEAR_VALUE_HEAD_DIM,
        linear_num_key_heads=LINEAR_NUM_KEY_HEADS,
        linear_num_value_heads=LINEAR_NUM_VALUE_HEADS,
        linear_conv_kernel_dim=LINEAR_CONV_KERNEL_DIM,
        partial_rotary_factor=1.0,
        use_subgraphs=use_subgraphs,
    )


def _cache_params(config: Qwen3_5Config) -> MultiKVCacheParams:
    """The attention leaf plus the recurrent state child, as the model sees it."""
    attn = attn_cache(config.kv_params)
    state = RecurrentStateParams(
        regions=linear_state_regions(
            num_linear_layers=NUM_LINEAR,
            key_head_dim=LINEAR_KEY_HEAD_DIM,
            num_key_heads=LINEAR_NUM_KEY_HEADS,
            value_head_dim=LINEAR_VALUE_HEAD_DIM,
            num_value_heads=LINEAR_NUM_VALUE_HEADS,
            conv_kernel_dim=LINEAR_CONV_KERNEL_DIM,
            dtype=DType.float32,
            num_devices=1,
        ),
        devices=attn.devices,
        data_parallel_degree=attn.data_parallel_degree,
    )
    return MultiKVCacheParams.from_params(
        {ATTN_CACHE_KEY: attn, STATE_CACHE_KEY: state}
    )


def _rows_selected_building(
    config: Qwen3_5Config, monkeypatch: pytest.MonkeyPatch
) -> list[int]:
    """Builds the graph and returns the row column each state access asked for."""
    model = Qwen3_5(config)
    # Weights are named through the module tree, so they have to be registered
    # before the graph is built or every layer contributes a bare 'weight'.
    rng = np.random.default_rng(0)
    model.load_state_dict(
        {
            name: rng.standard_normal(
                [int(d) for d in w.shape], dtype=np.float32
            )
            * 0.05
            for name, w in model.raw_state_dict().items()
        },
        weight_alignment=1,
        strict=False,
        override_quantization_encoding=True,
    )
    cache_params = _cache_params(config)

    selected: list[int] = []
    original = RecurrentLeafInputs.live_row_id

    def record(
        self: RecurrentLeafInputs[TensorValue, BufferValue], layer: int
    ) -> TensorValue:
        selected.append(layer)
        return original(self, layer)

    monkeypatch.setattr(RecurrentLeafInputs, "live_row_id", record)

    with Graph(
        "qwen3_5_state_rows", input_types=model.input_types(cache_params)
    ) as graph:
        tokens, input_row_offsets, return_n_logits, *variadic = graph.inputs
        signal_buffers = [v.buffer for v in variadic[:1]]
        kv_tree = cache_params.unflatten_kv_inputs(iter(variadic[1:]))
        assert isinstance(kv_tree, dict)
        kv_collections = tree.leaves(
            kv_tree[ATTN_CACHE_KEY], leaf=KVCacheInputsPerDevice
        )
        state = tree.leaves(
            kv_tree[STATE_CACHE_KEY], leaf=RecurrentStateInputsPerDevice
        )
        outputs = model(
            tokens.tensor,
            kv_collections,
            return_n_logits.tensor,
            input_row_offsets.tensor,
            signal_buffers,
            list(state),
        )
        graph.output(*outputs)

    return selected


@pytest.mark.parametrize("use_subgraphs", [True, False])
def test_each_linear_layer_reads_its_own_state_row(
    use_subgraphs: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each linear layer selects its own column, subgraphs on or off.

    With subgraphs on and the selection made inside the block, this records a
    single column zero -- the body is built once -- instead of one column per
    layer per leaf.
    """
    selected = _rows_selected_building(
        _config(use_subgraphs=use_subgraphs), monkeypatch
    )

    # Two leaves (conv, then recurrent) per linear layer, in layer order.
    expected = [layer for layer in range(NUM_LINEAR) for _ in range(2)]
    assert selected == expected, (
        f"expected each of {NUM_LINEAR} layers to select its own row twice"
    )
