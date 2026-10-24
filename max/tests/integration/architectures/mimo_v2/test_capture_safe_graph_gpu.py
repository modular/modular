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
"""Tests that the MiMo-V2 graph stays capturable across devices.

The graph is only built, never compiled; it needs a GPU because the model's
allreduce binds its accelerators when it is constructed.
"""

from __future__ import annotations

import re

import pytest
from max import tree
from max.driver import accelerator_count
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.comm import Signals
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.mimo_v2.mimo_v2 import MiMoV2
from max.pipelines.architectures.mimo_v2.model_config import (
    FULL,
    SLIDING,
    MiMoV2Config,
    attention_head_dim,
    layer_types,
)
from transformers.configuration_utils import PretrainedConfig

LAYERS = 3
HIDDEN = 128
EXPERTS = 2


def _config() -> PretrainedConfig:
    """A 3-layer MiMo-V2: full + dense MLP, sliding + MoE, full + MoE."""
    return PretrainedConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        layernorm_epsilon=1e-6,
        hybrid_layer_pattern=[0, 1, 0],
        moe_layer_freq=[0, 1, 1],
        num_attention_heads=8,
        swa_num_attention_heads=8,
        num_key_value_heads=4,
        swa_num_key_value_heads=8,
        head_dim=192,
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
        partial_rotary_factor=0.334,
        rope_theta=10000000.0,
        swa_rope_theta=10000.0,
        sliding_window=128,
        attention_value_scale=0.707,
        add_full_attention_sink_bias=False,
        add_swa_attention_sink_bias=True,
        intermediate_size=256,
        moe_intermediate_size=256,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=2,
        n_shared_experts=None,
        norm_topk_prob=True,
        routed_scaling_factor=None,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=False,
        attention_projection_layout="fused_qkv",
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "store_dtype": "mxfp4",
            "mxfp4_block_size": 32,
        },
    )


def _build(
    num_devices: int,
    *,
    query_heads: int = 8,
    fused_sliding_qkv: bool = False,
    fused_full_qkv: bool = False,
    kv_dtype: DType = DType.bfloat16,
    full_kv_dtype: DType = DType.bfloat16,
) -> Graph:
    """Builds the graph over ``num_devices`` GPUs."""
    hf = _config()
    hf.num_attention_heads = query_heads
    hf.swa_num_attention_heads = query_heads
    devices = [DeviceRef.GPU(i) for i in range(num_devices)]
    groups = layer_types(hf)
    kv_params = MultiKVCacheParams.from_params(
        {
            SLIDING: MHAKVCacheParams(
                dtype=kv_dtype,
                n_kv_heads=hf.swa_num_key_value_heads,
                head_dim=attention_head_dim(hf),
                num_layers=groups.count(SLIDING),
                devices=devices,
                window_size=hf.sliding_window,
            ),
            FULL: MHAKVCacheParams(
                dtype=full_kv_dtype,
                n_kv_heads=hf.num_key_value_heads,
                head_dim=attention_head_dim(hf),
                num_layers=groups.count(FULL),
                devices=devices,
            ),
        }
    )
    config = MiMoV2Config.from_huggingface_config(
        hf, devices=devices, kv_params=kv_params, max_seq_len=4096
    )
    config.fused_sliding_qkv = fused_sliding_qkv
    config.fused_full_qkv = fused_full_qkv
    model = MiMoV2(config)
    # Names every weight by its path, as loading a checkpoint does.
    model.state_dict()
    input_types = [
        TensorType(DType.int64, ["total_seq_len"], device=devices[0]),
        TensorType(DType.int64, ["return_n_logits"], device=DeviceRef.CPU()),
        *(
            TensorType(DType.uint32, ["input_row_offsets_len"], device=d)
            for d in devices
        ),
        *Signals(devices=devices).input_types(),
        *tree.leaves(kv_params.get_symbolic_inputs()),
    ]
    with Graph("mimo_v2", input_types=input_types) as graph:
        tokens, return_n_logits, *rest = graph.inputs
        n = num_devices
        sliding, full = kv_params.unflatten_basic_kv_tree(iter(rest[2 * n :]))
        outputs = model(
            tokens=tokens.tensor,
            signal_buffers=[v.buffer for v in rest[n : 2 * n]],
            sliding_kv_collections=sliding,
            full_kv_collections=full,
            return_n_logits=return_n_logits.tensor,
            input_row_offsets=[v.tensor for v in rest[:n]],
        )
        graph.output(outputs)
    return graph


@pytest.mark.skipif(accelerator_count() < 2, reason="needs 2 GPUs")
def test_no_activation_crosses_devices_outside_a_collective() -> None:
    # Device graph capture records one graph per device stream, and a
    # GPU-to-GPU transfer makes one stream wait on another's, which CUDA
    # refuses to capture. The embedding reaches the other devices through
    # a signal-buffer broadcast instead.
    ir = str(_build(2))
    gpu_to_gpu = re.findall(
        r"mo\.transfer\[[^\n]*, gpu:\d+> to <\"gpu\", \d+>", ir
    )
    assert not gpu_to_gpu, gpu_to_gpu
    assert re.search(r"mo\.distributed\.broadcast", ir), ir[:2000]


@pytest.mark.skipif(accelerator_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("query_heads", [16, 64])
def test_sliding_qkv_fusion_is_opt_in(query_heads: int) -> None:
    default = _build(2, query_heads=query_heads)
    # MoE attention lives in shared callees outside the root graph.
    default_ir = default.module.mlir_module.asm()
    assert "mo.chunked_qkv_rope_store.ragged.paged" not in default_ir
    assert default_ir.count("mo.fused_qk_rope.ragged.paged") == 6

    fused = _build(2, query_heads=query_heads, fused_sliding_qkv=True)
    fused_ir = fused.module.mlir_module.asm()
    assert fused_ir.count("mo.chunked_qkv_rope_store.ragged.paged") == 2
    assert fused_ir.count("mo.fused_qk_rope.ragged.paged") == 4
    assert [str(value.type) for value in default.inputs] == [
        str(value.type) for value in fused.inputs
    ]
    assert [str(value) for value in default.output_types] == [
        str(value) for value in fused.output_types
    ]


@pytest.mark.skipif(accelerator_count() < 2, reason="needs 2 GPUs")
def test_enabled_sliding_qkv_rejects_ineligible_cache() -> None:
    with pytest.raises(ValueError, match=r"rank-6 BF16 cache.*float32"):
        _build(2, fused_sliding_qkv=True, kv_dtype=DType.float32)


@pytest.mark.skipif(accelerator_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("query_heads", [8, 16, 64])
@pytest.mark.parametrize("sliding", [False, True])
def test_full_qkv_fusion_is_independent(
    query_heads: int, sliding: bool
) -> None:
    default = _build(2, query_heads=query_heads)
    fused = _build(
        2,
        query_heads=query_heads,
        fused_sliding_qkv=sliding,
        fused_full_qkv=True,
    )
    ir = fused.module.mlir_module.asm()
    assert ir.count("mo.chunked_qkv_rope_store.ragged.paged") == (
        6 if sliding else 4
    )
    assert ir.count("mo.fused_qk_rope.ragged.paged") == (0 if sliding else 2)
    assert [str(value.type) for value in default.inputs] == [
        str(value.type) for value in fused.inputs
    ]
    assert [str(value) for value in default.output_types] == [
        str(value) for value in fused.output_types
    ]


@pytest.mark.skipif(accelerator_count() < 2, reason="needs 2 GPUs")
def test_enabled_full_qkv_rejects_ineligible_cache() -> None:
    with pytest.raises(ValueError, match=r"rank-6 BF16 cache.*float32"):
        _build(2, fused_full_qkv=True, full_kv_dtype=DType.float32)
