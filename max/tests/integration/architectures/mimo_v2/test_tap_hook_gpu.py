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
"""Tests the MiMo-V2 graph's tail KV group and tap hook, which a speculative
drafter's context writer uses on the target's forward, and that the graph
stays capturable across devices.

The graph is only built, never compiled; it needs a GPU because the model's
allreduce binds its accelerators when it is constructed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import pytest
from max import tree
from max.driver import accelerator_count
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Graph, TensorType, TensorValue
from max.nn.comm import Signals
from max.nn.kv_cache import (
    MHAKVCacheParams,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.pipelines.architectures.mimo_v2.mimo_v2 import MiMoV2, TapHook
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
TAIL = "draft"


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
        conversion_metadata={"qkv_layout": "global_q_k_v"},
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "quantized_layers": {
                f"model.layers.{layer}.mlp.experts.{expert}.{proj}": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                }
                for layer in (1, 2)
                for expert in range(EXPERTS)
                for proj in ("gate_proj", "up_proj", "down_proj")
            },
        },
    )


class _Recorder:
    """A tap hook that records what the graph hands it."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        taps: list[list[TensorValue]],
        tail_kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> None:
        self.calls.append(
            {
                "taps": taps,
                "tail": list(tail_kv_collections),
                "offsets": list(input_row_offsets),
                "signals": list(signal_buffers),
            }
        )


def _build(
    num_devices: int,
    tap_hook: TapHook | None,
    with_tail: bool,
    target_layer_ids: list[int] | None = None,
) -> tuple[list[Any], list[PagedCacheValues] | None, Graph]:
    """Builds the graph; returns its outputs, the tail group it passed and
    the graph."""
    hf = _config()
    devices = [DeviceRef.GPU(i) for i in range(num_devices)]
    groups = layer_types(hf)
    params = {
        SLIDING: MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=hf.swa_num_key_value_heads,
            head_dim=attention_head_dim(hf),
            num_layers=groups.count(SLIDING),
            devices=devices,
            window_size=hf.sliding_window,
        ),
        FULL: MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=hf.num_key_value_heads,
            head_dim=attention_head_dim(hf),
            num_layers=groups.count(FULL),
            devices=devices,
        ),
    }
    if with_tail:
        # A drafter's geometry: its own head dim and layer count.
        params[TAIL] = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=5,
            devices=devices,
            window_size=1024,
        )
    kv_params = MultiKVCacheParams.from_params(params)
    config = MiMoV2Config.from_huggingface_config(
        hf,
        devices=devices,
        kv_params=MultiKVCacheParams.from_params(
            {group: params[group] for group in (SLIDING, FULL)}
        ),
        max_seq_len=4096,
    )
    config.target_layer_ids = target_layer_ids
    model = MiMoV2(config, tap_hook=tap_hook)
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
    with Graph("mimo_v2_tap_hook", input_types=input_types) as graph:
        tokens, return_n_logits, *rest = graph.inputs
        n = num_devices
        kv = kv_params.unflatten_basic_kv_tree(iter(rest[2 * n :]))
        tail = kv[2] if with_tail else None
        outputs = model(
            tokens=tokens.tensor,
            signal_buffers=[v.buffer for v in rest[n : 2 * n]],
            sliding_kv_collections=kv[0],
            full_kv_collections=kv[1],
            return_n_logits=return_n_logits.tensor,
            input_row_offsets=[v.tensor for v in rest[:n]],
            tail_kv_collections=tail,
        )
        graph.output(*outputs)
    return list(outputs), tail, graph


def _device_counts() -> list[int]:
    return [n for n in (1, 2) if n <= accelerator_count()]


@pytest.mark.parametrize("num_devices", _device_counts())
def test_hook_gets_each_target_layer_and_the_tail_group(
    num_devices: int,
) -> None:
    hook = _Recorder()
    outputs, tail, _ = _build(
        num_devices, hook, with_tail=True, target_layer_ids=[0, 2]
    )
    plain, _, _ = _build(num_devices, None, with_tail=False)

    (call,) = hook.calls
    assert call["tail"] == tail
    assert len(call["offsets"]) == len(call["signals"]) == num_devices
    assert len(call["taps"]) == 2
    for per_device in call["taps"]:
        assert [t.device for t in per_device] == [
            DeviceRef.GPU(i) for i in range(num_devices)
        ]
        assert all(list(t.shape)[1:] == [HIDDEN] for t in per_device)
    # The taps feed the hook, not the graph's outputs.
    assert len(outputs) == len(plain)


def test_a_tail_group_needs_a_hook_and_a_hook_needs_taps() -> None:
    with pytest.raises(ValueError, match="come together"):
        _build(1, None, with_tail=True)
    with pytest.raises(ValueError, match="come together"):
        _build(1, _Recorder(), with_tail=False, target_layer_ids=[0])
    with pytest.raises(ValueError, match="needs target_layer_ids"):
        _build(1, _Recorder(), with_tail=True)


@pytest.mark.skipif(accelerator_count() < 2, reason="needs 2 GPUs")
def test_no_activation_crosses_devices_outside_a_collective() -> None:
    # Device graph capture records one graph per device stream, and a
    # GPU-to-GPU transfer makes one stream wait on another's, which CUDA
    # refuses to capture. The embedding reaches the other devices through
    # a signal-buffer broadcast instead.
    _, _, graph = _build(2, None, with_tail=False)
    ir = str(graph)
    gpu_to_gpu = re.findall(
        r"mo\.transfer\[[^\n]*, gpu:\d+> to <\"gpu\", \d+>", ir
    )
    assert not gpu_to_gpu, gpu_to_gpu
    assert re.search(r"mo\.distributed\.broadcast", ir), ir[:2000]
