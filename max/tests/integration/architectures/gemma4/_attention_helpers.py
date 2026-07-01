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
"""Shared helpers for the Gemma4 attention tests.

The actual pytest fixtures (`text_config`, `input_tensor`,
`attention_weights_*`, `session`, `device`) live in `conftest.py` so
pytest auto-discovers them without imports.  This module holds the
plain-Python helpers that the test files reference directly:
`build_max_attention`, `execute_max_attention`, `assert_fp8_matches_bf16`,
`generate_torch_outputs`, the `CompiledAttention` bundle, and
dtype constants.

`test_attention.py` uses Bazel test sharding (`per_test_shard_count = 4`)
to parallelize 4 tests across 4 CI workers via round-robin distribution.
Each shard runs as its own pytest process, so each test compiles its
graphs in parallel with the others.  Module-scoped fixtures ensure each
unique graph compiles once per shard.
"""

import copy
from typing import Any, NamedTuple

import numpy as np
import torch
from conftest import (  # type: ignore[import-not-found]
    Gemma4RotaryEmbedding,
    Gemma4TextAttention,
)
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.gemma4.layers.attention import (
    Gemma4Attention as MaxGemma4Attention,
)
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalRotaryEmbedding,
    ProportionalScalingParams,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

MAX_SEQ_LEN = 1152

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16


class CompiledAttention(NamedTuple):
    """Bundles a compiled attention graph with its dedicated KV-cache manager.

    Cached per `(layer_idx, cache_dtype, weight set)` in a `scope="module"`
    fixture (in `conftest.py`) so each unique compile happens once per
    test process.
    """

    compiled: Model
    kv_manager: PagedKVCacheManager


def _get_position_embeddings(
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    use_global_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates rotary position embeddings based on the input tensor shape."""
    seq_len = input_tensor.shape[1]
    position_ids = torch.arange(
        seq_len, dtype=torch.long, device="cuda"
    ).unsqueeze(0)

    rope_params = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_params, dict) and "sliding_attention" in rope_params:
        # v5: single embedding handles both layer types natively
        rotary_emb = Gemma4RotaryEmbedding(config=text_config, device="cuda")
        layer_type = (
            "full_attention" if use_global_rope else "sliding_attention"
        )
        cos, sin = rotary_emb(input_tensor, position_ids, layer_type=layer_type)
    else:
        # v4: need separate embedding with hacked config for local rope
        if use_global_rope:
            rotary_emb = Gemma4RotaryEmbedding(
                config=text_config, device="cuda"
            )
        else:
            config = copy.deepcopy(text_config)
            config.rope_theta = config.rope_local_base_freq
            config.rope_scaling = {"rope_type": "default"}
            rotary_emb = Gemma4RotaryEmbedding(config=config, device="cuda")
        cos, sin = rotary_emb(input_tensor, position_ids)

    return cos.to(TORCH_DTYPE).to("cuda"), sin.to(TORCH_DTYPE).to("cuda")


def _causal_attention_mask(seq_len: int) -> torch.Tensor:
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"),
        diagonal=1,
    )
    attention_mask = torch.zeros(
        1, 1, seq_len, seq_len, dtype=TORCH_DTYPE, device="cuda"
    )
    attention_mask = attention_mask.masked_fill(
        causal_mask[None, None, :, :], torch.finfo(TORCH_DTYPE).min
    )
    return attention_mask


@torch.no_grad()
def generate_torch_outputs(
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    layer_idx: int,
) -> torch.Tensor:
    """Generates the outputs of the MAX and PyTorch attention layers.

    `layer_idx` affects whether the local or global `RoPE` is used. When
    `layer_idx % 6 == 5`, the global `RoPE` is used. Otherwise, the local `RoPE`
    is used.
    """
    layer = (
        Gemma4TextAttention(
            text_config,
            layer_idx=layer_idx,
        )
        .to(TORCH_DTYPE)
        .to("cuda")
    )

    for name, param in layer.named_parameters():
        param.data = attention_weights[name].to(TORCH_DTYPE).to("cuda")

    attention_mask = _causal_attention_mask(input_tensor.shape[1])
    use_global_rope = layer_idx % 6 == 5
    position_embeddings = _get_position_embeddings(
        text_config, input_tensor, use_global_rope
    )

    return layer(input_tensor, position_embeddings, attention_mask)[0]


def build_max_attention(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights: dict[str, torch.Tensor],
    dtype: DType,
    device_ref: DeviceRef,
    layer_idx: int,
    *,
    cache_dtype: DType | None = None,
) -> CompiledAttention:
    """Builds and compiles the MAX Gemma4 attention graph.

    Hoist calls to this into a module-scoped fixture so each unique
    `(layer_idx, cache_dtype, weight set)` combination pays the compile cost
    only once per test process.

    `layer_idx` affects whether the local or global `RoPE` is used. When
    `layer_idx % 6 == 5`, the global `RoPE` is used. Otherwise, the local
    `RoPE` is used.

    `cache_dtype` controls the KV cache storage dtype.  Pass
    `DType.float8_e4m3fn` to exercise the fp8-KV path (automatically routed
    to the native pure-fp8 MHA op).  Defaults to `dtype` (= bf16).
    """
    state_dict = {
        weight_name: value.cpu()
        for weight_name, value in attention_weights.items()
    }

    cache_dtype_eff = cache_dtype if cache_dtype is not None else dtype
    kv_params_local = MHAKVCacheParams(
        dtype=cache_dtype_eff,
        devices=[device_ref],
        n_kv_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        num_layers=len(
            [lt for lt in text_config.layer_types if lt == "sliding_attention"]
        ),
        page_size=256,
    )

    kv_params_global = MHAKVCacheParams(
        dtype=cache_dtype_eff,
        devices=[device_ref],
        n_kv_heads=text_config.num_global_key_value_heads,
        head_dim=text_config.global_head_dim,
        num_layers=len(
            [lt for lt in text_config.layer_types if lt == "full_attention"]
        ),
        page_size=256,
    )

    kv_params = (
        kv_params_local
        if text_config.layer_types[layer_idx] == "sliding_attention"
        else kv_params_global
    )

    attention = MaxGemma4Attention(
        rope_global=ProportionalRotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=1000000.0,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.global_head_dim,
            interleaved=False,
            scaling_params=ProportionalScalingParams(0.25),
        ),
        rope_local=Llama3RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=10000.0,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.head_dim,
            interleaved=False,
            scaling_params=None,
        ),
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        num_global_key_value_heads=text_config.num_global_key_value_heads,
        attention_k_eq_v=text_config.attention_k_eq_v,
        hidden_size=text_config.hidden_size,
        kv_params=kv_params,
        global_head_dim=text_config.global_head_dim,
        layer_idx=layer_idx,
        layer_idx_in_cache=0,
        is_sliding=text_config.layer_types[layer_idx] == "sliding_attention",
        dtype=dtype,
        devices=[device_ref],
        qk_norm_eps=text_config.rms_norm_eps,
        local_window_size=text_config.sliding_window,
    )
    attention.load_state_dict(state_dict)

    # Set up blank KV cache.
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )

    # Construct input types.
    input_type = TensorType(
        dtype,
        ["total_seq_len", text_config.hidden_size],
        device=device_ref,
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    flattened_kv_types = kv_params.flattened_kv_inputs()

    # Build graph.
    with Graph(
        "Gemma3Attention",
        input_types=(
            input_type,
            input_row_offsets_type,
            *flattened_kv_types,
        ),
    ) as graph:
        inputs, input_row_offsets, *kv_cache = graph.inputs
        kv_collection = kv_params.unflatten_kv_inputs(iter(kv_cache)).inputs[0]

        graph.output(
            attention(
                inputs.tensor,
                kv_collection,
                input_row_offsets=input_row_offsets.tensor,
            )
        )

    compiled = session.load(graph, weights_registry=attention.state_dict())
    return CompiledAttention(compiled=compiled, kv_manager=kv_manager)


def execute_max_attention(
    compiled_attention: CompiledAttention,
    input_tensor: torch.Tensor,
    device: Device,
) -> torch.Tensor:
    """Runs a previously compiled attention graph against a fresh KV claim.

    Releases the request after execution so the shared kv_manager doesn't
    accumulate state across test invocations.
    """
    input_seq_len = input_tensor.shape[1]
    kv_manager = compiled_attention.kv_manager
    compiled = compiled_attention.compiled

    batch = [create_text_context(np.empty(input_seq_len))]
    kv_manager.claim(batch[0].request_id, replica_idx=0)
    try:
        kv_manager.alloc(batch[0], replica_idx=0)
        kv_runtime_inputs = kv_manager.runtime_inputs([batch])

        # Under fp8 KV the kv_params.get_symbolic_inputs() expands with
        # `kv_scales` buffer inputs.  Mirror that on the runtime side by
        # including them in the execute call when present.
        execute_args: list[Any] = [
            Buffer.from_dlpack(input_tensor[0]).to(device),
            Buffer.from_numpy(np.array([0, input_seq_len], dtype=np.uint32)).to(
                device
            ),
            *kv_runtime_inputs.flatten(),
        ]
        output = compiled.execute(*execute_args)[0]
    finally:
        kv_manager.release(batch[0].request_id, replica_idx=0)
    return output


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flat tensors (cast to fp32)."""
    af = a.to(torch.float32).flatten()
    bf = b.to(torch.float32).flatten()
    return float(
        torch.dot(af, bf) / (torch.linalg.norm(af) * torch.linalg.norm(bf))
    )


def assert_fp8_matches_bf16(
    bf16_compiled: CompiledAttention,
    fp8_compiled: CompiledAttention,
    input_tensor: torch.Tensor,
    device: Device,
    layer_idx: int,
    head_dim_for_log: int,
) -> None:
    """Shared helper: execute the bf16 reference and fp8 paths on the same
    inputs from already-compiled attention graphs; assert cosine >= 0.99.

    The bf16 reference uses the dtype = MAX_DTYPE cache (= bf16). The fp8
    path uses `cache_dtype=float8_e4m3fn` + per-block fp32 scales at
    granularity=64 (production Gemma4 wiring).  Both paths use
    `rope.interleaved=False` (the trained Gemma4 RoPE convention).
    """
    bf16_out = execute_max_attention(bf16_compiled, input_tensor, device)
    fp8_out = execute_max_attention(fp8_compiled, input_tensor, device)

    bf16_t = from_dlpack(bf16_out).to(torch.float32)
    fp8_t = from_dlpack(fp8_out).to(torch.float32)
    cos = _cosine_similarity(bf16_t, fp8_t)
    max_abs_diff = float((bf16_t - fp8_t).abs().max())
    print(
        f"[fp8_vs_bf16] layer_idx={layer_idx} head_dim={head_dim_for_log} "
        f"cosine={cos:.6f} max_abs_diff={max_abs_diff:.4f}"
    )
    assert cos >= 0.99, (
        "fp8 KV attention output diverged from bf16 baseline: "
        f"cosine={cos:.4f} < 0.99 (layer_idx={layer_idx} "
        f"head_dim={head_dim_for_log})"
    )
