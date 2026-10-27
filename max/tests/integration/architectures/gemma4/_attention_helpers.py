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

This module holds the
plain-Python helpers that the test files reference directly:
`build_max_attention`, `execute_max_attention`, `assert_fp8_matches_bf16`,
the `CompiledAttention` bundle, and dtype constants.
"""

import json
import math
import os
from pathlib import Path

import torch
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.kv_cache import MHAKVCacheParams, PagedCacheValues
from max.nn.layer import Module
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.gemma4.layers.attention import (
    Gemma4Attention as MaxGemma4Attention,
)
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalRotaryEmbedding,
    ProportionalScalingParams,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

MAX_SEQ_LEN = 1152

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16


class CompiledAttention:
    """Bundles a compiled attention graph with its dedicated KV-cache manager.

    Cached per `(layer_idx, cache_dtype, weight set)` in a `scope="module"`
    fixture (in `conftest.py`) so each unique compile happens once per
    test process.

    The cache is built on the first :meth:`claim_kv_manager`, not here,
    because allocating needs the GPU and compiling does not: a CPU build
    action that records this test's compiles gets through every one of them
    only if nothing has asked for the device yet. See
    ``docs/internal/CompileOnCpuRunOnGpu.md``.
    """

    def __init__(
        self,
        compiled: Model,
        session: InferenceSession,
        kv_params: MHAKVCacheParams,
    ) -> None:
        self.compiled = compiled
        self._session = session
        self._kv_params = kv_params
        self._kv_manager: PagedKVCacheManager | None = None

    def claim_kv_manager(self) -> PagedKVCacheManager:
        """Returns this graph's cache manager, allocating it on first call.

        Returns:
            The manager, which every call after the first shares.
        """
        if self._kv_manager is None:
            self._kv_manager = PagedKVCacheManager(
                params=self._kv_params,
                total_num_pages=8,
                session=self._session,
                max_batch_size=128,
            )
        return self._kv_manager


# ---------------------------------------------------------------------------
# Shared fixtures for the attention tests
# (test_attention.py, test_attention_fp8_local.py, test_attention_fp8_global.py)
#
# All scope="module" so each unique compile in the file pays its cost once.
# These names are specific enough that other gemma4 tests do not collide.
# ---------------------------------------------------------------------------


def _attention_test_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    """Generate a unit-stddev-ish tensor for attention test fixtures."""
    return (torch.randn(shape) * (1.0 / math.sqrt(shape[-1]))).to(TORCH_DTYPE)


def make_attention_weights_local(
    text_config: Gemma4TextConfig,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)

    # calculated from google/gemma-3-1b-it checkpoint
    O_PROJ_STD = 0.0237
    K_PROJ_STD = 0.0309
    Q_PROJ_STD = 0.0284
    V_PROJ_STD = 0.0309
    K_NORM_STD = 0.793
    Q_NORM_STD = 0.68

    q_dim = text_config.head_dim * text_config.num_attention_heads
    kv_dim = text_config.head_dim * text_config.num_key_value_heads
    hidden_size = text_config.hidden_size

    return {
        "k_norm.weight": _attention_test_tensor((text_config.head_dim,))
        * K_NORM_STD,
        "k_proj.weight": _attention_test_tensor((kv_dim, hidden_size))
        * K_PROJ_STD,
        "o_proj.weight": _attention_test_tensor((hidden_size, q_dim))
        * O_PROJ_STD,
        "q_norm.weight": _attention_test_tensor((text_config.head_dim,))
        * Q_NORM_STD,
        "q_proj.weight": _attention_test_tensor((q_dim, hidden_size))
        * Q_PROJ_STD,
        "v_proj.weight": _attention_test_tensor((kv_dim, hidden_size))
        * V_PROJ_STD,
    }


def make_attention_weights_global(
    text_config: Gemma4TextConfig,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)

    # calculated from google/gemma-3-1b-it checkpoint
    O_PROJ_STD = 0.0237
    K_PROJ_STD = 0.0309
    Q_PROJ_STD = 0.0284
    K_NORM_STD = 0.793
    Q_NORM_STD = 0.68

    q_dim = text_config.global_head_dim * text_config.num_attention_heads
    kv_dim = (
        text_config.global_head_dim * text_config.num_global_key_value_heads
    )
    hidden_size = text_config.hidden_size

    return {
        "k_norm.weight": _attention_test_tensor((text_config.global_head_dim,))
        * K_NORM_STD,
        "k_proj.weight": _attention_test_tensor((kv_dim, hidden_size))
        * K_PROJ_STD,
        "o_proj.weight": _attention_test_tensor((hidden_size, q_dim))
        * O_PROJ_STD,
        "q_norm.weight": _attention_test_tensor((text_config.global_head_dim,))
        * Q_NORM_STD,
        "q_proj.weight": _attention_test_tensor((q_dim, hidden_size))
        * Q_PROJ_STD,
    }


def make_text_config() -> Gemma4TextConfig:
    path = os.environ["PIPELINES_TESTDATA"]
    config_path = Path(path) / "config.json"
    with open(config_path) as file:
        data = json.load(file)
    # Use "text_config" for the multimodal variants
    if "text_config" in data:
        return Gemma4TextConfig(
            **data["text_config"], attn_implementation="eager"
        )
    else:
        return Gemma4TextConfig(**data, attn_implementation="eager")


def build_max_attention_graph(
    text_config: Gemma4TextConfig,
    attention_weights: dict[str, torch.Tensor],
    dtype: DType,
    device_ref: DeviceRef,
    layer_idx: int,
    *,
    cache_dtype: DType | None = None,
) -> tuple[Graph, MaxGemma4Attention, MHAKVCacheParams]:
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
        kv_collection = kv_params.unflatten_kv_inputs(iter(kv_cache))[0]

        graph.output(
            attention(
                inputs.tensor,
                kv_collection,
                input_row_offsets=input_row_offsets.tensor,
            )
        )

    return graph, attention, kv_params


# ---------------------------------------------------------------------------
# Gemma 4's shared keys and values
#
# A sharing layer projects no keys and values of its own: it reuses the
# ones of the last layer of the same attention type before the sharing
# point, which live in that source layer's cache slot.
# ---------------------------------------------------------------------------

SHARING_NUM_HIDDEN_LAYERS = 4
SHARING_LAYER_TYPES = [
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
]
SHARING_NUM_KV_SHARED_LAYERS = 2
SHARING_HIDDEN_SIZE = 64
SHARING_NUM_HEADS = 4
SHARING_NUM_KV_HEADS = 2
SHARING_HEAD_DIM = 32
SHARING_GLOBAL_HEAD_DIM = 64
SHARING_SLIDING_WINDOW = 64


def make_kv_sharing_text_config() -> Gemma4TextConfig:
    """Builds a small Gemma 4 config whose trailing layers share keys and values."""
    return Gemma4TextConfig(
        hidden_size=SHARING_HIDDEN_SIZE,
        intermediate_size=2 * SHARING_HIDDEN_SIZE,
        num_hidden_layers=SHARING_NUM_HIDDEN_LAYERS,
        num_attention_heads=SHARING_NUM_HEADS,
        num_key_value_heads=SHARING_NUM_KV_HEADS,
        num_global_key_value_heads=SHARING_NUM_KV_HEADS,
        head_dim=SHARING_HEAD_DIM,
        global_head_dim=SHARING_GLOBAL_HEAD_DIM,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        sliding_window=SHARING_SLIDING_WINDOW,
        layer_types=list(SHARING_LAYER_TYPES),
        num_kv_shared_layers=SHARING_NUM_KV_SHARED_LAYERS,
        attention_bias=False,
        attn_implementation="eager",
    )


def make_kv_sharing_attention_weights(
    text_config: Gemma4TextConfig,
    layer_idx: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Builds random attention weights for one layer, in the reference layout.

    The names match the reference implementation's, so the same tensors load
    into both the MAX layer and the torch one.
    """
    generator = torch.Generator().manual_seed(seed)
    is_sliding = text_config.layer_types[layer_idx] == "sliding_attention"
    head_dim = (
        text_config.head_dim if is_sliding else text_config.global_head_dim
    )
    num_kv_heads = (
        text_config.num_key_value_heads
        if is_sliding
        else text_config.num_global_key_value_heads
    )
    q_dim = head_dim * text_config.num_attention_heads
    kv_dim = head_dim * num_kv_heads

    def _randn(*shape: int, std: float) -> torch.Tensor:
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32) * std
        ).to(TORCH_DTYPE)

    # The norms scale the queries and keys, and the output projection sets
    # the magnitude of the layer's output.
    return {
        "q_proj.weight": _randn(q_dim, text_config.hidden_size, std=0.05),
        "k_proj.weight": _randn(kv_dim, text_config.hidden_size, std=0.05),
        "v_proj.weight": _randn(kv_dim, text_config.hidden_size, std=0.05),
        "o_proj.weight": _randn(text_config.hidden_size, q_dim, std=0.1),
        "q_norm.weight": _randn(head_dim, std=0.7),
        "k_norm.weight": _randn(head_dim, std=0.7),
    }


class SourceAndSharingAttention(Module):
    """A source attention layer and a layer that shares its keys and values."""

    def __init__(
        self, source: MaxGemma4Attention, sharing: MaxGemma4Attention
    ) -> None:
        super().__init__()
        self.source = source
        self.sharing = sharing

    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Runs the source layer, then the sharing layer over its output."""
        h = self.source(x, kv_collection, input_row_offsets=input_row_offsets)
        return self.sharing(
            h, kv_collection, input_row_offsets=input_row_offsets
        )


def build_max_kv_sharing_attention_graph(
    text_config: Gemma4TextConfig,
    source_weights: dict[str, torch.Tensor],
    sharing_weights: dict[str, torch.Tensor],
    dtype: DType,
    device_ref: DeviceRef,
    source_layer_idx: int,
    sharing_layer_idx: int,
) -> tuple[Graph, Module, MHAKVCacheParams]:
    """Builds a graph with a source attention layer and a sharing layer.

    Both layers have the same attention type, so they share one KV cache,
    and the sharing layer reads the source layer's slot in it.
    """
    layer_type = text_config.layer_types[source_layer_idx]
    is_sliding = layer_type == "sliding_attention"
    head_dim = (
        text_config.head_dim if is_sliding else text_config.global_head_dim
    )
    num_kv_heads = (
        text_config.num_key_value_heads
        if is_sliding
        else text_config.num_global_key_value_heads
    )
    kv_params = MHAKVCacheParams(
        dtype=dtype,
        devices=[device_ref],
        n_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=sum(1 for t in text_config.layer_types if t == layer_type),
        page_size=256,
    )

    def _slot_in_cache(layer_idx: int) -> int:
        """The index of a layer within its own attention type's KV cache."""
        return sum(
            1 for t in text_config.layer_types[:layer_idx] if t == layer_type
        )

    def _attention(
        layer_idx: int, kv_share_source_layer_idx_in_cache: int | None = None
    ) -> MaxGemma4Attention:
        return MaxGemma4Attention(
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
            layer_idx_in_cache=_slot_in_cache(layer_idx),
            is_sliding=is_sliding,
            dtype=dtype,
            devices=[device_ref],
            qk_norm_eps=text_config.rms_norm_eps,
            local_window_size=text_config.sliding_window,
            kv_share_source_layer_idx_in_cache=(
                kv_share_source_layer_idx_in_cache
            ),
        )

    attention = SourceAndSharingAttention(
        source=_attention(source_layer_idx),
        sharing=_attention(
            sharing_layer_idx,
            kv_share_source_layer_idx_in_cache=_slot_in_cache(source_layer_idx),
        ),
    )
    attention.load_state_dict(
        {
            **{
                f"source.{name}": value.cpu()
                for name, value in source_weights.items()
            },
            **{
                f"sharing.{name}": value.cpu()
                for name, value in sharing_weights.items()
            },
        }
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
        "Gemma4KVSharingAttention",
        input_types=(
            input_type,
            input_row_offsets_type,
            *flattened_kv_types,
        ),
    ) as graph:
        inputs, input_row_offsets, *kv_cache = graph.inputs
        kv_collection = kv_params.unflatten_kv_inputs(iter(kv_cache))[0]

        graph.output(
            attention(
                inputs.tensor,
                kv_collection,
                input_row_offsets=input_row_offsets.tensor,
            )
        )

    return graph, attention, kv_params
