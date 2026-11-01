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
"""Gemma4 attention tests (bf16 and fp8-KV variants).

Module-scoped fixtures ensure each unique graph compiles once per shard.
See `_attention_helpers.py` for build/execute helpers and `conftest.py`
for shared fixtures.
"""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from _attention_helpers import (
    MAX_DTYPE,
    TORCH_DTYPE,
    CompiledAttention,
    _attention_test_tensor,
    build_max_attention_graph,
    build_max_kv_sharing_attention_graph,
    make_attention_weights_global,
    make_attention_weights_local,
    make_kv_sharing_attention_weights,
    make_kv_sharing_text_config,
    make_text_config,
)
from conftest import (
    Gemma4RotaryEmbedding,
    Gemma4TextAttention,
)
from max import tree
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from test_common.context_utils import create_text_context
from test_common.graph_utils import is_b100_b200
from torch.utils.dlpack import from_dlpack
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig


@pytest.fixture(scope="module")
def session(device: Device) -> InferenceSession:
    return InferenceSession(devices=[device])


@pytest.fixture(scope="module")
def device() -> Device:
    return Accelerator()


@pytest.fixture(scope="module")
def text_config() -> Gemma4TextConfig:
    return make_text_config()


# Staging this needs a GPU, so every test below requests it after the compiled
# fixtures it uses: that way the CPU build action that records the compiles
# reaches them before anything asks for hardware it does not have. See
# `docs/internal/CompileOnCpuRunOnGpu.md`.
@pytest.fixture(scope="module")
def input_tensor(text_config: Gemma4TextConfig) -> torch.Tensor:
    torch.manual_seed(42)
    return _attention_test_tensor((1, 11, text_config.hidden_size)).to("cuda")


@pytest.fixture(scope="module")
def attention_weights_local(
    text_config: Gemma4TextConfig,
) -> dict[str, torch.Tensor]:
    return make_attention_weights_local(text_config)


@pytest.fixture(scope="module")
def attention_weights_global(
    text_config: Gemma4TextConfig,
) -> dict[str, torch.Tensor]:
    return make_attention_weights_global(text_config)


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
    graph, attention, kv_params = build_max_attention_graph(
        text_config,
        attention_weights,
        dtype,
        device_ref,
        layer_idx,
        cache_dtype=cache_dtype,
    )

    # Only the compile: the bundle allocates its KV cache on first use, so a
    # CPU build action recording this test reaches every compile before
    # anything asks for the device.
    compiled = session.load(graph, weights_registry=attention.state_dict())
    return CompiledAttention(compiled, session, kv_params)


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
    kv_manager = compiled_attention.claim_kv_manager()
    compiled = compiled_attention.compiled

    batch = [create_text_context(np.empty(input_seq_len))]
    kv_manager.claim(batch[0])
    try:
        kv_manager.alloc(batch[0])
        kv_runtime_inputs = kv_manager.runtime_inputs([batch])

        # Under fp8 KV the kv_params.get_symbolic_inputs() expands with
        # `kv_scales` buffer inputs.  Mirror that on the runtime side by
        # including them in the execute call when present.
        execute_args: list[Any] = [
            Buffer.from_dlpack(input_tensor[0]).to(device),
            Buffer.from_numpy(np.array([0, input_seq_len], dtype=np.uint32)).to(
                device
            ),
            *tree.leaves(kv_runtime_inputs),
        ]
        output = compiled.execute(*execute_args)[0]
    finally:
        kv_manager.release(batch[0])
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


# ---------------------------------------------------------------------------
# BF16 compiled fixtures (shared by bf16 tests and as baselines for fp8)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_local_bf16(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights_local: dict[str, torch.Tensor],
) -> CompiledAttention:
    return build_max_attention(
        session,
        text_config,
        attention_weights_local,
        MAX_DTYPE,
        DeviceRef.GPU(),
        layer_idx=0,
    )


@pytest.fixture(scope="module")
def compiled_global_bf16(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights_global: dict[str, torch.Tensor],
) -> CompiledAttention:
    return build_max_attention(
        session,
        text_config,
        attention_weights_global,
        MAX_DTYPE,
        DeviceRef.GPU(),
        layer_idx=5,
    )


# ---------------------------------------------------------------------------
# Native pure-fp8 compiled fixtures (no per-block scales, scale=1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_local_native_fp8(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights_local: dict[str, torch.Tensor],
) -> CompiledAttention:
    if not is_b100_b200():
        pytest.skip("Native FP8 MHA requires B200 (SM100)")
    return build_max_attention(
        session,
        text_config,
        attention_weights_local,
        MAX_DTYPE,
        DeviceRef.GPU(),
        layer_idx=0,
        cache_dtype=DType.float8_e4m3fn,
    )


@pytest.fixture(scope="module")
def compiled_global_native_fp8(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights_global: dict[str, torch.Tensor],
) -> CompiledAttention:
    if not is_b100_b200():
        pytest.skip("Native FP8 MHA requires B200 (SM100)")
    return build_max_attention(
        session,
        text_config,
        attention_weights_global,
        MAX_DTYPE,
        DeviceRef.GPU(),
        layer_idx=5,
        cache_dtype=DType.float8_e4m3fn,
    )


# ---------------------------------------------------------------------------
# BF16 attention tests
# ---------------------------------------------------------------------------


def test_attention_local(
    text_config: Gemma4TextConfig,
    attention_weights_local: dict[str, torch.Tensor],
    compiled_local_bf16: CompiledAttention,
    input_tensor: torch.Tensor,
    device: Device,
) -> None:
    max_output = execute_max_attention(
        compiled_local_bf16, input_tensor, device
    )

    torch_output = generate_torch_outputs(
        text_config, input_tensor, attention_weights_local, layer_idx=0
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(TORCH_DTYPE),
        from_dlpack(max_output).to(TORCH_DTYPE),
        rtol=2 * torch.finfo(TORCH_DTYPE).eps,
        atol=8 * torch.finfo(TORCH_DTYPE).eps,
    )


def test_attention_global(
    text_config: Gemma4TextConfig,
    attention_weights_global: dict[str, torch.Tensor],
    compiled_global_bf16: CompiledAttention,
    input_tensor: torch.Tensor,
    device: Device,
) -> None:
    max_output = execute_max_attention(
        compiled_global_bf16, input_tensor, device
    )
    torch_output = generate_torch_outputs(
        text_config,
        input_tensor,
        attention_weights_global,
        layer_idx=5,
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(TORCH_DTYPE),
        from_dlpack(max_output).to(TORCH_DTYPE),
        rtol=2 * torch.finfo(TORCH_DTYPE).eps,
        atol=8 * torch.finfo(TORCH_DTYPE).eps,
    )


# ---------------------------------------------------------------------------
# Native pure-fp8 regression tests
# ---------------------------------------------------------------------------
# These exercise the native pure-fp8 path: fp8 Q/K/V read directly from the
# paged cache, Q@K^T and P@V both raw fp8 MMAs at tensorwise scale=1, bf16
# output. Routes through `mo.rope_split_store.ragged.paged` (outputs roped Q
# as fp8) and `mo.mha.ragged.paged` (fp8 Q+KV in, bf16 out).
# Cosine vs the bf16 baseline must clear the same 0.99 smoke bar. Same
# sensitivity caveat: random-weight smoke gate, not a sufficient bug-detector
# — end-to-end serving accuracy (e.g. gsm8k under a real checkpoint) is the
# authoritative correctness gate.


def test_attention_native_fp8_matches_bf16_local(
    text_config: Gemma4TextConfig,
    compiled_local_bf16: CompiledAttention,
    compiled_local_native_fp8: CompiledAttention,
    input_tensor: torch.Tensor,
    device: Device,
) -> None:
    """Native pure-fp8 (no scales) sliding (head_dim=256) layer vs bf16."""
    assert_fp8_matches_bf16(
        compiled_local_bf16,
        compiled_local_native_fp8,
        input_tensor,
        device,
        layer_idx=0,
        head_dim_for_log=text_config.head_dim,
    )


def test_attention_native_fp8_matches_bf16_global(
    text_config: Gemma4TextConfig,
    compiled_global_bf16: CompiledAttention,
    compiled_global_native_fp8: CompiledAttention,
    input_tensor: torch.Tensor,
    device: Device,
) -> None:
    """Native pure-fp8 (no scales) global (head_dim=512) layer vs bf16."""
    assert_fp8_matches_bf16(
        compiled_global_bf16,
        compiled_global_native_fp8,
        input_tensor,
        device,
        layer_idx=5,
        head_dim_for_log=text_config.global_head_dim,
    )


# ---------------------------------------------------------------------------
# Gemma 4's shared keys and values
# ---------------------------------------------------------------------------


@dataclass
class SharingCase:
    """The source layer and the sharing layer of one attention type."""

    layer_type: str
    source_layer_idx: int
    sharing_layer_idx: int


class SharedKVCache:
    """The cache the reference implementation shares keys and values through.

    The reference's ``Gemma4TextAttention`` keeps the keys and values a
    sharing layer reuses in ``past_key_values.shared_layers``, keyed by the
    layer that stored them, and skips the cache update for a sharing layer.
    This stands in for that cache: the update hands back the states
    unchanged, so the stored states are the source layer's own.
    """

    def __init__(self) -> None:
        self.shared_layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return key_states, value_states


@pytest.fixture(scope="module")
def sharing_text_config() -> Gemma4TextConfig:
    return make_kv_sharing_text_config()


@pytest.fixture(scope="module", params=["sliding_attention", "full_attention"])
def sharing_case(
    request: Any, sharing_text_config: Gemma4TextConfig
) -> SharingCase:
    """The source layer and the sharing layer of one attention type."""
    layer_type = request.param
    layer_types = sharing_text_config.layer_types
    first_shared_layer = (
        sharing_text_config.num_hidden_layers
        - sharing_text_config.num_kv_shared_layers
    )
    source_layer_idx = max(
        i for i in range(first_shared_layer) if layer_types[i] == layer_type
    )
    sharing_layer_idx = next(
        i
        for i in range(
            first_shared_layer, sharing_text_config.num_hidden_layers
        )
        if layer_types[i] == layer_type
    )
    return SharingCase(
        layer_type=layer_type,
        source_layer_idx=source_layer_idx,
        sharing_layer_idx=sharing_layer_idx,
    )


@pytest.fixture(scope="module")
def sharing_weights(
    sharing_text_config: Gemma4TextConfig, sharing_case: SharingCase
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The source layer's and the sharing layer's attention weights.

    The two layers get different weights, so an output that uses the sharing
    layer's own keys and values cannot pass for one that uses the source
    layer's.
    """
    return (
        make_kv_sharing_attention_weights(
            sharing_text_config, sharing_case.source_layer_idx, seed=1
        ),
        make_kv_sharing_attention_weights(
            sharing_text_config, sharing_case.sharing_layer_idx, seed=2
        ),
    )


@pytest.fixture(scope="module")
def compiled_sharing(
    session: InferenceSession,
    sharing_text_config: Gemma4TextConfig,
    sharing_case: SharingCase,
    sharing_weights: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> CompiledAttention:
    source_weights, sharing_layer_weights = sharing_weights
    graph, attention, kv_params = build_max_kv_sharing_attention_graph(
        sharing_text_config,
        source_weights,
        sharing_layer_weights,
        MAX_DTYPE,
        DeviceRef.GPU(),
        sharing_case.source_layer_idx,
        sharing_case.sharing_layer_idx,
    )
    compiled = session.load(graph, weights_registry=attention.state_dict())
    return CompiledAttention(compiled, session, kv_params)


@torch.no_grad()
def generate_torch_sharing_outputs(
    text_config: Gemma4TextConfig,
    case: SharingCase,
    input_tensor: torch.Tensor,
    source_weights: dict[str, torch.Tensor],
    sharing_weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates the sharing layer's outputs from the reference implementation.

    The source layer runs first and stores its keys and values in the
    cache; the sharing layer then attends to those instead of projecting
    its own. The second output is what the sharing layer computes with no
    cache to share from, which is what it would produce if it did not
    share.
    """
    source = (
        Gemma4TextAttention(text_config, layer_idx=case.source_layer_idx)
        .to(TORCH_DTYPE)
        .to("cuda")
    )
    sharing = (
        Gemma4TextAttention(text_config, layer_idx=case.sharing_layer_idx)
        .to(TORCH_DTYPE)
        .to("cuda")
    )
    for name, param in source.named_parameters():
        param.data = source_weights[name].to(TORCH_DTYPE).to("cuda")
    for name, param in sharing.named_parameters():
        param.data = sharing_weights[name].to(TORCH_DTYPE).to("cuda")

    rotary_emb = Gemma4RotaryEmbedding(config=text_config, device="cuda")
    position_ids = torch.arange(
        input_tensor.shape[1], dtype=torch.long, device="cuda"
    ).unsqueeze(0)
    cos, sin = rotary_emb(
        input_tensor, position_ids, layer_type=case.layer_type
    )
    attention_mask = _causal_attention_mask(input_tensor.shape[1])

    cache = SharedKVCache()
    hidden_states = source(
        input_tensor, (cos, sin), attention_mask, past_key_values=cache
    )[0]
    shared = sharing(
        hidden_states, (cos, sin), attention_mask, past_key_values=cache
    )[0]
    own = sharing(hidden_states, (cos, sin), attention_mask)[0]
    return shared, own


def test_attention_kv_sharing_matches_torch(
    sharing_text_config: Gemma4TextConfig,
    sharing_case: SharingCase,
    sharing_weights: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    compiled_sharing: CompiledAttention,
    device: Device,
) -> None:
    """A sharing layer attends to the source layer's keys and values.

    The sequence is shorter than the sliding window, so the window does not
    bound the history the sharing layer sees.
    """
    seq_len = 11
    torch.manual_seed(42)
    input_tensor = _attention_test_tensor(
        (1, seq_len, sharing_text_config.hidden_size)
    ).to("cuda")

    max_output = execute_max_attention(compiled_sharing, input_tensor, device)
    source_weights, sharing_layer_weights = sharing_weights
    torch_output, torch_output_without_sharing = generate_torch_sharing_outputs(
        sharing_text_config,
        sharing_case,
        input_tensor,
        source_weights,
        sharing_layer_weights,
    )

    # The two differ, so the comparison below tells the sharing apart.
    assert not torch.allclose(torch_output, torch_output_without_sharing)

    max_result = from_dlpack(max_output).to(TORCH_DTYPE).squeeze(0)
    shared_error = (max_result - torch_output.squeeze(0)).abs().max()
    own_error = (
        (max_result - torch_output_without_sharing.squeeze(0)).abs().max()
    )
    # A layer that projected its own keys and values lands on the second
    # reference, which puts the two the other way round at any tolerance.
    assert shared_error < own_error, (
        f"closer to the unshared reference ({own_error.item()}) than to the "
        f"shared one ({shared_error.item()})"
    )

    # The full-attention layer projects twice the channels the sliding one
    # does (global_head_dim over four heads, a 256-wide o_proj against 128
    # for the sliding layer), which widens its bf16 accumulation floor: mi355
    # put one element of 704 at 1.2x the bound the sliding case clears.
    torch.testing.assert_close(
        torch_output.squeeze(0).to(TORCH_DTYPE),
        max_result,
        rtol=2 * torch.finfo(TORCH_DTYPE).eps,
        atol=16 * torch.finfo(TORCH_DTYPE).eps,
    )
