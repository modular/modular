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
"""Gemma4 ModuleV3 attention tests (bf16, sliding and global k==v layers).

Ports the graph-arch tests in
``max/tests/integration/architectures/gemma4/test_attention.py`` to the
ModuleV3 ``Gemma4Attention``. The layer is wrapped in a small harness module
that reconstructs :class:`PagedCacheValues` from variadic KV graph inputs the
same way ``Gemma4.forward`` does, compiled directly (no precompiled MEFs),
and validated against the HF torch reference with the same tolerances.
"""

from __future__ import annotations

import copy
import math
from dataclasses import fields
from typing import NamedTuple

import numpy as np
import pytest
import torch
from conftest import (
    TEXT_GLOBAL_HEAD_DIM,
    TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR,
    TEXT_GLOBAL_ROPE_THETA,
    TEXT_HEAD_DIM,
    TEXT_HIDDEN_SIZE,
    TEXT_NUM_ATTENTION_HEADS,
    TEXT_NUM_GLOBAL_KEY_VALUE_HEADS,
    TEXT_NUM_HIDDEN_LAYERS,
    TEXT_NUM_KEY_VALUE_HEADS,
    TEXT_RMS_NORM_EPS,
    TEXT_SLIDING_WINDOW,
    TEXT_SLIDING_WINDOW_ROPE_THETA,
    Gemma4RotaryEmbedding,
    Gemma4TextAttention,
)
from max import tree
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.compilation import CompiledCallable
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.sharding import (
    DeviceMesh,
    PlacementMapping,
    Replicated,
)
from max.experimental.tensor import Tensor, default_dtype
from max.graph import DeviceRef, TensorType
from max.nn.kv_cache import KVCacheParams, MHAKVCacheParams
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.gemma4_modulev3.layers.attention import (
    Gemma4Attention,
)
from max.pipelines.architectures.gemma4_modulev3.layers.rotary_embedding import (
    ProportionalRotaryEmbedding,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16

# Rope table length. The V2 test builds ropes from
# ``max_position_embeddings`` (131072); the inverse frequencies are
# independent of table length, so a short table (>= the 11 test positions)
# keeps compile time down without changing numerics.
MAX_SEQ_LEN = 1152


def make_text_config() -> Gemma4TextConfig:
    """Gemma4TextConfig with the gemma-4-31B-it text_config values.

    Mirrors the testdata config.json the V2 test loads; the rope parameters
    and layer_types (5 sliding : 1 full) come from the class defaults, which
    match the 31B checkpoint.
    """
    return Gemma4TextConfig(
        hidden_size=TEXT_HIDDEN_SIZE,
        num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
        num_key_value_heads=TEXT_NUM_KEY_VALUE_HEADS,
        head_dim=TEXT_HEAD_DIM,
        num_hidden_layers=TEXT_NUM_HIDDEN_LAYERS,
        sliding_window=TEXT_SLIDING_WINDOW,
        num_global_key_value_heads=TEXT_NUM_GLOBAL_KEY_VALUE_HEADS,
        global_head_dim=TEXT_GLOBAL_HEAD_DIM,
        attention_k_eq_v=True,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        attn_implementation="eager",
    )


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


# --------------------------------------------------------------------------- #
# Torch reference (identical to the V2 test)
# --------------------------------------------------------------------------- #


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
    """Generates the reference outputs of the PyTorch attention layer.

    `layer_idx` affects whether the local or global `RoPE` is used. When
    `layer_idx % 6 == 5`, the global `RoPE` is used. Otherwise, the local
    `RoPE` is used.
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


# --------------------------------------------------------------------------- #
# ModuleV3 harness
# --------------------------------------------------------------------------- #


class AttentionHarness(Module[..., Tensor]):
    """Wraps ``Gemma4Attention`` for standalone compilation.

    Reconstructs :class:`PagedCacheValues` from the variadic flattened KV
    graph inputs the same way ``Gemma4.forward`` does (unflatten +
    ``from_upstream`` with a replicated placement over a 1-device mesh), and
    prepares the active rope's ``freqs_cis`` the way
    ``Gemma4TextModel.prepare_freq_cis`` does.
    """

    def __init__(
        self,
        attention: Gemma4Attention,
        kv_params: KVCacheParams,
        mesh: DeviceMesh,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.kv_params = kv_params
        self.mesh = mesh
        self.dtype = dtype

    def forward(
        self,
        x: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> Tensor:
        x = x.to(self.mesh)
        input_row_offsets = input_row_offsets.to(self.mesh)

        rope = (
            self.attention.rope_local
            if self.attention.use_local
            else self.attention.rope_global
        )
        rope.freqs_cis = rope.freqs_cis.cast(self.dtype).to(self.mesh)

        kv_inputs = iter(t._graph_value for t in variadic_args)
        kv_concrete = self.kv_params.unflatten_kv_inputs(kv_inputs)
        kv_mapping = PlacementMapping(
            self.mesh, (Replicated(),) * self.mesh.ndim
        )
        kv_collection = PagedCacheValues.from_upstream(kv_concrete, kv_mapping)
        return self.attention(
            x, kv_collection, input_row_offsets=input_row_offsets
        )


class CompiledAttention(NamedTuple):
    """Bundles a compiled attention harness with its KV-cache manager."""

    compiled: CompiledCallable[..., Tensor]
    kv_manager: PagedKVCacheManager


def make_attention_module(
    text_config: Gemma4TextConfig, device: Device, layer_idx: int
) -> tuple[Gemma4Attention, KVCacheParams]:
    """Builds the ModuleV3 Gemma4 attention layer and its KV-cache params.

    ``layer_idx`` selects the layer geometry: ``layer_idx % 6 == 5`` is a
    global (full-attention, k==v, head_dim 512) layer, anything else a
    sliding (head_dim 256) layer. The caller picks the construction context
    (``F.lazy()`` for compilation, eager otherwise).
    """
    assert text_config.layer_types is not None
    is_sliding = text_config.layer_types[layer_idx] == "sliding_attention"
    layer_type_counts = {
        layer_type: text_config.layer_types.count(layer_type)
        for layer_type in ("sliding_attention", "full_attention")
    }
    kv_params = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        devices=[DeviceRef.GPU()],
        n_kv_heads=(
            text_config.num_key_value_heads
            if is_sliding
            else text_config.num_global_key_value_heads
        ),
        head_dim=(
            text_config.head_dim if is_sliding else text_config.global_head_dim
        ),
        num_layers=layer_type_counts[
            "sliding_attention" if is_sliding else "full_attention"
        ],
        page_size=256,
    )

    with default_dtype(MAX_DTYPE):
        rope_local = RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=TEXT_SLIDING_WINDOW_ROPE_THETA,
            max_seq_len=MAX_SEQ_LEN,
            device=device,
            head_dim=text_config.head_dim,
            interleaved=False,
        )
        rope_global = ProportionalRotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=TEXT_GLOBAL_ROPE_THETA,
            max_seq_len=MAX_SEQ_LEN,
            device=device,
            head_dim=text_config.global_head_dim,
            interleaved=False,
            scaling_params=ProportionalScalingParams(
                partial_rotary_factor=TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR
            ),
        )
        attention = Gemma4Attention(
            rope_global=rope_global,
            rope_local=rope_local,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            num_global_key_value_heads=text_config.num_global_key_value_heads,
            attention_k_eq_v=text_config.attention_k_eq_v,
            hidden_size=text_config.hidden_size,
            kv_params=kv_params,
            layer_idx_in_cache=0,
            is_sliding=is_sliding,
            qk_norm_eps=text_config.rms_norm_eps,
            local_window_size=text_config.sliding_window,
        )
    return attention, kv_params


def make_kv_manager(
    session: InferenceSession, kv_params: KVCacheParams
) -> PagedKVCacheManager:
    return PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )


def build_max_attention(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights: dict[str, torch.Tensor],
    device: Device,
    layer_idx: int,
) -> CompiledAttention:
    """Builds and compiles the ModuleV3 Gemma4 attention harness."""
    mesh = DeviceMesh((device,), (1,), ("tp",))
    with F.lazy():
        attention, kv_params = make_attention_module(
            text_config, device, layer_idx
        )
        harness = AttentionHarness(attention, kv_params, mesh, MAX_DTYPE)
        harness.to(mesh)

    weights = {
        f"attention.{name}": value.cpu()
        for name, value in attention_weights.items()
    }
    input_type = TensorType(
        MAX_DTYPE,
        ["total_seq_len", text_config.hidden_size],
        device=DeviceRef.GPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=DeviceRef.GPU()
    )
    kv_types = kv_params.flattened_kv_inputs()
    compiled = harness.compile(
        input_type, input_row_offsets_type, *kv_types, weights=weights
    )
    return CompiledAttention(
        compiled=compiled, kv_manager=make_kv_manager(session, kv_params)
    )


def execute_max_attention(
    compiled_attention: CompiledAttention,
    input_tensor: torch.Tensor,
    device: Device,
) -> Buffer:
    """Runs a compiled attention harness against a fresh KV claim.

    Releases the request after execution so the shared kv_manager doesn't
    accumulate state across test invocations.
    """
    input_seq_len = input_tensor.shape[1]
    kv_manager = compiled_attention.kv_manager
    compiled = compiled_attention.compiled

    batch = [create_text_context(np.empty(input_seq_len))]
    kv_manager.claim(batch[0])
    try:
        kv_manager.alloc(batch[0])
        kv_runtime_inputs = kv_manager.runtime_inputs([batch])

        execute_args = [
            Buffer.from_dlpack(input_tensor[0]).to(device),
            Buffer.from_numpy(np.array([0, input_seq_len], dtype=np.uint32)).to(
                device
            ),
            *tree.leaves(kv_runtime_inputs),
        ]
        output = compiled.execute_raw(*execute_args)[0]
    finally:
        kv_manager.release(batch[0])
    assert isinstance(output, Buffer)
    return output


def execute_eager_attention(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    device: Device,
    layer_idx: int,
) -> torch.Tensor:
    """Runs ``Gemma4Attention`` op by op against a real paged KV cache.

    No ``Module.compile``: the KV manager's runtime buffers are wrapped as
    realized eager tensors, so every kernel that takes the
    :class:`PagedCacheValues` sees device buffers rather than graph values.
    """
    mesh = DeviceMesh((device,), (1,), ("tp",))
    attention, kv_params = make_attention_module(text_config, device, layer_idx)
    attention.load_state_dict(
        {name: value.cpu() for name, value in attention_weights.items()}
    )
    attention.to(mesh)
    rope = (
        attention.rope_local if attention.use_local else attention.rope_global
    )
    rope.freqs_cis = rope.freqs_cis.cast(MAX_DTYPE).to(mesh)

    input_seq_len = input_tensor.shape[1]
    kv_manager = make_kv_manager(session, kv_params)
    batch = [create_text_context(np.empty(input_seq_len))]
    kv_manager.claim(batch[0])
    try:
        kv_manager.alloc(batch[0])
        kv_runtime_inputs = kv_manager.runtime_inputs([batch])
        kv_leaves = (Tensor(storage=b) for b in tree.leaves(kv_runtime_inputs))
        (kv_concrete,) = kv_params.unflatten_kv_inputs(kv_leaves)
        kv_collection = PagedCacheValues(
            **{
                f.name: getattr(kv_concrete, f.name)
                for f in fields(kv_concrete)
            }
        )

        x = Tensor(storage=Buffer.from_dlpack(input_tensor[0]).to(device))
        input_row_offsets = Tensor(
            storage=Buffer.from_numpy(
                np.array([0, input_seq_len], dtype=np.uint32)
            ).to(device)
        )
        output = attention(
            x.to(mesh),
            kv_collection,
            input_row_offsets=input_row_offsets.to(mesh),
        )
        return torch.from_dlpack(output).cpu()
    finally:
        kv_manager.release(batch[0])


# --------------------------------------------------------------------------- #
# Fixtures (module-scoped: each unique compile happens once per process)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def device() -> Device:
    return Accelerator()


@pytest.fixture(scope="module")
def session(device: Device) -> InferenceSession:
    return InferenceSession(devices=[device])


@pytest.fixture(scope="module")
def text_config() -> Gemma4TextConfig:
    return make_text_config()


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


@pytest.fixture(scope="module")
def compiled_local_bf16(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights_local: dict[str, torch.Tensor],
    device: Device,
) -> CompiledAttention:
    return build_max_attention(
        session,
        text_config,
        attention_weights_local,
        device,
        layer_idx=0,
    )


@pytest.fixture(scope="module")
def compiled_global_bf16(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights_global: dict[str, torch.Tensor],
    device: Device,
) -> CompiledAttention:
    return build_max_attention(
        session,
        text_config,
        attention_weights_global,
        device,
        layer_idx=5,
    )


# --------------------------------------------------------------------------- #
# BF16 attention tests
# --------------------------------------------------------------------------- #


def test_attention_local(
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    attention_weights_local: dict[str, torch.Tensor],
    compiled_local_bf16: CompiledAttention,
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
    input_tensor: torch.Tensor,
    attention_weights_global: dict[str, torch.Tensor],
    compiled_global_bf16: CompiledAttention,
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


# --------------------------------------------------------------------------- #
# Eager (op-by-op) attention tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("layer_idx", "weights_fixture"),
    [(0, "attention_weights_local"), (5, "attention_weights_global")],
    ids=["local", "global"],
)
def test_attention_eager(
    request: pytest.FixtureRequest,
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    device: Device,
    layer_idx: int,
    weights_fixture: str,
) -> None:
    """Eager attention with a real paged KV cache matches the torch reference.

    Regression test for the functional kernel wrapper converting KV tensors
    to graph values outside a realization context.
    """
    attention_weights = request.getfixturevalue(weights_fixture)
    max_output = execute_eager_attention(
        session, text_config, attention_weights, input_tensor, device, layer_idx
    )
    torch_output = generate_torch_outputs(
        text_config, input_tensor, attention_weights, layer_idx=layer_idx
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(TORCH_DTYPE).cpu(),
        max_output.to(TORCH_DTYPE),
        rtol=2 * torch.finfo(TORCH_DTYPE).eps,
        atol=8 * torch.finfo(TORCH_DTYPE).eps,
    )
