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
"""End-to-end GPT-2 greedy-decode test on the default (AsyncRT) accelerator.

This is the AsyncRT counterpart to ``test_gpt2_hal.py``: identical model,
weights, prompt, and CPU-reference comparison, but it runs under the default
device context (built WITHOUT ``--config=hal-device-context`` and WITHOUT
``MODULAR_DRIVER_PLUGINS``). Running both targets and confirming they produce
the same greedy token sequence compares the AsyncRT and HAL device-context
paths against the shared CPU reference.

It builds GPT-2 small from primitive ``max.experimental.nn`` layers (following
the "Build an LLM from scratch with MAX" guide), loads the real HuggingFace
``openai-community/gpt2`` safetensors checkpoint directly, and greedy-decodes a
short prompt. Weights are adapted at compile time
(``model.compile(weights=...)``) exactly as the guide does: Conv1D weights are
transposed to Linear layout, keys are prefixed with ``transformer.``, and
``lm_head.weight`` is tied to ``transformer.wte.weight``. There is no
intermediate ``.bin`` export.

Correctness is checked against a CPU run of the identical MAX model and
weights: the accelerator must produce the same greedy token sequence as the
host. This is a device-conformance contract that needs no external reference
(no PyTorch, no hardcoded token list). The test is hardware-agnostic: it
contains no vendor- or device-specific code and uses only generic
graph-compiler kernels (no ``custom_extensions``).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import max.experimental.functional as F
import numpy as np
from huggingface_hub import hf_hub_download
from max.driver import CPU, Accelerator, Device, accelerator_count
from max.dtype import DType
from max.experimental.nn import (
    Embedding,
    LayerNorm,
    Linear,
    Module,
    Sequential,
)
from max.experimental.tensor import (
    Tensor,
    TensorType,
    default_device,
    default_dtype,
)
from max.graph import Dim, DimLike, TensorValue
from max.graph.weights import SafetensorWeights, WeightData, Weights


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int | None = None
    layer_norm_epsilon: float = 1e-5


class GPT2MLP(Module):  # type: ignore[type-arg]
    def __init__(self, intermediate_size: int, config: GPT2Config) -> None:
        embed_dim = config.n_embd
        self.c_fc = Linear(embed_dim, intermediate_size, bias=True)
        self.c_proj = Linear(intermediate_size, embed_dim, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


@F.functional
def causal_mask(
    sequence_length: DimLike,
    num_tokens: DimLike,
    *,
    dtype: DType,
    device: Device,
) -> Tensor:
    n = Dim(sequence_length) + num_tokens
    mask = Tensor(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(sequence_length, n))
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)


class GPT2MultiHeadAttention(Module):  # type: ignore[type-arg]
    def __init__(self, config: GPT2Config) -> None:
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _attn(
        self,
        query: Tensor | TensorValue,
        key: Tensor | TensorValue,
        value: Tensor | TensorValue,
    ) -> Tensor | TensorValue:
        attn_weights = query @ key.transpose(-1, -2)
        attn_weights = attn_weights / math.sqrt(int(value.shape[-1]))

        seq_len = query.shape[-2]
        mask = causal_mask(seq_len, 0, dtype=query.dtype, device=query.device)
        attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights)
        return attn_weights @ value

    def _split_heads(
        self, tensor: Tensor | TensorValue, num_heads: int, head_size: int
    ) -> Tensor | TensorValue:
        new_shape = list(tensor.shape[:-1]) + [num_heads, head_size]
        return tensor.reshape(new_shape).transpose(-3, -2)

    def _merge_heads(
        self, tensor: Tensor | TensorValue, num_heads: int, head_size: int
    ) -> Tensor | TensorValue:
        tensor = tensor.transpose(-3, -2)
        new_shape = list(tensor.shape[:-2]) + [num_heads * head_size]
        return tensor.reshape(new_shape)

    def forward(self, hidden_states: Tensor) -> Tensor:
        qkv = self.c_attn(hidden_states)
        split = F.split(
            qkv,
            [self.split_size, self.split_size, self.split_size],
            axis=2,
        )
        query = cast(Tensor | TensorValue, split[0])
        key = cast(Tensor | TensorValue, split[1])
        value = cast(Tensor | TensorValue, split[2])

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = self._attn(query, key, value)
        attn_output = self._merge_heads(
            attn_output, self.num_heads, self.head_dim
        )
        return cast(Tensor, self.c_proj(cast(Tensor, attn_output)))


class GPT2Block(Module):  # type: ignore[type-arg]
    def __init__(self, config: GPT2Config) -> None:
        hidden_size = config.n_embd
        inner_dim = (
            config.n_inner if config.n_inner is not None else 4 * hidden_size
        )
        self.ln_1 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2MultiHeadAttention(config)
        self.ln_2 = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.attn(self.ln_1(hidden_states)) + hidden_states
        hidden_states = self.mlp(self.ln_2(hidden_states)) + hidden_states
        return hidden_states


class GPT2Model(Module):  # type: ignore[type-arg]
    def __init__(self, config: GPT2Config) -> None:
        self.wte = Embedding(config.vocab_size, dim=config.n_embd)
        self.wpe = Embedding(config.n_positions, dim=config.n_embd)
        self.h = Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: Tensor) -> Tensor:
        # Positions are 0..S-1 and each block is unary (no KV cache).
        _, seq_length = input_ids.shape
        tok_embeds = self.wte(input_ids)
        pos_embeds = self.wpe(
            Tensor.arange(
                seq_length, dtype=input_ids.dtype, device=input_ids.device
            )
        )
        x = tok_embeds + pos_embeds
        x = self.h(x)
        return self.ln_f(x)


class GPT2LMHeadModel(Module):  # type: ignore[type-arg]
    def __init__(self, config: GPT2Config) -> None:
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.lm_head(self.transformer(input_ids))


# GPT-2 small, matching the standard HuggingFace ``openai-community/gpt2``
# checkpoint.
_GPT2_SMALL = GPT2Config()

# HuggingFace repo for the GPT-2 small checkpoint. The single
# ``model.safetensors`` file is downloaded (and cached) on first run; set
# ``GPT2_SAFETENSORS_PATH`` to point at a local copy to run offline.
_GPT2_REPO = "openai-community/gpt2"
_GPT2_SAFETENSORS_FILE = "model.safetensors"

# Run the model in float32 (the checkpoint's native dtype). float32 keeps the
# per-step argmax stable enough that the host and accelerator agree token for
# token; an fp16 path would need a looser, device-dependent contract.
_DTYPE = DType.float32

# Prompt "The capital of France is" and how many tokens to greedy-decode.
_PROMPT_TOKENS: list[int] = [464, 3139, 286, 4881, 318]
_NUM_GENERATE = 20


# ---------------------------------------------------------------------------
# Weight loading — the guide's HuggingFace safetensors adapter.
#
# GPT-2 uses Conv1D (not Linear) for attention/MLP projections, which stores
# weights as [in, out]; MAX Linear expects [out, in], so those are transposed.
# Keys are prefixed with ``transformer.``, the causal-mask buffers are skipped,
# and ``lm_head.weight`` is tied to ``transformer.wte.weight``.
# ---------------------------------------------------------------------------

_CONV1D_LAYERS = ("c_attn", "c_proj", "c_fc")
_SKIP_SUFFIXES = (".attn.bias", ".attn.masked_bias")


def _to_numpy(wd: WeightData) -> np.ndarray:
    return np.array(np.from_dlpack(wd))


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
) -> dict[str, WeightData]:
    result: dict[str, WeightData] = {}

    for key, value in state_dict.items():
        if any(key.endswith(suffix) for suffix in _SKIP_SUFFIXES):
            continue

        mapped_key = (
            key if key.startswith("transformer.") else f"transformer.{key}"
        )
        arr = _to_numpy(value.data())

        if any(
            layer in mapped_key for layer in _CONV1D_LAYERS
        ) and mapped_key.endswith(".weight"):
            arr = np.ascontiguousarray(arr.T)

        result[mapped_key] = WeightData.from_numpy(arr, mapped_key)

    wte_key = "transformer.wte.weight"
    if "lm_head.weight" not in result and wte_key in result:
        wte_arr = np.array(result[wte_key].data)
        result["lm_head.weight"] = WeightData.from_numpy(
            wte_arr, "lm_head.weight"
        )

    return result


def _load_gpt2_numpy_state() -> dict[str, np.ndarray]:
    """Download (and cache) the GPT-2 safetensors checkpoint, adapt it to the
    model's parameter layout, and return plain numpy arrays.

    Returning numpy (rather than ``WeightData``) lets us wrap a fresh
    ``WeightData`` per device, so compiling the same weights onto both the host
    and the accelerator never shares a host buffer between the two graphs.
    """
    local = os.environ.get("GPT2_SAFETENSORS_PATH")
    path = (
        Path(local)
        if local
        else Path(hf_hub_download(_GPT2_REPO, _GPT2_SAFETENSORS_FILE))
    )
    weights = SafetensorWeights([path])
    adapted = convert_safetensor_state_dict(dict(weights.items()))
    return {key: np.asarray(wd.data) for key, wd in adapted.items()}


def _state_dict_for_device(
    numpy_state: dict[str, np.ndarray],
) -> dict[str, WeightData]:
    return {
        key: WeightData.from_numpy(arr, key) for key, arr in numpy_state.items()
    }


def _greedy_decode(
    config: GPT2Config,
    numpy_state: dict[str, np.ndarray],
    dev: Device,
    prompt_tokens: list[int],
    num_generate: int,
) -> list[int]:
    """Build, compile, and greedy-decode GPT-2 on ``dev`` from the given
    weights, returning the generated token IDs.

    Stateless (no KV cache): each step re-runs the whole sequence, so this is
    a correctness path, not a serving-latency path.
    """
    with F.lazy(), default_device(dev), default_dtype(_DTYPE):
        model = GPT2LMHeadModel(config)

    input_type = TensorType(
        dtype=DType.int64, shape=("batch", "seq_len"), device=dev
    )
    compiled = model.compile(
        input_type, weights=_state_dict_for_device(numpy_state)
    )

    tokens = np.array([prompt_tokens], dtype=np.int64)
    generated: list[int] = []
    for _ in range(num_generate):
        out = compiled(Tensor.from_dlpack(tokens).to(dev))
        out_np = np.from_dlpack(out.to(CPU()))
        next_token = int(np.argmax(out_np[0, -1]))
        generated.append(next_token)
        tokens = np.concatenate([tokens, [[next_token]]], axis=1)
    return generated


# ---------------------------------------------------------------------------
# Per-module device-conformance tests.
#
# These mirror the "Build an LLM from scratch with MAX" accelerator test
# (``oss/max-llm-book/tests/test_gpt2_accel.py``): each GPT-2 submodule is run
# on both CPU and the default (AsyncRT) accelerator from identical weights and
# the outputs are compared element-wise. Unlike the guide's eager
# ``Module.to(...)`` comparison, every forward pass here is COMPILED to a graph
# and executed, exercising the same device-context graph-execution path as the
# end-to-end decode below. This is the AsyncRT counterpart to the same tests in
# ``test_gpt2_hal.py``.
# ---------------------------------------------------------------------------


def _module_weights(
    build: Callable[[], Module],  # type: ignore[type-arg]
) -> dict[str, Tensor]:
    """Build a module eagerly on CPU and return its parameters for reuse.

    Handing the same CPU-resident tensors to every device's compile (which
    transfers them) keeps the only intended source of divergence the backend
    kernels, never a fresh random initialization. The parameters are passed
    through as ``Tensor`` objects rather than numpy: a materialized weight is
    read-only, and ``np.from_dlpack`` cannot import a read-only buffer with the
    DLPack version in use.
    """
    with default_device(CPU()), default_dtype(_DTYPE):
        module = build()
    return dict(module.parameters)


def _compiled_forward(
    build: Callable[[], Module],  # type: ignore[type-arg]
    weights: dict[str, Tensor],
    dev: Device,
    input_array: np.ndarray,
) -> np.ndarray:
    """Compile ``build()`` on ``dev`` with ``weights`` and run one forward."""
    with F.lazy(), default_device(dev), default_dtype(_DTYPE):
        module = build()
    input_type = TensorType(
        dtype=DType.from_numpy(input_array.dtype),
        shape=input_array.shape,
        device=dev,
    )
    compiled = module.compile(input_type, weights=weights)
    out = compiled(Tensor.from_dlpack(input_array).to(dev))
    return np.from_dlpack(out.to(CPU()))


def _probe_device(dev: Device) -> None:
    """Compile and run a one-layer graph on ``dev`` to confirm it can execute.

    Surfaces an accelerator that is detected but unusable as an immediate
    failure at import rather than a confusing mid-suite crash.
    """
    weights = _module_weights(lambda: Linear(4, 4))
    _compiled_forward(
        lambda: Linear(4, 4), weights, dev, np.ones((1, 4), dtype=np.float32)
    )


def _resolve_device() -> Accelerator:
    """Resolve the accelerator, failing hard if one is unavailable.

    This test is gated to GPU machines, so a missing or unusable accelerator is
    a broken environment, not a reason to skip. Skipping would silently mask a
    regression in the device-context path the test exists to exercise, so raise
    instead: the whole module then fails rather than reporting green.
    """
    if accelerator_count() == 0:
        raise RuntimeError(
            "No accelerator detected; test_gpt2_default requires a GPU."
        )
    device = Accelerator()
    _probe_device(device)
    return device


# Resolve at import so a missing or unusable device fails the whole module up
# front rather than surfacing later as a confusing per-test error.
_DEVICE = _resolve_device()


def _device() -> Accelerator:
    return _DEVICE


def _forward_cpu_vs_device(
    build: Callable[[], Module],  # type: ignore[type-arg]
    input_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compile+run ``build()`` on CPU and the device from identical weights.

    Returns ``(cpu_out, device_out)`` as numpy arrays; the only intended
    difference between them is the backend kernel implementation.
    """
    weights = _module_weights(build)
    cpu_out = _compiled_forward(build, weights, CPU(), input_array)
    device_out = _compiled_forward(build, weights, _device(), input_array)
    return cpu_out, device_out


# GPU-vs-CPU drift for single-module forward passes. Attention softmax on short
# inner axes (seq_len <= 32) can differ by ~3e-3 across backends.
_BLOCK_FORWARD_ATOL = 5e-3
# Whole-model outputs accumulate fp32 noise over 12 transformer blocks, and
# near-zero output positions inflate relative error, so the full-model
# comparison uses a much looser atol that still covers cross-backend drift.
_MODEL_FORWARD_ATOL = 2e-1


class TestGPT2Config:
    """GPT2Config dataclass defaults and overrides."""

    def test_default_values(self) -> None:
        config = GPT2Config()
        assert config.vocab_size == 50257
        assert config.n_positions == 1024
        assert config.n_embd == 768
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_inner is None
        assert config.layer_norm_epsilon == 1e-5

    def test_custom_values(self) -> None:
        config = GPT2Config()
        config.n_embd = 512
        config.n_layer = 6
        assert config.n_embd == 512
        assert config.n_layer == 6


class TestGPT2MLP:
    """GPT2MLP construction and CPU-vs-device forward conformance."""

    def test_initialization(self) -> None:
        with F.lazy(), default_device(CPU()), default_dtype(_DTYPE):
            mlp = GPT2MLP(intermediate_size=3072, config=GPT2Config())
        assert mlp.c_fc is not None
        assert mlp.c_proj is not None

    def test_forward_pass_matches_cpu(self) -> None:
        config = GPT2Config()
        cpu_out, device_out = _forward_cpu_vs_device(
            lambda: GPT2MLP(intermediate_size=3072, config=config),
            np.ones((2, 10, config.n_embd), dtype=np.float32),
        )
        np.testing.assert_allclose(
            device_out, cpu_out, atol=_BLOCK_FORWARD_ATOL, rtol=1e-3
        )


class TestCausalMask:
    """causal_mask shape and values on the accelerator."""

    def test_causal_mask_shape(self) -> None:
        seq_len = 5
        with F.lazy(), default_device(_device()), default_dtype(_DTYPE):
            mask = causal_mask(seq_len, 0, dtype=_DTYPE, device=_device())
        assert [int(d) for d in mask.shape] == [seq_len, seq_len]

    def test_causal_mask_values(self) -> None:
        seq_len = 4

        class _MaskModule(Module):  # type: ignore[type-arg]
            # Wrap causal_mask so it compiles and runs like any module; the
            # dummy input only carries the dtype/device to the mask.
            def forward(self, x: Tensor) -> Tensor:
                return causal_mask(seq_len, 0, dtype=x.dtype, device=x.device)

        _, mask_np = _forward_cpu_vs_device(
            _MaskModule, np.zeros((1,), dtype=np.float32)
        )

        for i in range(seq_len):
            for j in range(i + 1):
                assert mask_np[i, j] == 0.0
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask_np[i, j] == float("-inf")


class TestGPT2MultiHeadAttention:
    """GPT2MultiHeadAttention construction, head reshapes, and forward."""

    def test_initialization(self) -> None:
        config = GPT2Config()
        with F.lazy(), default_device(CPU()), default_dtype(_DTYPE):
            attn = GPT2MultiHeadAttention(config)
        assert attn.embed_dim == config.n_embd
        assert attn.num_heads == config.n_head
        assert attn.head_dim == config.n_embd // config.n_head
        assert attn.c_attn is not None
        assert attn.c_proj is not None

    def test_split_heads_shape(self) -> None:
        # Pure reshape + transpose: the shape is static, so it resolves under
        # F.lazy() without executing anything on the device.
        config = GPT2Config()
        batch_size, seq_len = 2, 10
        with F.lazy(), default_device(_device()), default_dtype(_DTYPE):
            attn = GPT2MultiHeadAttention(config)
            tensor = Tensor.ones(
                [batch_size, seq_len, config.n_embd], dtype=_DTYPE
            )
            split = attn._split_heads(tensor, config.n_head, attn.head_dim)
        assert [int(d) for d in split.shape] == [
            batch_size,
            config.n_head,
            seq_len,
            attn.head_dim,
        ]

    def test_merge_heads_shape(self) -> None:
        # Pure reshape + transpose; see test_split_heads_shape.
        config = GPT2Config()
        batch_size, seq_len = 2, 10
        with F.lazy(), default_device(_device()), default_dtype(_DTYPE):
            attn = GPT2MultiHeadAttention(config)
            tensor = Tensor.ones(
                [batch_size, config.n_head, seq_len, attn.head_dim],
                dtype=_DTYPE,
            )
            merged = attn._merge_heads(tensor, config.n_head, attn.head_dim)
        assert [int(d) for d in merged.shape] == [
            batch_size,
            seq_len,
            config.n_embd,
        ]

    def test_forward_pass_matches_cpu(self) -> None:
        config = GPT2Config()
        cpu_out, device_out = _forward_cpu_vs_device(
            lambda: GPT2MultiHeadAttention(config),
            np.ones((2, 10, config.n_embd), dtype=np.float32),
        )
        np.testing.assert_allclose(
            device_out, cpu_out, atol=_BLOCK_FORWARD_ATOL, rtol=1e-3
        )


class TestLayerNorm:
    """LayerNorm construction and CPU-vs-device forward conformance."""

    def test_initialization(self) -> None:
        dim = 768
        with F.lazy(), default_device(CPU()), default_dtype(_DTYPE):
            ln = LayerNorm(dim, eps=1e-5)
        assert ln.eps == 1e-5
        assert isinstance(ln.weight, Tensor)
        assert isinstance(ln.bias, Tensor)
        assert [int(d) for d in ln.weight.shape] == [dim]
        assert [int(d) for d in ln.bias.shape] == [dim]

    def test_forward_pass_matches_cpu(self) -> None:
        dim = 768
        cpu_out, device_out = _forward_cpu_vs_device(
            lambda: LayerNorm(dim),
            np.ones((2, 10, dim), dtype=np.float32),
        )
        np.testing.assert_allclose(
            device_out, cpu_out, atol=_BLOCK_FORWARD_ATOL, rtol=1e-3
        )


class TestGPT2Block:
    """GPT2Block construction and CPU-vs-device forward conformance."""

    def test_initialization(self) -> None:
        with F.lazy(), default_device(CPU()), default_dtype(_DTYPE):
            block = GPT2Block(GPT2Config())
        assert block.ln_1 is not None
        assert block.attn is not None
        assert block.ln_2 is not None
        assert block.mlp is not None

    def test_initialization_with_custom_inner_dim(self) -> None:
        config = GPT2Config()
        config.n_inner = 2048
        with F.lazy(), default_device(CPU()), default_dtype(_DTYPE):
            block = GPT2Block(config)
        assert block.mlp is not None

    def test_forward_pass_matches_cpu(self) -> None:
        config = GPT2Config()
        cpu_out, device_out = _forward_cpu_vs_device(
            lambda: GPT2Block(config),
            np.ones((2, 10, config.n_embd), dtype=np.float32),
        )
        np.testing.assert_allclose(
            device_out, cpu_out, atol=_BLOCK_FORWARD_ATOL, rtol=1e-3
        )


class TestGPT2Model:
    """GPT2Model construction and CPU-vs-device forward conformance."""

    def test_initialization(self) -> None:
        with F.lazy(), default_device(CPU()), default_dtype(_DTYPE):
            model = GPT2Model(GPT2Config())
        assert model.wte is not None
        assert model.wpe is not None
        assert model.h is not None
        assert model.ln_f is not None

    def test_forward_pass_matches_cpu(self) -> None:
        config = GPT2Config()
        cpu_out, device_out = _forward_cpu_vs_device(
            lambda: GPT2Model(config),
            np.zeros((2, 10), dtype=np.int64),
        )
        np.testing.assert_allclose(
            device_out, cpu_out, atol=_MODEL_FORWARD_ATOL, rtol=1e-3
        )


class TestGPT2LMHeadModel:
    """GPT2LMHeadModel construction and CPU-vs-device forward conformance."""

    def test_initialization(self) -> None:
        config = GPT2Config()
        with F.lazy(), default_device(CPU()), default_dtype(_DTYPE):
            model = GPT2LMHeadModel(config)
        assert model.config == config
        assert model.transformer is not None
        assert model.lm_head is not None

    def test_forward_pass_matches_cpu(self) -> None:
        config = GPT2Config()
        cpu_out, device_out = _forward_cpu_vs_device(
            lambda: GPT2LMHeadModel(config),
            np.zeros((2, 10), dtype=np.int64),
        )
        np.testing.assert_allclose(
            device_out, cpu_out, atol=_MODEL_FORWARD_ATOL, rtol=1e-3
        )


class TestModelDimensions:
    """Model dimensions are internally consistent."""

    def test_head_dimensions(self) -> None:
        config = GPT2Config()
        assert config.n_embd % config.n_head == 0
        assert config.n_embd // config.n_head == 64

    def test_mlp_inner_dimension(self) -> None:
        config = GPT2Config()
        assert 4 * config.n_embd == 3072


def test_gpt2_default_matches_cpu_reference() -> None:
    """GPT-2 greedy-decodes identically on the default accelerator and CPU.

    This runs under the default device context (AsyncRT) — the AsyncRT
    counterpart to ``test_gpt2_hal``. The model code is identical; no
    ``MODULAR_DRIVER_PLUGINS`` is needed (AsyncRT links its driver directly).
    The accelerator's greedy token sequence must match a CPU run of the
    identical model and weights.
    """
    # A failed weights download is a hard error, not a skip: silently skipping
    # would mask a broken weights path or a sandbox with no network (this test
    # carries the `requires-network`/`no-sandbox` tags so the download works).
    numpy_state = _load_gpt2_numpy_state()

    cpu_tokens = _greedy_decode(
        _GPT2_SMALL, numpy_state, CPU(), _PROMPT_TOKENS, _NUM_GENERATE
    )
    acc_tokens = _greedy_decode(
        _GPT2_SMALL, numpy_state, _device(), _PROMPT_TOKENS, _NUM_GENERATE
    )

    print(f"[gpt2_default] cpu: {cpu_tokens}")
    print(f"[gpt2_default] acc: {acc_tokens}")

    # Guard against a degenerate decode (e.g. all-same token from zeroed or
    # mis-loaded weights) that would make the comparison pass trivially.
    assert len(set(cpu_tokens)) > 1, (
        f"CPU reference decode is degenerate: {cpu_tokens}"
    )
    assert acc_tokens == cpu_tokens, (
        "Default-accelerator greedy decode does not match the CPU reference.\n"
        f"  cpu: {cpu_tokens}\n"
        f"  acc: {acc_tokens}"
    )
