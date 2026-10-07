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
same way ``Gemma4.forward`` does, compiled, and validated against the HF
torch reference with the same tolerances.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
import torch
from _attention_helpers import (
    MAX_DTYPE,
    TORCH_DTYPE,
    attention_test_tensor,
    generate_torch_outputs,
    make_attention_module,
    make_attention_weights_global,
    make_attention_weights_local,
    make_kv_manager,
    make_text_config,
)
from max import tree
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.compilation import CompiledCallable
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Replicated,
)
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.gemma4_modulev3.layers.attention import (
    Gemma4Attention,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

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
        kv_mapping = DeviceMapping(self.mesh, (Replicated(),) * self.mesh.ndim)
        kv_collection = PagedCacheValues.from_upstream(kv_concrete, kv_mapping)
        return self.attention(
            x, kv_collection, input_row_offsets=input_row_offsets
        )


class CompiledAttention(NamedTuple):
    """Bundles a compiled attention harness with its KV-cache manager."""

    compiled: CompiledCallable[..., Tensor]
    kv_manager: PagedKVCacheManager


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
    return attention_test_tensor((1, 11, text_config.hidden_size)).to("cuda")


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


# Each test requests its compiled fixture first, so the compile runs before
# `input_tensor` touches CUDA and can be recorded on CPU.
def test_attention_local(
    compiled_local_bf16: CompiledAttention,
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    attention_weights_local: dict[str, torch.Tensor],
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
    compiled_global_bf16: CompiledAttention,
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    attention_weights_global: dict[str, torch.Tensor],
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
