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
"""Gemma4 ModuleV3 attention run op by op, without ``Module.compile``.

Kept apart from ``test_attention.py`` because it runs through the eager
interpreter, so it needs the build-warmed eager op cache rather than CPU
recorded graph compiles.
"""

from __future__ import annotations

from dataclasses import fields

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
from max.engine import InferenceSession
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import Tensor
from test_common.context_utils import create_text_context
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig


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
