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

"""CPU-only regression tests for Gemma3 multimodal language weight name remaps."""

from __future__ import annotations

from unittest.mock import NonCallableMock

from max.pipelines.architectures.gemma3multimodal.weight_adapters import (
    convert_safetensor_language_state_dict as convert_mm,
)
from max.pipelines.architectures.gemma3multimodal_modulev3.weight_adapters import (
    convert_safetensor_language_state_dict as convert_mm_v3,
)


def _weight_stub() -> NonCallableMock:
    w = NonCallableMock()
    w.data.return_value = "stub_weight_data"
    return w


def test_gemma3multimodal_language_preserves_hf_qkv_proj_names() -> None:
    """HF checkpoints use self_attn.{q,k,v}_proj; unfused StackedLinear exposes
    its children under those same names, so the adapter must pass them through
    unchanged after stripping the ``language_model.model.`` prefix."""
    state_dict = {
        "language_model.model.layers.0.self_attn.q_proj.weight": _weight_stub(),
        "language_model.model.layers.0.self_attn.k_proj.weight": _weight_stub(),
        "language_model.model.layers.0.self_attn.v_proj.weight": _weight_stub(),
        "language_model.model.layers.1.self_attn.q_proj.weight_scale": _weight_stub(),
        "unrelated.prefix.weight": _weight_stub(),
    }

    out = convert_mm(state_dict)  # type: ignore[arg-type]

    assert "layers.0.self_attn.q_proj.weight" in out
    assert "layers.0.self_attn.k_proj.weight" in out
    assert "layers.0.self_attn.v_proj.weight" in out
    assert "layers.1.self_attn.q_proj.weight_scale" in out
    assert "unrelated.prefix.weight" not in out


def test_gemma3multimodal_modulev3_language_preserves_hf_qkv_proj_names() -> (
    None
):
    """V3 uses language_model.layers.* after stripping language_model.model.,
    and the q/k/v_proj names are passed through unchanged."""
    state_dict = {
        "language_model.model.layers.0.self_attn.q_proj.weight": _weight_stub(),
        "language_model.model.layers.0.self_attn.v_proj.weight": _weight_stub(),
    }

    out = convert_mm_v3(state_dict)  # type: ignore[arg-type]

    assert "language_model.layers.0.self_attn.q_proj.weight" in out
    assert "language_model.layers.0.self_attn.v_proj.weight" in out
