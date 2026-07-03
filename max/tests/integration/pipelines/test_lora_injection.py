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

"""Structural tests for ``LoRAManager.apply``."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.lora import LoRALinear, StackedLinearLoRA
from max.nn.stacked_linear import StackedLinear
from max.pipelines.lora import LoRAConfig, LoRAManager

_HIDDEN = 16
_DEVICE = DeviceRef.CPU()
_DTYPE = DType.float32


class _Container(Module):
    """Minimal concrete Module for structural tests; forward is unused."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError


class _Attention(_Container):
    def __init__(self) -> None:
        super().__init__()
        # Unfused stacked QKV: q/k/v are child Linears folded into one matmul.
        self.qkv_proj = StackedLinear(
            in_dim=_HIDDEN,
            out_dims=[_HIDDEN, _HIDDEN, _HIDDEN],
            names=["q_proj", "k_proj", "v_proj"],
            dtype=_DTYPE,
            device=_DEVICE,
            stacked=False,
        )
        self.o_proj = Linear(_HIDDEN, _HIDDEN, dtype=_DTYPE, device=_DEVICE)


class _MLP(_Container):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = Linear(_HIDDEN, _HIDDEN, dtype=_DTYPE, device=_DEVICE)


class _Block(_Container):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()


def _manager() -> LoRAManager:
    return LoRAManager(
        config=LoRAConfig(max_num_loras=2, max_lora_rank=8),
        base_model_path="test-base",
        base_dtype=_DTYPE,
        n_heads=1,
        n_kv_heads=1,
        head_dim=_HIDDEN,
        max_lora_seq_len=128,
    )


def test_apply_wraps_standalone_linears() -> None:
    model = _Block()
    matched = _manager().apply(model, {"o_proj", "down_proj"})
    assert isinstance(model.self_attn.o_proj, LoRALinear)
    assert isinstance(model.mlp.down_proj, LoRALinear)
    assert matched == {"o_proj", "down_proj"}


def test_apply_wraps_qkv_stacked_linear() -> None:
    model = _Block()
    matched = _manager().apply(model, {"q_proj", "k_proj", "v_proj"})
    qkv = model.self_attn.qkv_proj
    assert isinstance(qkv, StackedLinearLoRA)
    # q/k/v stay plain base Linears inside the fused module; they are not
    # wrapped individually.
    assert not isinstance(qkv.sublayers["q_proj"], LoRALinear)
    assert matched == {"q_proj", "k_proj", "v_proj"}


def test_apply_wraps_qkv_when_any_child_targeted() -> None:
    model = _Block()
    # Targeting just q_proj still wraps the whole fused projection.
    matched = _manager().apply(model, {"o_proj", "q_proj"})
    assert isinstance(model.self_attn.qkv_proj, StackedLinearLoRA)
    assert isinstance(model.self_attn.o_proj, LoRALinear)
    assert matched == {"o_proj", "q_proj"}


def test_apply_preserves_qkv_weight_fqns() -> None:
    model = _Block()
    _manager().apply(model, {"q_proj", "k_proj", "v_proj"})
    keys = set(model.raw_state_dict().keys())
    # Base q/k/v keep their checkpoint names (StackedLinear name-omit).
    assert "self_attn.q_proj.weight" in keys
    # Fused adapter weights sit at qkv_lora.*, matching LoRAManager's combine.
    assert "self_attn.qkv_lora.lora_A.weight" in keys
    assert "self_attn.qkv_lora.lora_B_q.weight" in keys
    assert "self_attn.qkv_lora.lora_B_kv.weight" in keys


def test_apply_qkv_is_idempotent() -> None:
    model = _Block()
    manager = _manager()
    manager.apply(model, {"q_proj"})
    wrapped = model.self_attn.qkv_proj
    again = manager.apply(model, {"q_proj"})
    assert model.self_attn.qkv_proj is wrapped
    assert again == set()


def test_apply_preserves_weight_fqns() -> None:
    model = _Block()
    _manager().apply(model, {"o_proj"})
    keys = set(model.raw_state_dict().keys())
    # Base weight keeps its checkpoint name; adapter weights sit beside it.
    assert "self_attn.o_proj.weight" in keys
    assert "self_attn.o_proj.lora_A.weight" in keys
    assert "self_attn.o_proj.lora_B.weight" in keys
    # The base weight is not nested under a wrapper attribute.
    assert "self_attn.o_proj.base.weight" not in keys


def test_apply_is_idempotent() -> None:
    model = _Block()
    manager = _manager()
    manager.apply(model, {"o_proj"})
    wrapped = model.self_attn.o_proj
    again = manager.apply(model, {"o_proj"})
    assert model.self_attn.o_proj is wrapped
    assert again == set()
