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
"""Checkpoint names map onto the ModuleV3 GLM-5.3-Flash module tree.

The checkpoint is mixed precision layer by layer, so an FP8 weight has to be
recognized by its scale, not by its projection's name: every KDA layer has a
BF16 ``q_proj`` that the DeepSeek adapter would mistake for an MLA projection.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Accelerator
from max.dtype import DType
from max.graph.weights import WeightData
from max.pipelines.architectures.glm5_next_modulev3 import (
    model as glm5_next_model,
)
from max.pipelines.architectures.glm5_next_modulev3.weight_adapters import (
    convert_glm5_next_state_dict,
)

_LM = "model.language_model."


class _Weight:
    def __init__(self, data: WeightData) -> None:
        self._data = data

    def data(self) -> WeightData:
        return self._data


def _weight(
    name: str, shape: tuple[int, ...], dtype: npt.DTypeLike = np.float32
) -> _Weight:
    return _Weight(WeightData.from_numpy(np.zeros(shape, dtype=dtype), name))


def _convert(
    names: dict[str, tuple[int, ...]], dtype: npt.DTypeLike = np.float32
) -> dict[str, WeightData]:
    state = {name: _weight(name, shape, dtype) for name, shape in names.items()}
    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=4))
    return convert_glm5_next_state_dict(state, config)  # type: ignore[arg-type]


def test_mixed_precision_names() -> None:
    out = _convert(
        {
            f"{_LM}layers.0.self_attn.q_proj.weight": (8, 4),
            f"{_LM}layers.0.self_attn.q_conv1d.weight": (8, 1, 4),
            f"{_LM}layers.0.mlp.gate_proj.weight": (8, 4),
            f"{_LM}layers.0.mlp.gate_proj.weight_scale_inv": (1, 1),
            f"{_LM}layers.3.self_attn.q_a_proj.weight": (8, 4),
            f"{_LM}layers.3.self_attn.q_a_proj.weight_scale_inv": (1, 1),
            f"{_LM}layers.3.self_attn.kv_b_proj.weight": (8, 4),
            f"{_LM}layers.3.self_attn.o_proj.weight": (4, 8),
            f"{_LM}layers.3.self_attn.o_proj.weight_scale_inv": (1, 1),
            f"{_LM}layers.3.self_attn.kv_a_layernorm.weight": (4,),
            f"{_LM}layers.3.self_attn.indexer.index_kpool_compress_gate": (
                8,
                4,
            ),
            f"{_LM}layers.3.mlp.gate.weight": (16, 4),
            "lm_head.weight": (16, 4),
        }
    )
    lm = "language_model."
    assert set(out) == {
        # A BF16 KDA projection is a plain linear, whatever its name.
        f"{lm}layers.0.self_attn.q_proj.weight",
        f"{lm}layers.0.self_attn.q_conv1d",
        f"{lm}layers.0.mlp.gate_proj.weight",
        f"{lm}layers.0.mlp.gate_proj.weight_scale_inv",
        # A BF16 raw MLA projection drops `.weight`; an FP8 one keeps its
        # checkpoint names, as a linear does.
        f"{lm}layers.3.self_attn.q_a_proj.weight",
        f"{lm}layers.3.self_attn.q_a_proj.weight_scale_inv",
        f"{lm}layers.3.self_attn.kv_b_proj",
        f"{lm}layers.3.self_attn.o_proj.weight",
        f"{lm}layers.3.self_attn.o_proj.weight_scale_inv",
        f"{lm}layers.3.self_attn.kv_a_proj_layernorm",
        f"{lm}layers.3.self_attn.indexer.index_kpool_compress_gate.weight",
        f"{lm}layers.3.mlp.gate.gate_score.weight",
        f"{lm}lm_head.weight",
    }
    # The depthwise conv loses its singleton group axis.
    assert list(out[f"{lm}layers.0.self_attn.q_conv1d"].shape) == [8, 4]


def test_mhc_sites_and_float32_parameters() -> None:
    out = _convert(
        {
            f"{_LM}layers.0.hc_attn_fn": (24, 16),
            f"{_LM}layers.0.hc_ffn_base": (24,),
            f"{_LM}layers.0.self_attn.A_log": (2,),
        }
    )
    lm = "language_model."
    assert set(out) == {
        f"{lm}layers.0.hc_attn.hc_fn",
        f"{lm}layers.0.hc_ffn.hc_base",
        f"{lm}layers.0.self_attn.A_log",
    }
    assert all(w.dtype == DType.float32 for w in out.values())


def test_mhc_projection_keeps_checkpoint_dtype() -> None:
    """The model, not the adapter, picks the projection's dtype per device."""
    out = _convert(
        {
            f"{_LM}layers.0.hc_attn_fn": (24, 16),
            f"{_LM}layers.0.hc_attn_base": (24,),
        },
        np.float16,
    )
    lm = "language_model."
    assert out[f"{lm}layers.0.hc_attn.hc_fn"].dtype == DType.float16
    assert out[f"{lm}layers.0.hc_attn.hc_base"].dtype == DType.float32


def _device(api: str, architecture_name: str) -> MagicMock:
    device = MagicMock(spec=Accelerator)
    device.api = api
    device.architecture_name = architecture_name
    return device


@pytest.mark.parametrize(
    ("api", "architecture_name", "stored", "hidden_size", "expected"),
    [
        ("hip", "gfx950", DType.bfloat16, 4096, DType.bfloat16),
        # A float32 checkpoint is never narrowed.
        ("hip", "gfx950", DType.float32, 4096, DType.float32),
        # Where the fused op does not run, the op chain reads float32.
        ("hip", "gfx950", DType.bfloat16, 1536, DType.float32),
        ("hip", "gfx942", DType.bfloat16, 4096, DType.float32),
        ("cuda", "sm_100a", DType.bfloat16, 4096, DType.float32),
    ],
)
def test_hc_fn_dtype(
    monkeypatch: pytest.MonkeyPatch,
    api: str,
    architecture_name: str,
    stored: DType,
    hidden_size: int,
    expected: DType,
) -> None:
    monkeypatch.setattr(glm5_next_model, "accelerator_api", lambda: api)
    dtype = glm5_next_model._hc_fn_dtype(
        stored, 4, hidden_size, DType.bfloat16, _device(api, architecture_name)
    )
    assert dtype == expected


@pytest.mark.parametrize("hc_fn_dtype", [DType.bfloat16, DType.float32])
def test_prepare_state_dict_holds_hc_fn_in_config_dtype(
    hc_fn_dtype: DType,
) -> None:
    """A bfloat16 projection is kept or widened as the config says."""
    # bfloat16-exact, so widening must reproduce them exactly.
    values = np.array([[0.5, -1.25], [2.0, 0.0]], dtype=np.float32)
    hc_fn = WeightData.from_numpy(values, "hc_attn.hc_fn").astype(
        DType.bfloat16
    )
    hc_base = WeightData.from_numpy(np.zeros(2, np.float32), "hc_attn.hc_base")
    model = object.__new__(glm5_next_model.Glm5NextModel)
    out = model._prepare_state_dict(
        {"layers.0.hc_attn.hc_fn": hc_fn, "layers.0.hc_attn.hc_base": hc_base},
        SimpleNamespace(hc_fn_dtype=hc_fn_dtype),
    )
    assert out["layers.0.hc_attn.hc_base"] is hc_base
    prepared = out["layers.0.hc_attn.hc_fn"]
    assert prepared.dtype == hc_fn_dtype
    if hc_fn_dtype == DType.bfloat16:
        assert prepared is hc_fn
    else:
        np.testing.assert_array_equal(np.from_dlpack(prepared.data), values)


def test_stored_hc_fn_dtype() -> None:
    """One checkpoint dtype is shared by every site; a mix is rejected."""

    def hc_fn(dtype: DType) -> WeightData:
        return WeightData.from_numpy(np.zeros((2, 2), np.float32), "w").astype(
            dtype
        )

    stored = glm5_next_model._stored_hc_fn_dtype
    assert stored({}) == DType.float32
    assert (
        stored(
            {
                "layers.0.hc_attn.hc_fn": hc_fn(DType.bfloat16),
                "layers.0.hc_ffn.hc_fn": hc_fn(DType.bfloat16),
            }
        )
        == DType.bfloat16
    )
    with pytest.raises(ValueError, match="mixes"):
        stored(
            {
                "layers.0.hc_attn.hc_fn": hc_fn(DType.bfloat16),
                "layers.1.hc_attn.hc_fn": hc_fn(DType.float32),
            }
        )


def test_drops_mtp_layer_and_vision_tower() -> None:
    out = _convert(
        {
            f"{_LM}layers.4.input_layernorm.weight": (4,),
            "model.visual.patch_embed.proj.weight": (4, 4),
            f"{_LM}norm.weight": (4,),
        }
    )
    assert set(out) == {"language_model.norm.weight"}
