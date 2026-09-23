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
"""Checkpoint -> MAX weight names and storage for DeepSeek-V4.

The checkpoint carries no ``model.`` prefix and names every module after the
reference ``inference/model.py`` attributes, so the MAX modules are named to
match. What the adapter does change is the *storage* of the quantized
tensors, which the reference's ``convert.py`` step would otherwise do:

* fp8 projections (attention, ``indexer.wq_b``, the shared expert) keep their
  ``float8_e4m3fn`` weight; the ``<proj>.scale`` companion (e8m0, one per
  128x128 block) becomes ``<proj>.weight_scale`` **in float32**, because the
  blockwise fp8 matmul reads float32 scales. ``2 ** (e - 127)`` is exact.
* ``attn.wo_a`` is the exception: the checkpoint stores it fp8, but the
  reference declares it bf16 and uses the raw weight in a grouped einsum
  rather than through ``linear()``, so no activation quantization happens
  there and the block scale has to be folded in on the host. Loading it fp8
  and dropping the scale is what makes wo_a about 2400x too large.
* routed experts (``ffn.experts.<e>.w{1,2,3}``, e2m1 packed two per int8 byte,
  e8m0 scale per 32 elements) are stacked per layer into the two
  ``[E, N, K/2]`` uint8 tensors of :class:`DeepseekV4RoutedExperts`
  (``gate_up_proj`` = ``[w1; w3]`` along N, ``down_proj`` = ``w2``) with
  ``[E, N, K/32]`` e8m0 scales; the grouped W4A8 kernel reads both as is.
* everything else (bf16 / f32 / the int64 ``tid2eid`` tables) passes through.

The adapter keys on the *stored* dtypes, so a dequantized (f32 / bf16) copy of
the checkpoint passes through untouched and drives the dense f32 modules.

Note the scale rename is a suffix match on ``.scale`` and must stay one: the
mHC mixing parameters (``hc_attn_scale``, ``hc_ffn_scale``, ``hc_head_scale``)
are underscore-suffixed tensors of their own, not scales of a ``weight``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
from max.driver import Buffer, DLPackArray
from max.dtype import DType
from max.graph.type import Shape
from max.graph.weights import WeightData, Weights
from transformers.configuration_utils import PretrainedConfig

_SCALE_SUFFIX = ".scale"
_MAX_SCALE_SUFFIX = ".weight_scale"

# ``layers.<l>.ffn.experts.<e>.<w1|w2|w3>.<weight|scale>``
_EXPERT_TENSOR = re.compile(
    r"^(?P<prefix>.*\.ffn\.experts)\.(?P<expert>\d+)\.(?P<proj>w[123])\."
    r"(?P<kind>weight|scale)$"
)

GATE_UP_PROJ = "gate_up_proj"
DOWN_PROJ = "down_proj"

# Projections stored fp8 that the model runs dequantized (see the module
# docstring): matched on the module name, so ``mtp.*`` stages come along.
_HOST_DEQUANT_PROJECTIONS = (".attn.wo_a",)

_FP8_WEIGHT_BLOCK = 128


def as_uint8(data: DLPackArray) -> np.ndarray:
    """The raw bytes of a one-byte-per-element tensor (fp8, e8m0, int8)."""
    buffer = data if isinstance(data, Buffer) else Buffer.from_dlpack(data)
    return np.from_dlpack(buffer.view(DType.uint8))


def e8m0_to_float32(raw: np.ndarray) -> np.ndarray:
    """``2 ** (e - 127)``; exact in float32 for every e8m0 code but NaN (255)."""
    return np.exp2(raw.astype(np.float32) - 127.0)


def _e4m3_table() -> np.ndarray:
    """The 256 ``float8_e4m3fn`` values, indexed by their byte."""
    code = np.arange(256, dtype=np.uint8)
    exponent = ((code >> 3) & 0x0F).astype(np.int32)
    mantissa = (code & 0x07).astype(np.float32)
    magnitude = np.where(
        exponent == 0,
        mantissa * 2.0**-9,
        (1.0 + mantissa / 8.0) * np.exp2((exponent - 7).astype(np.float32)),
    ).astype(np.float32)
    # e4m3fn has no infinities; the all-ones exponent with a full mantissa is
    # the only NaN.
    magnitude[0x7F] = np.nan
    magnitude[0xFF] = np.nan
    return np.where(code >> 7 == 1, -magnitude, magnitude).astype(np.float32)


_E4M3 = _e4m3_table()


def with_dtype(array: np.ndarray, dtype: DType, name: str) -> WeightData:
    """``WeightData`` whose bytes are ``array`` reinterpreted as ``dtype``."""
    buffer = Buffer.from_numpy(np.ascontiguousarray(array))
    if dtype != DType.from_numpy(array.dtype):
        buffer = buffer.view(dtype)
    return WeightData(buffer, name, dtype, Shape(buffer.shape))


def dequantize_fp8_blocks(
    weight: WeightData, scale: WeightData, name: str
) -> WeightData:
    """An fp8 weight times its ``[N/128, K/128]`` e8m0 block scale, float32."""
    values = _E4M3[as_uint8(weight.data)]
    blocks = e8m0_to_float32(as_uint8(scale.data))
    expanded = np.repeat(
        np.repeat(blocks, _FP8_WEIGHT_BLOCK, axis=0), _FP8_WEIGHT_BLOCK, axis=1
    )[: values.shape[0], : values.shape[1]]
    return with_dtype(values * expanded, DType.float32, name)


def stack_experts(
    experts: Mapping[int, Mapping[str, np.ndarray]], prefix: str
) -> dict[str, WeightData]:
    """Per-expert packed fp4 tensors -> the two stacked routed-expert weights.

    ``experts[e]`` holds ``w1``, ``w2``, ``w3`` (uint8 ``[N, K/2]``) and
    ``w1.scale``, ``w2.scale``, ``w3.scale`` (uint8 e8m0 ``[N, K/32]``).
    Expert ids must be dense ``0..E-1``; the stacking order is the expert id.
    """
    ids = sorted(experts)
    if ids != list(range(len(ids))):
        raise ValueError(f"{prefix}: expert ids are not dense: {ids}")
    gate_up = np.stack(
        [
            np.concatenate([experts[e]["w1"], experts[e]["w3"]], axis=0)
            for e in ids
        ]
    )
    gate_up_scale = np.stack(
        [
            np.concatenate(
                [experts[e]["w1.scale"], experts[e]["w3.scale"]], axis=0
            )
            for e in ids
        ]
    )
    down = np.stack([experts[e]["w2"] for e in ids])
    down_scale = np.stack([experts[e]["w2.scale"] for e in ids])
    return {
        f"{prefix}.{GATE_UP_PROJ}.weight": with_dtype(
            gate_up, DType.uint8, f"{prefix}.{GATE_UP_PROJ}.weight"
        ),
        f"{prefix}.{GATE_UP_PROJ}.weight_scale": with_dtype(
            gate_up_scale,
            DType.float8_e8m0fnu,
            f"{prefix}.{GATE_UP_PROJ}.weight_scale",
        ),
        f"{prefix}.{DOWN_PROJ}.weight": with_dtype(
            down, DType.uint8, f"{prefix}.{DOWN_PROJ}.weight"
        ),
        f"{prefix}.{DOWN_PROJ}.weight_scale": with_dtype(
            down_scale,
            DType.float8_e8m0fnu,
            f"{prefix}.{DOWN_PROJ}.weight_scale",
        ),
    }


def convert_weight_data(
    state_dict: Mapping[str, WeightData],
) -> dict[str, WeightData]:
    """The storage conversion, on already-loaded ``WeightData``.

    Separate from :func:`convert_safetensor_state_dict` so a harness that
    reads the safetensors itself gets the same tensors as the pipeline.
    """
    new_state_dict: dict[str, WeightData] = {}
    # ``prefix -> expert id -> tensor``, filled only for packed-fp4 experts.
    packed_experts: dict[str, dict[int, dict[str, np.ndarray]]] = {}

    for name, data in state_dict.items():
        expert = _EXPERT_TENSOR.match(name)
        if expert is not None:
            # Routed experts are packed fp4 in the checkpoint (int8 storage).
            # A dequantized copy stores them wide and falls through.
            base = f"{expert['prefix']}.{expert['expert']}.{expert['proj']}"
            weight = state_dict.get(f"{base}.weight")
            if weight is not None and weight.dtype == DType.int8:
                key = expert["proj"] + (
                    ".scale" if expert["kind"] == "scale" else ""
                )
                packed_experts.setdefault(expert["prefix"], {}).setdefault(
                    int(expert["expert"]), {}
                )[key] = as_uint8(data.data)
                continue

        if name.endswith(_SCALE_SUFFIX):
            base = name[: -len(_SCALE_SUFFIX)]
            if base.endswith(_HOST_DEQUANT_PROJECTIONS):
                # Folded into the weight below; the module has no scale.
                continue
            max_name = base + _MAX_SCALE_SUFFIX
            weight = state_dict.get(f"{base}.weight")
            if (
                weight is not None
                and weight.dtype == DType.float8_e4m3fn
                and data.dtype == DType.float8_e8m0fnu
            ):
                new_state_dict[max_name] = with_dtype(
                    e8m0_to_float32(as_uint8(data.data)),
                    DType.float32,
                    max_name,
                )
                continue
            new_state_dict[max_name] = data
            continue

        if name.endswith(".weight") and data.dtype == DType.float8_e4m3fn:
            base = name[: -len(".weight")]
            scale = state_dict.get(base + _SCALE_SUFFIX)
            if base.endswith(_HOST_DEQUANT_PROJECTIONS) and scale is not None:
                new_state_dict[name] = dequantize_fp8_blocks(data, scale, name)
                continue

        new_state_dict[name] = data

    for prefix, experts in packed_experts.items():
        new_state_dict.update(stack_experts(experts, prefix))
    return new_state_dict


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    loaded: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        # TODO: Support DSpark (commit 8). The ``mtp.*`` stages are dropped for
        # now, mirroring what the DeepSeek-V3 checkpoint converter does with its
        # own MTP layer.
        if name.startswith("mtp."):
            continue
        loaded[name] = value.data()
    return convert_weight_data(loaded)
