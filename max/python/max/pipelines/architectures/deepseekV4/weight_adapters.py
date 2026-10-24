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
match. The one exception is a block's mHC triple: the checkpoint stores
``hc_attn_fn`` flat on the block, and the model holds it in a
:class:`~max.nn.HyperConnection` as ``hc_attn.hc_fn``. Beyond that, the
adapter changes the *storage* of the quantized tensors, which the reference's
``convert.py`` step would otherwise do:

* fp8 projections (attention, ``indexer.wq_b``, the shared expert) keep their
  ``float8_e4m3fn`` weight; the ``<proj>.scale`` companion (e8m0, one per
  128x128 block) becomes ``<proj>.weight_scale`` **in float32**, because the
  blockwise fp8 matmul reads float32 scales. ``2 ** (e - 127)`` is exact.
* routed experts (``ffn.experts.<e>.w{1,2,3}``, e2m1 packed two per int8 byte,
  e8m0 scale per 32 elements) are stacked per layer into the two
  ``[E, N, K/2]`` uint8 tensors of :class:`DeepseekV4RoutedExperts`
  (``gate_up_proj`` = ``[w1; w3]`` along N, ``down_proj`` = ``w2``) with
  ``[E, N, K/32]`` e8m0 scales; the grouped W4A8 kernel reads both as is.
* the compressors' ``wkv`` / ``wgate`` and every norm weight are stored bf16
  but declared float32 by the reference, which upcasts them on load; so does
  this adapter (exact). ``head`` is the exception: the reference upcasts it
  too, but here it stays bf16 (see ``DeepseekV4._lm_head``).
* everything else (bf16 / f32 / the int64 ``tid2eid`` tables) passes through.

Last, projections that read the same input are stacked along N into one
module weight, so each pair runs as one GEMM: ``attn.wq_a`` + ``attn.wkv`` ->
``attn.wqa_wkv``, ``shared_experts.w1`` + ``w3`` -> ``w13``, and each
compressor's ``wkv`` + ``wgate`` -> ``wkv_wgate``. The rows are concatenated
after the storage conversion, the ``weight_scale`` block rows with them: every
fp8 projection's N is a multiple of the 128-row scale block, so block ``i`` of
the second projection lands at block ``N1 / 128 + i``, as its rows do.

The adapter keys on the *stored* dtypes, so a dequantized f32 copy of the
checkpoint passes through untouched and drives the dense f32 modules.

Note the scale rename is a suffix match on ``.scale`` and must stay one: the
mHC mixing parameters (``hc_attn_scale``, ``hc_ffn_scale``, ``hc_head_scale``)
are underscore-suffixed tensors of their own, not scales of a ``weight``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
from max.driver import Buffer, DLPackArray, bfloat16_as_uint16_to_float32
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

# ``<block>.hc_<attn|ffn>_<fn|base|scale>``; ``hc_head_*`` is not a
# ``HyperConnection`` and keeps its name.
_HC_SITE_TENSOR = re.compile(
    r"^(?P<prefix>.*)\.hc_(?P<site>attn|ffn)_(?P<part>fn|base|scale)$"
)

GATE_UP_PROJ = "gate_up_proj"
DOWN_PROJ = "down_proj"

# bf16 in the checkpoint, float32 in the reference (and here), which upcasts
# them on load: the compressor's raw projections (it pools in float32), every
# norm weight (``*norm.weight``).
_FLOAT32_WEIGHTS = (
    ".compressor.wkv.weight",
    ".compressor.wgate.weight",
    "norm.weight",
)

_FP8_WEIGHT_BLOCK = 128

# ``(first, second, stacked)`` module-name suffixes; ``stacked`` is
# ``[first; second]`` along N. ``.attn.wkv`` does not match the compressors'
# ``.compressor.wkv``.
_STACKED_PROJECTIONS = (
    (".attn.wq_a", ".attn.wkv", ".attn.wqa_wkv"),
    (".shared_experts.w1", ".shared_experts.w3", ".shared_experts.w13"),
    (".compressor.wkv", ".compressor.wgate", ".compressor.wkv_wgate"),
)


def as_uint8(data: DLPackArray) -> np.ndarray:
    """The raw bytes of a one-byte-per-element tensor (fp8, e8m0, int8)."""
    buffer = data if isinstance(data, Buffer) else Buffer.from_dlpack(data)
    return np.from_dlpack(buffer.view(DType.uint8))


def e8m0_to_float32(raw: np.ndarray) -> np.ndarray:
    """``2 ** (e - 127)``; exact in float32 for every e8m0 code but NaN (255)."""
    return np.exp2(raw.astype(np.float32) - 127.0)


def with_dtype(array: np.ndarray, dtype: DType, name: str) -> WeightData:
    """``WeightData`` whose bytes are ``array`` reinterpreted as ``dtype``."""
    buffer = Buffer.from_numpy(np.ascontiguousarray(array))
    if dtype != DType.from_numpy(array.dtype):
        buffer = buffer.view(dtype)
    return WeightData(buffer, name, dtype, Shape(buffer.shape))


def bfloat16_to_float32(weight: WeightData, name: str) -> WeightData:
    """A bf16 weight widened to float32 (exact: bf16 is f32's top 16 bits)."""
    buffer = (
        weight.data
        if isinstance(weight.data, Buffer)
        else Buffer.from_dlpack(weight.data)
    )
    widened = bfloat16_as_uint16_to_float32(
        np.from_dlpack(buffer.view(DType.uint16)).view(np.uint16)
    )
    return with_dtype(widened, DType.float32, name)


def _row_bytes(weight: WeightData) -> np.ndarray:
    """``weight``'s bytes as ``[N, row_bytes]`` uint8, whatever its dtype."""
    buffer = (
        weight.data
        if isinstance(weight.data, Buffer)
        else Buffer.from_dlpack(weight.data)
    )
    rows = int(weight.shape[0])
    width = buffer.num_elements * buffer.element_size // rows
    return np.from_dlpack(buffer.view(DType.uint8, [rows, width]))


def concat_rows(first: WeightData, second: WeightData, name: str) -> WeightData:
    """``[first; second]`` along axis 0, bytes copied unchanged."""
    dims = [[int(d) for d in w.shape] for w in (first, second)]
    if first.dtype != second.dtype or dims[0][1:] != dims[1][1:]:
        raise ValueError(
            f"{name}: cannot stack {first.name} {first.dtype} {dims[0]} "
            f"with {second.name} {second.dtype} {dims[1]}"
        )
    joined = np.concatenate([_row_bytes(first), _row_bytes(second)], axis=0)
    shape = [dims[0][0] + dims[1][0], *dims[0][1:]]
    buffer = Buffer.from_numpy(joined).view(first.dtype, shape)
    return WeightData(buffer, name, first.dtype, Shape(shape))


def stack_projections(state_dict: dict[str, WeightData]) -> None:
    """Replaces each :data:`_STACKED_PROJECTIONS` pair with its stack, in place.

    ``weight`` always, ``weight_scale`` when the pair is fp8.
    """
    for first, second, stacked in _STACKED_PROJECTIONS:
        suffix = f"{first}.weight"
        prefixes = [n[: -len(suffix)] for n in state_dict if n.endswith(suffix)]
        for prefix in prefixes:
            n_first = int(state_dict[prefix + suffix].shape[0])
            for part in ("weight", "weight_scale"):
                a = f"{prefix}{first}.{part}"
                b = f"{prefix}{second}.{part}"
                if a not in state_dict and b not in state_dict:
                    continue
                if a not in state_dict or b not in state_dict:
                    raise ValueError(f"{a} and {b} must both be present")
                if (
                    part == "weight_scale"
                    and int(state_dict[a].shape[0]) * _FP8_WEIGHT_BLOCK
                    != n_first
                ):
                    # The second projection's scale rows would land on the
                    # wrong weight rows.
                    raise ValueError(
                        f"{a}: {n_first} rows are not whole "
                        f"{_FP8_WEIGHT_BLOCK}-row scale blocks"
                    )
                out = f"{prefix}{stacked}.{part}"
                state_dict[out] = concat_rows(
                    state_dict.pop(a), state_dict.pop(b), out
                )


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
        hc_site = _HC_SITE_TENSOR.match(name)
        if hc_site is not None:
            new_state_dict[
                f"{hc_site['prefix']}.hc_{hc_site['site']}.hc_{hc_site['part']}"
            ] = data
            continue

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

        if name.endswith(_FLOAT32_WEIGHTS) and data.dtype == DType.bfloat16:
            new_state_dict[name] = bfloat16_to_float32(data, name)
            continue

        new_state_dict[name] = data

    for prefix, experts in packed_experts.items():
        new_state_dict.update(stack_experts(experts, prefix))
    stack_projections(new_state_dict)
    return new_state_dict


_DSPARK_PREFIX = "mtp."
# Read only by adaptive verification, which the draft does not run.
_UNUSED_DSPARK = (".confidence_head.",)


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """The trunk's weights; the DSpark stages (``mtp.*``) are dropped.

    For the non-speculative graph, which does not build the stages.
    """
    return convert_weight_data(
        {
            name: value.data()
            for name, value in state_dict.items()
            if not name.startswith(_DSPARK_PREFIX)
        }
    )


def convert_dspark_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: PretrainedConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """The DSpark stages' weights (``mtp.*``), under the trunk's rules.

    For the draft beside the trunk, which loads the same checkpoint through
    :func:`convert_safetensor_state_dict`. The stages' projections need no
    rule of their own: ``mtp.0.main_proj`` is an fp8 weight with a 128x128
    block scale like the attention projections, in the minimized checkpoint
    as in the real one.
    """
    return convert_weight_data(
        {
            name: value.data()
            for name, value in state_dict.items()
            if name.startswith(_DSPARK_PREFIX)
            and not any(part in name for part in _UNUSED_DSPARK)
        }
    )
