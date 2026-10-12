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
"""Per-module quantization of Nemotron-H checkpoints.

The scale tensors beside each weight are checked against the module's declared
format, because a module read in the wrong format loads without error and
produces wrong logits.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from max.dtype import DType
from max.nn.quant_config import (
    NVFP4_BLOCK_SIZE,
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.pipelines.weights.quant import read_modelopt_quantized_layers

NVFP4_GROUP_SIZE = NVFP4_BLOCK_SIZE
"""Inputs covered by one NVFP4 block scale."""


class Parallelism(enum.Enum):
    """How a linear layer's weight is split across the tensor-parallel axis."""

    REPLICATED = "replicated"
    """Every device holds the whole weight."""

    COLUMN = "column"
    """Each device holds a share of the output rows."""

    ROW = "row"
    """Each device holds a share of the input columns, and the outputs are
    partial sums."""


# The checkpoint's dense linear modules, by name suffix, and how each is
# split. Routed experts are stacked rather than built as linear layers.
_LINEAR_PARALLELISM: dict[str, Parallelism] = {
    ".mixer.in_proj": Parallelism.COLUMN,
    ".mixer.out_proj": Parallelism.ROW,
    ".shared_experts.up_proj": Parallelism.COLUMN,
    ".shared_experts.down_proj": Parallelism.ROW,
    # The MLP mixers and the LM head are replicated.
    ".mixer.up_proj": Parallelism.REPLICATED,
    ".mixer.down_proj": Parallelism.REPLICATED,
}


def linear_parallelism(module: str) -> Parallelism:
    """Returns how a quantizable dense linear module is split.

    Args:
        module: The module's checkpoint path.

    Raises:
        NotImplementedError: If the module is not built as a quantizable
            linear layer, such as an attention projection.
    """
    if module == "lm_head":
        return Parallelism.REPLICATED
    for suffix, parallelism in _LINEAR_PARALLELISM.items():
        if module.endswith(suffix):
            return parallelism
    raise NotImplementedError(
        f"'{module}' is quantized, but Nemotron-H builds it in BF16 only"
    )


class ModuleFormat(enum.Enum):
    """How a module's weight is stored in the checkpoint."""

    BF16 = "bf16"
    """Not quantized."""

    FP8_STATIC_TENSOR = "fp8_static_tensor"
    """E4M3 weight with static per-tensor weight and input scales."""

    NVFP4_WEIGHT_ONLY = "nvfp4_weight_only"
    """Packed E2M1 weight, an E4M3 scale per 16 inputs and a float32 global
    scale. There is no input scale: activations stay BF16."""


FP8_STATIC_TENSOR_QUANT = QuantConfig(
    input_scale=InputScaleSpec(
        granularity=ScaleGranularity.TENSOR,
        origin=ScaleOrigin.STATIC,
        dtype=DType.float32,
    ),
    weight_scale=WeightScaleSpec(
        granularity=ScaleGranularity.TENSOR, dtype=DType.float32
    ),
    mlp_quantized_layers=set(),
    attn_quantized_layers=set(),
    format=QuantFormat.COMPRESSED_TENSORS_FP8,
)
"""The quant config of a :attr:`ModuleFormat.FP8_STATIC_TENSOR` linear."""

_ALGO_FORMATS: dict[str, ModuleFormat] = {
    "FP8": ModuleFormat.FP8_STATIC_TENSOR,
    "W4A16_NVFP4": ModuleFormat.NVFP4_WEIGHT_ONLY,
}

_SCALES: dict[ModuleFormat, frozenset[str]] = {
    ModuleFormat.BF16: frozenset(),
    ModuleFormat.FP8_STATIC_TENSOR: frozenset({"weight_scale", "input_scale"}),
    ModuleFormat.NVFP4_WEIGHT_ONLY: frozenset(
        {"weight_scale", "weight_scale_2"}
    ),
}


@dataclass(frozen=True)
class NemotronHQuantScheme:
    """The storage format of every module of a Nemotron-H checkpoint."""

    quantized: Mapping[str, ModuleFormat]
    """Format of each quantized module, keyed by its checkpoint path
    (``backbone.layers.0.mixer.in_proj``, ``lm_head``, ...). Every other
    module is BF16."""

    def format_of(self, module: str) -> ModuleFormat:
        """Returns the format ``module`` is stored in."""
        return self.quantized.get(module, ModuleFormat.BF16)

    def routed_experts_format(
        self, mixer: str, num_experts: int
    ) -> ModuleFormat:
        """Returns the one format every routed expert of a mixer is stored in.

        Args:
            mixer: The mixer's checkpoint path, ``backbone.layers.1.mixer``.
            num_experts: The number of routed experts.

        Raises:
            NotImplementedError: If the experts are FP8, or stored in more
                than one format. The grouped matmul reads all of a mixer's
                experts in one format, BF16 or NVFP4.
        """
        formats = {
            self.format_of(f"{mixer}.experts.{e}.{proj}")
            for e in range(num_experts)
            for proj in ("up_proj", "down_proj")
        }
        if len(formats) != 1 or ModuleFormat.FP8_STATIC_TENSOR in formats:
            raise NotImplementedError(
                f"'{mixer}' stores its routed experts as "
                f"{sorted(f.name for f in formats)}; Nemotron-H reads them "
                "all BF16 or all NVFP4"
            )
        return formats.pop()

    def check_dense_modules(self) -> None:
        """Checks that every quantized dense module is one built quantized.

        Raises:
            NotImplementedError: If a quantized module, such as an attention
                projection, is built in BF16 only.
        """
        for module in self.quantized:
            if ".experts." not in module:
                linear_parallelism(module)

    def check_weights(self, weight_names: Collection[str]) -> None:
        """Checks the checkpoint's tensors against the declared formats.

        Args:
            weight_names: Every tensor name in the checkpoint.

        Raises:
            ValueError: If a quantized module has no weight, or a module's
                scale tensors do not match its format.
        """
        names = set(weight_names)
        missing = sorted(
            m for m in self.quantized if f"{m}.weight" not in names
        )
        if missing:
            raise ValueError(
                f"quantized_layers names {len(missing)} modules with no weight "
                f"in the checkpoint, for example '{missing[0]}'"
            )
        scale_names = frozenset().union(*_SCALES.values())
        for name in names:
            if not name.endswith(".weight"):
                continue
            module = name.removesuffix(".weight")
            fmt = self.format_of(module)
            scales = {s for s in scale_names if f"{module}.{s}" in names}
            if scales != _SCALES[fmt]:
                raise ValueError(
                    f"'{module}' is declared {fmt.name} but the checkpoint "
                    f"stores {sorted(scales) or 'no scales'} beside its "
                    f"weight; {fmt.name} needs "
                    f"{sorted(_SCALES[fmt]) or 'none'}."
                )


def parse_quant_scheme(
    hf_quant_config: Mapping[str, object] | None,
) -> NemotronHQuantScheme:
    """Reads a modelopt quantization config into a scheme.

    Args:
        hf_quant_config: The checkpoint's quantization config, or ``None`` for
            an unquantized checkpoint.

    Returns:
        The scheme.

    Raises:
        ValueError: If the config is not modelopt ``MIXED_PRECISION``, has
            no ``quantized_layers`` map, or quantizes a module with an
            algorithm or group size Nemotron-H does not read.
    """
    if not hf_quant_config:
        return NemotronHQuantScheme(quantized={})
    modules = read_modelopt_quantized_layers(
        hf_quant_config,
        allowed={("FP8", None), ("W4A16_NVFP4", NVFP4_GROUP_SIZE)},
        model_name="Nemotron-H",
    )
    return NemotronHQuantScheme(
        quantized={
            module: _ALGO_FORMATS[quant.quant_algo]
            for module, quant in modules.items()
        }
    )
