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

NVFP4_GROUP_SIZE = 16
"""Inputs covered by one NVFP4 block scale."""


class ModuleFormat(enum.Enum):
    """How a module's weight is stored in the checkpoint."""

    BF16 = "bf16"
    """Not quantized."""

    FP8_STATIC_TENSOR = "fp8_static_tensor"
    """E4M3 weight with static per-tensor weight and input scales."""

    NVFP4_WEIGHT_ONLY = "nvfp4_weight_only"
    """Packed E2M1 weight, an E4M3 scale per 16 inputs and a float32 global
    scale. There is no input scale: activations stay BF16."""


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

    def has_nvfp4_routed_experts(self, mixer: str, num_experts: int) -> bool:
        """Returns whether every routed expert projection of a mixer is NVFP4.

        Args:
            mixer: The mixer's checkpoint path, ``backbone.layers.1.mixer``.
            num_experts: The number of routed experts.
        """
        return all(
            self.format_of(f"{mixer}.experts.{e}.{proj}")
            is ModuleFormat.NVFP4_WEIGHT_ONLY
            for e in range(num_experts)
            for proj in ("up_proj", "down_proj")
        )

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
        NotImplementedError: If the config uses an algorithm Nemotron-H does
            not read.
        ValueError: If a ``MIXED_PRECISION`` config has no
            ``quantized_layers`` map.
    """
    if not hf_quant_config:
        return NemotronHQuantScheme(quantized={})
    quant_algo = hf_quant_config.get("quant_algo")
    if quant_algo != "MIXED_PRECISION":
        raise NotImplementedError(
            f"Nemotron-H cannot read quant_algo {quant_algo!r}; only "
            "modelopt 'MIXED_PRECISION' checkpoints are supported."
        )
    quantized_layers = hf_quant_config.get("quantized_layers")
    if not isinstance(quantized_layers, Mapping) or not quantized_layers:
        raise ValueError(
            "quant_algo 'MIXED_PRECISION' needs a 'quantized_layers' map "
            "naming each module's algorithm, and the config has none"
        )
    quantized: dict[str, ModuleFormat] = {}
    for module, entry in quantized_layers.items():
        algo = entry.get("quant_algo") if isinstance(entry, Mapping) else None
        fmt = _ALGO_FORMATS.get(algo) if isinstance(algo, str) else None
        if fmt is None:
            raise NotImplementedError(
                f"'{module}' is quantized as {algo!r}; Nemotron-H reads "
                f"{sorted(_ALGO_FORMATS)} only."
            )
        if fmt is ModuleFormat.NVFP4_WEIGHT_ONLY:
            group_size = entry.get("group_size")
            if group_size != NVFP4_GROUP_SIZE:
                raise NotImplementedError(
                    f"'{module}' declares NVFP4 group_size {group_size!r}; "
                    f"only {NVFP4_GROUP_SIZE} is supported."
                )
        quantized[str(module)] = fmt
    return NemotronHQuantScheme(quantized=quantized)
