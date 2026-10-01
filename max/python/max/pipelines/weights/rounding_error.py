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
"""Reports dequant vs source rounding error for an MXFP6 or NVFP4 checkpoint.

Name-driven: only tensors that have scale keys are compared. Each pair is
loaded on its own, so peak memory is one weight, not a shard. Used as a
standalone CLI and from :mod:`quantize_checkpoint` during export.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch  # type: ignore
from max.pipelines.weights.fp4_quantization import (
    FP4Format,
    dequantize_nvfp4,
)
from max.pipelines.weights.fp6_quantization import (
    FP6Format,
    dequantize_mxfp6,
)
from numpy.typing import NDArray
from safetensors.torch import safe_open

logger = logging.getLogger("max.pipelines")

QuantFormat = FP4Format | FP6Format

_WEIGHT_INDEX = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class TensorError:
    """Rounding error for one dequantized weight against its source."""

    name: str
    rel_l2: float
    max_abs: float


@dataclass(frozen=True, slots=True)
class CheckpointErrorReport:
    """Per-tensor errors plus the aggregates the CLI and exporter log."""

    tensors: tuple[TensorError, ...]

    @property
    def mean_rel_l2(self) -> float:
        """Mean relative L2 over compared tensors, or 0 if none."""
        if not self.tensors:
            return 0.0
        return sum(item.rel_l2 for item in self.tensors) / len(self.tensors)

    @property
    def max_rel_l2(self) -> float:
        """Largest relative L2, or 0 if none."""
        return max((item.rel_l2 for item in self.tensors), default=0.0)

    def worst(self, n: int = 5) -> tuple[TensorError, ...]:
        """The ``n`` tensors with the largest relative L2."""
        return tuple(
            sorted(self.tensors, key=lambda item: item.rel_l2, reverse=True)[:n]
        )


def compare_arrays(
    name: str,
    reference: NDArray[np.floating],
    reconstructed: NDArray[np.floating],
) -> TensorError:
    """Relative L2 and max-abs of ``reconstructed - reference``."""
    ref = np.ascontiguousarray(reference, dtype=np.float32)
    recon = np.ascontiguousarray(reconstructed, dtype=np.float32)
    if ref.shape != recon.shape:
        raise ValueError(
            f"{name}: dequant shape {recon.shape} != source {ref.shape}"
        )
    diff = recon - ref
    ref_norm = float(np.linalg.norm(ref))
    rel_l2 = 0.0 if ref_norm == 0.0 else float(np.linalg.norm(diff) / ref_norm)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    return TensorError(name=name, rel_l2=rel_l2, max_abs=max_abs)


def dequantize_written(
    name: str, tensors: Mapping[str, torch.Tensor], fmt: QuantFormat
) -> NDArray[np.float32]:
    """Reconstructs float32 from the checkpoint keys of one quantized weight."""
    packed = _codes_uint8(tensors[name])
    scales = _codes_uint8(tensors[f"{name}_scale"])
    if isinstance(fmt, FP4Format):
        # Export squeezes the trailing singleton so on-disk scales are
        # ``(N, K/16)``; dequant broadcasts against ``(N, K/16, 16)``.
        if scales.ndim == packed.ndim:
            scales = scales[..., None]
        scale_2 = float(
            tensors[f"{name}_scale_2"]
            .cpu()
            .to(torch.float32)
            .numpy()
            .reshape(-1)[0]
        )
        return dequantize_nvfp4(packed, scales, scale_2, fmt)
    return dequantize_mxfp6(packed, scales, fmt)


def compare_written(
    name: str,
    reference: NDArray[np.floating],
    tensors: Mapping[str, torch.Tensor],
    fmt: QuantFormat,
) -> TensorError:
    """Dequants ``tensors`` and compares them to the source values."""
    return compare_arrays(
        name, reference, dequantize_written(name, tensors, fmt)
    )


def format_report(report: CheckpointErrorReport, *, worst: int = 5) -> str:
    """One-line summary plus the worst offenders."""
    if not report.tensors:
        return "rounding error: no quantized tensors compared"
    lines = [
        f"rounding error ({len(report.tensors)} tensors): "
        f"mean rel L2={report.mean_rel_l2:.4f}, "
        f"max rel L2={report.max_rel_l2:.4f}"
    ]
    for item in report.worst(worst):
        lines.append(
            f"  {item.name}: rel L2={item.rel_l2:.4f} max abs={item.max_abs:.4g}"
        )
    return "\n".join(lines)


def infer_format(dst: Path, explicit: str | None = None) -> QuantFormat:
    """Resolves the encoding from ``--format`` or dest ``config.json``."""
    if explicit:
        return _parse_format(explicit)
    config_path = dst / "config.json"
    if not config_path.exists():
        raise ValueError(
            f"no --format and no {config_path}; pass --format explicitly"
        )
    config = json.loads(config_path.read_text())
    qc = config.get("quantization_config", {})
    method = qc.get("quant_method")
    if method == "nvfp4" or qc.get("quant_algo") == "NVFP4":
        return FP4Format.NVFP4
    if method == "mxfp6":
        return FP6Format(qc["fp6_format"])
    # ModelOpt mixed dumps (e.g. nvidia/MiniMax-M3-NVFP4) set
    # quant_algo=MIXED_PRECISION; NVFP4 tensors still have weight_scale_2.
    if _config_has_nvfp4(qc):
        return FP4Format.NVFP4
    raise ValueError(
        f"cannot infer format from {config_path} "
        f"(quant_method={method!r}); pass --format"
    )


def quantized_weight_names(keys: Sequence[str], fmt: QuantFormat) -> list[str]:
    """Weight names that have the scale keys this encoding writes."""
    present = set(keys)
    names: list[str] = []
    for key in present:
        if key.endswith(("_scale", "_scale_2")):
            continue
        if f"{key}_scale" not in present:
            continue
        if isinstance(fmt, FP4Format) and f"{key}_scale_2" not in present:
            continue
        names.append(key)
    return sorted(names)


def compare_checkpoints(
    src: Path,
    dst: Path,
    fmt: QuantFormat | None = None,
    *,
    targets: Sequence[str] = (),
) -> CheckpointErrorReport:
    """Compares every quantized dest weight to the same name in ``src``.

    Args:
        src: The bf16 (or other float) source checkpoint directory.
        dst: The MXFP6 or NVFP4 checkpoint directory.
        fmt: Encoding to decode. Inferred from dest ``config.json`` if omitted.
        targets: Optional regexes; when set, only matching names are compared.

    Returns:
        One :class:`TensorError` per compared weight.
    """
    resolved = fmt if fmt is not None else infer_format(dst)
    dest_index = _tensor_index(dst)
    source_index = _tensor_index(src)
    names = quantized_weight_names(list(dest_index), resolved)
    if targets:
        patterns = [re.compile(p) for p in targets]
        names = [n for n in names if any(p.search(n) for p in patterns)]
    total = len(names)
    logger.info("comparing %d quantized tensors", total)

    errors: list[TensorError] = []
    for i, name in enumerate(names, start=1):
        dest_shard = dest_index[name]
        source_shard = source_index.get(name)
        if source_shard is None:
            logger.warning(
                "%s is quantized in dest but missing from %s", name, src
            )
            continue
        written = _load_written(dest_shard, name, resolved)
        reference = _load_reference(source_shard, name)
        error = compare_written(name, reference, written, resolved)
        errors.append(error)
        if i == 1 or i == total or i % 100 == 0:
            logger.info(
                "%d/%d %s rel L2=%.4f",
                i,
                total,
                name,
                error.rel_l2,
            )
    return CheckpointErrorReport(tensors=tuple(errors))


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the rounding-error checker from the command line."""
    parser = argparse.ArgumentParser(
        prog="check_rounding_error",
        description=(
            "Compare an MXFP6 or NVFP4 checkpoint to its bf16 source. "
            "Only tensors with scale keys are read."
        ),
    )
    parser.add_argument(
        "src", type=Path, help="source (bf16) checkpoint directory"
    )
    parser.add_argument("dst", type=Path, help="quantized checkpoint directory")
    parser.add_argument(
        "--format",
        choices=[f.value for f in (*FP6Format, *FP4Format)],
        default=None,
        help="Element encoding. Inferred from dest config.json when omitted.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="REGEX",
        help=(
            "only compare tensor names matching this regex (repeatable). "
            "On a mixed ModelOpt dump, restrict to NVFP4 experts, e.g. "
            r"'block_sparse_moe\.experts\.\d+\.w[123]\.weight$'"
        ),
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=5,
        metavar="N",
        help="how many worst tensors to print (default 5)",
    )
    parser.add_argument(
        "--max-rel-l2",
        type=float,
        default=None,
        metavar="LIMIT",
        help="exit 1 if any tensor's relative L2 exceeds LIMIT",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    fmt = infer_format(args.dst, args.format)
    report = compare_checkpoints(args.src, args.dst, fmt, targets=args.target)
    logger.info("%s", format_report(report, worst=args.worst))
    if args.max_rel_l2 is not None and report.max_rel_l2 > args.max_rel_l2:
        logger.error(
            "max rel L2 %.4f exceeds --max-rel-l2 %s",
            report.max_rel_l2,
            args.max_rel_l2,
        )
        return 1
    return 0


def _config_has_nvfp4(qc: Mapping[str, object]) -> bool:
    """True if a ModelOpt mixed config still lists NVFP4 layers."""
    layers = qc.get("quantized_layers")
    if not isinstance(layers, dict):
        return False
    for value in layers.values():
        algo = value.get("quant_algo") if isinstance(value, dict) else value
        if algo == "NVFP4":
            return True
    return False


def _codes_uint8(tensor: torch.Tensor) -> NDArray[np.uint8]:
    """Returns packed or E4M3 codes as uint8.

    ModelOpt stores E4M3 block scales as ``float8_e4m3fn``; numpy cannot
    convert that dtype, so the bits are viewed as uint8.
    """
    cpu = tensor.detach().cpu()
    if cpu.dtype == torch.uint8:
        return np.ascontiguousarray(cpu.numpy())
    if cpu.dtype == torch.float8_e4m3fn:
        return np.ascontiguousarray(cpu.view(torch.uint8).numpy())
    raise TypeError(f"expected uint8 or float8_e4m3fn codes, got {cpu.dtype}")


def _parse_format(value: str) -> QuantFormat:
    """Resolves a CLI format string to an FP4 or FP6 encoding."""
    for enum in (FP4Format, FP6Format):
        try:
            return enum(value)
        except ValueError:
            continue
    raise ValueError(f"unknown quantization format {value!r}")


def _tensor_index(checkpoint: Path) -> dict[str, Path]:
    """Maps each tensor name to the shard that holds it."""
    index_path = checkpoint / _WEIGHT_INDEX
    if index_path.exists():
        mapping = json.loads(index_path.read_text())["weight_map"]
        return {name: checkpoint / shard for name, shard in mapping.items()}

    found: dict[str, Path] = {}
    shards = sorted(checkpoint.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(
            f"no safetensors shards found under {checkpoint}"
        )
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118
                found[name] = shard
    return found


def _load_written(
    shard: Path, name: str, fmt: QuantFormat
) -> dict[str, torch.Tensor]:
    """Loads the packed weight and its scale keys from one dest shard."""
    keys = [name, f"{name}_scale"]
    if isinstance(fmt, FP4Format):
        keys.append(f"{name.removesuffix('.weight')}.input_scale")
        keys.append(f"{name}_scale_2")
    with safe_open(shard, framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in keys}


def _load_reference(shard: Path, name: str) -> NDArray[np.float32]:
    """Loads one source weight as float32 (bf16 widens exactly)."""
    with safe_open(shard, framework="pt") as handle:
        tensor = handle.get_tensor(name)
    return tensor.to(torch.float32).numpy()


if __name__ == "__main__":
    sys.exit(main())
