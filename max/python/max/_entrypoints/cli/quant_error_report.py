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
"""Reads the quantization-error dumps a ``max quantize`` run leaves behind.

With an error-report directory set, each quantized weight prints
``[sum_sq_error, sum_sq_source, max_abs_error]`` from inside the compiled
graph as ``quant_error.<weight name>.max``. This turns those files into a
table and a JSON report. Everything here works on the files alone, so a
report can be rebuilt later without the model (``max quantize-report``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from max.driver.buffer import load_max_buffer

PREFIX = "quant_error."
SUFFIX = ".max"
JSON_NAME = "quant_error_report.json"


@dataclass(frozen=True)
class LayerError:
    """Quantization error of one weight, as the graph measured it."""

    label: str
    sum_sq_error: float
    sum_sq_source: float
    max_abs_error: float

    @property
    def relative_error(self) -> float:
        """``||Q(W) - W|| / ||W||``; 0 for an all-zero weight."""
        if self.sum_sq_source <= 0.0:
            return 0.0
        return math.sqrt(self.sum_sq_error / self.sum_sq_source)

    @property
    def snr_db(self) -> float:
        """Signal-to-quantization-noise ratio in dB; ``inf`` when lossless."""
        if self.sum_sq_error <= 0.0:
            return math.inf
        if self.sum_sq_source <= 0.0:
            return -math.inf
        return 10.0 * math.log10(self.sum_sq_source / self.sum_sq_error)


def clear_error_dumps(directory: Path) -> None:
    """Creates ``directory`` and removes the dumps an earlier run left in it.

    Run before the graph, so the report holds only this run's weights. Only
    ``quant_error.*.max`` files are removed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob(f"{PREFIX}*{SUFFIX}"):
        path.unlink()


def read_error_dumps(directory: Path) -> list[LayerError]:
    """Loads every ``quant_error.*.max`` dump in ``directory``.

    Returns:
        One entry per weight, ordered by label.
    """
    layers = []
    for path in sorted(directory.glob(f"{PREFIX}*{SUFFIX}")):
        label = path.name[len(PREFIX) : -len(SUFFIX)]
        values = load_max_buffer(path).to_numpy().reshape(-1)
        if values.shape != (3,):
            raise ValueError(
                f"{path.name} holds {values.shape[0]} values, expected 3 "
                "(sum_sq_error, sum_sq_source, max_abs_error)"
            )
        layers.append(
            LayerError(
                label=label,
                sum_sq_error=float(values[0]),
                sum_sq_source=float(values[1]),
                max_abs_error=float(values[2]),
            )
        )
    return layers


def format_table(layers: Iterable[LayerError]) -> str:
    """Renders the layers as a table, worst relative error first."""
    rows = sorted(layers, key=lambda layer: -layer.relative_error)
    if not rows:
        return "No quantization-error dumps found."
    width = max(len("layer"), *(len(layer.label) for layer in rows))
    header = (
        f"{'layer':<{width}}  {'rel_error':>10}  {'snr_db':>8}  {'max_abs':>10}"
    )
    lines = [header, "-" * len(header)]
    for layer in rows:
        lines.append(
            f"{layer.label:<{width}}  {layer.relative_error:>10.4e}  "
            f"{layer.snr_db:>8.2f}  {layer.max_abs_error:>10.4e}"
        )
    return "\n".join(lines)


def write_json(layers: Iterable[LayerError], path: Path) -> None:
    """Writes the layers, with derived metrics, as a JSON report."""
    entries = []
    for layer in layers:
        entry = asdict(layer)
        entry["relative_error"] = layer.relative_error
        # JSON has no infinity; a lossless layer reports null.
        entry["snr_db"] = None if math.isinf(layer.snr_db) else layer.snr_db
        entries.append(entry)
    path.write_text(json.dumps({"layers": entries}, indent=2) + "\n")


def write_report(directory: Path) -> list[LayerError]:
    """Prints the table for the dumps in ``directory`` and writes the JSON.

    Needs only the dump files, not the model or a GPU.

    Returns:
        The layers that were reported.
    """
    layers = read_error_dumps(directory)
    print(format_table(layers))
    write_json(layers, directory / JSON_NAME)
    return layers
