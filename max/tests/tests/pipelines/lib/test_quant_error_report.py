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

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from max._entrypoints.cli import quant_error_report as report
from max._entrypoints.cli.quant_error_report import LayerError


def test_relative_error_and_snr() -> None:
    layer = LayerError(
        "a", sum_sq_error=1.0, sum_sq_source=100.0, max_abs_error=0.5
    )
    assert layer.relative_error == pytest.approx(0.1)
    assert layer.snr_db == pytest.approx(20.0)


def test_degenerate_layers() -> None:
    lossless = LayerError("a", 0.0, 4.0, 0.0)
    assert lossless.relative_error == 0.0
    assert math.isinf(lossless.snr_db)
    zero = LayerError("b", 0.0, 0.0, 0.0)
    assert zero.relative_error == 0.0


def test_table_lists_worst_layer_first() -> None:
    table = report.format_table(
        [
            LayerError("layers.0.mlp.down_proj", 1.0, 100.0, 0.1),
            LayerError("layers.1.mlp.down_proj", 9.0, 100.0, 0.2),
        ]
    )
    lines = table.splitlines()
    assert "layers.1.mlp.down_proj" in lines[2]
    assert "layers.0.mlp.down_proj" in lines[3]


def test_table_without_dumps_says_so() -> None:
    assert "No quantization-error dumps" in report.format_table([])


def test_json_report_maps_lossless_snr_to_null(tmp_path: Path) -> None:
    out = tmp_path / report.JSON_NAME
    report.write_json(
        [LayerError("a", 1.0, 100.0, 0.5), LayerError("b", 0.0, 1.0, 0.0)], out
    )
    layers = json.loads(out.read_text())["layers"]
    assert layers[0]["label"] == "a"
    assert layers[0]["relative_error"] == pytest.approx(0.1)
    assert layers[1]["snr_db"] is None


def _fake_buffers(
    monkeypatch: pytest.MonkeyPatch, values: dict[str, list[float]]
) -> None:
    """Makes ``load_max_buffer`` return the values registered for a file name."""

    def load(path: Path) -> SimpleNamespace:
        array = np.array(values[Path(path).name], dtype=np.float32)
        return SimpleNamespace(to_numpy=lambda: array)

    monkeypatch.setattr(report, "load_max_buffer", load)


def test_read_error_dumps_recovers_labels_with_dots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "quant_error.language_model.layers.3.mlp.down_proj.max"
    (tmp_path / name).write_bytes(b"")
    (tmp_path / "unrelated.max").write_bytes(b"")
    _fake_buffers(monkeypatch, {name: [4.0, 400.0, 0.25]})

    (layer,) = report.read_error_dumps(tmp_path)
    assert layer == LayerError(
        "language_model.layers.3.mlp.down_proj", 4.0, 400.0, 0.25
    )


def test_clear_error_dumps_removes_only_earlier_dumps(tmp_path: Path) -> None:
    out = tmp_path / "report"
    out.mkdir()
    (out / "quant_error.old.max").write_bytes(b"")
    (out / "unrelated.max").write_bytes(b"")
    (out / report.JSON_NAME).write_text("{}")

    report.clear_error_dumps(out)

    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["unrelated.max", report.JSON_NAME]
    )


def test_clear_error_dumps_creates_the_directory(tmp_path: Path) -> None:
    out = tmp_path / "new" / "report"
    report.clear_error_dumps(out)
    assert out.is_dir()


def test_write_report_prints_the_table_and_writes_the_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    name = "quant_error.layers.0.mlp.down_proj.weight.max"
    (tmp_path / name).write_bytes(b"")
    _fake_buffers(monkeypatch, {name: [4.0, 400.0, 0.25]})

    layers = report.write_report(tmp_path)

    assert [layer.label for layer in layers] == [
        "layers.0.mlp.down_proj.weight"
    ]
    assert "layers.0.mlp.down_proj.weight" in capsys.readouterr().out
    saved = json.loads((tmp_path / report.JSON_NAME).read_text())
    assert saved["layers"][0]["label"] == "layers.0.mlp.down_proj.weight"


def test_read_error_dumps_rejects_a_malformed_dump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "quant_error.bad.max"
    (tmp_path / name).write_bytes(b"")
    _fake_buffers(monkeypatch, {name: [1.0, 2.0]})
    with pytest.raises(ValueError, match="expected 3"):
        report.read_error_dumps(tmp_path)
