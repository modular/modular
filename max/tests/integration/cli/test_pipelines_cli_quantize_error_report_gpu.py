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
"""``max quantize --error-report-dir`` end to end on real models.

One test, parametrized by quantization format, for the model named by
``$QUANT_MODEL`` (a key of :data:`MODELS`). Each Bazel target pins one model
and the formats its hardware can run through ``$QUANT_FORMATS``
(comma-separated): NVFP4 and MXFP8 on SM100, MXFP4 on AMD CDNA. Running the
same model under each format is what makes the reports comparable.

Why these models fit the formats: the SM100 MXFP8 kernel needs the contraction
dimension to be a multiple of 128, and every targeted weight here satisfies
it (Gemma-3 1B: 1024 for ``o_proj``, 6912 for ``down_proj``; OLMo-2 1B: 2048
and 8192).

The table is printed into the test log (``--test_output=all``), headed
``=== <repo> / <format> ===``. Each run's report directory is
``$TEST_UNDECLARED_OUTPUTS_DIR/quant_error/<model>/<format>``, which
BuildBuddy lists under the invocation's Artifacts tab.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from max._entrypoints import pipelines
from max._entrypoints.cli.quant_error_report import JSON_NAME


@dataclass(frozen=True)
class _Model:
    repo_id: str
    gated: bool
    # Fraction of the memory pool the model may use. CI sizes the pool to the
    # test's gpu-memory resource, so it must clear the weights, the measured
    # transients and a 1 GiB margin with room for one prefill chunk of KV.
    memory_utilization: float


MODELS = {
    # 1.86 GiB of weights.
    "gemma3_1b": _Model(
        "google/gemma-3-1b-it", gated=True, memory_utilization=0.9
    ),
    # About 3 GB of weights.
    "olmo2_1b": _Model(
        "allenai/OLMo-2-0425-1B-Instruct", gated=False, memory_utilization=0.9
    ),
}

MODEL_KEY = os.environ.get("QUANT_MODEL", "gemma3_1b")
FORMATS = tuple(
    name for name in os.environ.get("QUANT_FORMATS", "nvfp4").split(",") if name
)
# Dumps are named after the weight, so each label ends in ``.weight``.
_TARGET_SUFFIXES = ("mlp.down_proj.weight", "self_attn.o_proj.weight")


def _report_dir(tmp_path: Path, fmt: str) -> Path:
    undeclared = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    base = Path(undeclared) / "quant_error" if undeclared else tmp_path
    return base / MODEL_KEY / fmt


@pytest.mark.parametrize("fmt", FORMATS)
def test_quantize_error_report_lists_every_targeted_layer(
    fmt: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    model = MODELS[MODEL_KEY]
    if model.gated and not os.environ.get("HF_TOKEN"):
        pytest.skip(f"{model.repo_id} is gated; set HF_TOKEN")
    out_dir = _report_dir(tmp_path, fmt)
    try:
        pipelines.main(
            [
                "quantize",
                "--model-path",
                model.repo_id,
                "--prefer-module-v3",
                "--trust-remote-code",
                f"--device-memory-utilization={model.memory_utilization}",
                # One short prompt, one token. Left alone the planner sizes
                # the KV cache to fill the memory budget, and on a shared GPU
                # that request can exceed what the allocator can hand out.
                "--max-batch-size=1",
                "--max-length=4096",
                "--max-new-tokens=1",
                "--quantization-encoding=bfloat16",
                "--quantization-format",
                fmt,
                "--devices=gpu",
                "--error-report-dir",
                str(out_dir),
            ]
        )
    except SystemExit as exit_info:
        # click exits through SystemExit even on success.
        assert exit_info.code in (0, None), exit_info.code
    captured = capsys.readouterr()

    # Pytest hides a passing test's output; put the table in the test log.
    with capsys.disabled():
        print(f"\n=== {model.repo_id} / {fmt} ===")
        print(captured.out)

    report_path = out_dir / JSON_NAME
    assert report_path.exists(), (
        f"no report written:\n{(captured.out + captured.err)[-4000:]}"
    )
    layers = json.loads(report_path.read_text())["layers"]

    labels = [layer["label"] for layer in layers]
    assert labels, "the report is empty"
    assert len(labels) == len(set(labels)), "a layer was reported twice"
    for label in labels:
        assert label.endswith(_TARGET_SUFFIXES), label
    # One entry of each kind per transformer layer.
    counts = [
        sum(label.endswith(suffix) for label in labels)
        for suffix in _TARGET_SUFFIXES
    ]
    assert counts[0] == counts[1] > 0, counts

    for layer in layers:
        assert layer["sum_sq_source"] > 0, layer
        assert math.isfinite(layer["relative_error"]), layer
        # Block-scaled 4/8-bit formats are lossy but nowhere near destroying
        # a weight.
        assert 0.0 < layer["relative_error"] < 0.5, layer
