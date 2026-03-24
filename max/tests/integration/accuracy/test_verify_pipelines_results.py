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

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import verify_pipelines
from click.testing import CliRunner

PIXEL_PIPELINE = "black-forest-labs/FLUX.1-dev-bfloat16"
RUNNER = CliRunner()


def _fail_run_llm_verification(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("LLM verification should not run for pixel pipelines")


def _invoke_verify_pipelines(
    monkeypatch: pytest.MonkeyPatch, *args: str
) -> Path | None:
    """Run the CLI with pixel verification mocked and return the saved dir."""
    captured: dict[str, Path | None] = {}

    def fake_run_pixel_generation_verification(
        config: verify_pipelines.PipelineConfig,
        *,
        device_type: verify_pipelines.DeviceKind,
        devices: str,
        find_tolerances: bool,
        print_suggested_tolerances: bool,
        pixel_results_dir: Path | None,
    ) -> verify_pipelines.VerificationVerdict:
        del config
        del device_type
        del devices
        del find_tolerances
        del print_suggested_tolerances
        captured["pixel_results_dir"] = pixel_results_dir
        return verify_pipelines.VerificationVerdict(
            status=verify_pipelines.VerificationStatus.OK
        )

    monkeypatch.setattr(
        verify_pipelines,
        "run_pixel_generation_verification",
        fake_run_pixel_generation_verification,
    )
    monkeypatch.setattr(
        verify_pipelines, "run_llm_verification", _fail_run_llm_verification
    )
    monkeypatch.setattr(
        verify_pipelines, "dump_results", lambda *args, **kwargs: None
    )

    result = RUNNER.invoke(
        verify_pipelines.main,
        [
            "--pipeline",
            PIXEL_PIPELINE,
            "--devices",
            "gpu:0",
            *args,
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    return captured["pixel_results_dir"]


def test_save_pixel_outputs_writes_expected_files(tmp_path: Path) -> None:
    stale_dir = tmp_path / "org__model" / "bfloat16" / "max"
    stale_dir.mkdir(parents=True)
    (stale_dir / "stale.png").write_bytes(b"stale")
    (stale_dir / "stale.txt").write_text("stale")
    (stale_dir / "manifest.json").write_text("{}")

    output_dir = verify_pipelines._save_pixel_outputs(
        results=[
            {
                "prompt": "hello world",
                "images": np.array(
                    [[[0.0, 0.5, 1.0], [1.0, 0.0, 0.0]]], dtype=np.float32
                ),
            },
            {"prompt": "no image"},
        ],
        output_root=tmp_path,
        pipeline="org/model",
        encoding="bfloat16",
        framework_label="max",
    )

    assert output_dir == stale_dir
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "000.png",
        "manifest.json",
    ]
    assert json.loads((output_dir / "manifest.json").read_text()) == {
        "pipeline": "org/model",
        "encoding": "bfloat16",
        "framework": "max",
        "samples": [
            {
                "index": 0,
                "image": "000.png",
                "prompt": "hello world",
            }
        ],
    }


def test_main_defaults_to_not_saving_pixel_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _invoke_verify_pipelines(monkeypatch) is None


def test_main_passes_pixel_results_dir_when_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert (
        _invoke_verify_pipelines(
            monkeypatch, "--pixel-results-dir", str(tmp_path)
        )
        == tmp_path
    )
