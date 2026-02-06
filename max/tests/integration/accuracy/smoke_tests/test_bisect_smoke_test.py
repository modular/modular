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
"""Tests for bisect_smoke_test.py parsing compatibility with smoke_test.py output."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from smoke_tests.bisect_smoke_test import parse_results
from smoke_tests.smoke_test import (
    TEXT_TASK,
    VISION_TASK,
    build_eval_summary,
    write_results,
)


def test_eval_metrics_roundtrip() -> None:
    """Verify smoke_test.py output can be parsed by bisect_smoke_test.py."""
    model = "test/model"

    # Mock LmEvalResults structure (what lm_eval produces)
    lm_eval_results = [
        {
            "results": {
                TEXT_TASK: {
                    "exact_match,flexible-extract": 0.85,
                    "exact_match_stderr,flexible-extract": 0.01,
                }
            },
            "total_evaluation_time_seconds": "100.0",
            "task_hashes": {TEXT_TASK: "abc123"},
        },
        {
            "results": {
                VISION_TASK: {
                    "relaxed_accuracy,none": 0.72,
                    "relaxed_accuracy_stderr,none": 0.02,
                }
            },
            "total_evaluation_time_seconds": "80.0",
            "task_hashes": {VISION_TASK: "def456"},
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        model_dir = output_path / model.lower().replace("/", "__")
        model_dir.mkdir(parents=True)

        # Mock GPU detection to avoid calling nvidia-smi
        with patch(
            "smoke_tests.smoke_test.get_gpu_name_and_count",
            return_value=("Test GPU", 1),
        ):
            summary = build_eval_summary(
                lm_eval_results, startup_time_seconds=10.0
            )

        # Write using smoke_test.py's function
        write_results(model_dir, summary, results=[], all_samples=[], tasks=[])

        # Parse using bisect_smoke_test.py's function
        results = parse_results(output_path, model)

        assert results is not None
        assert results["text"] == 0.85
        assert results["vision"] == 0.72
