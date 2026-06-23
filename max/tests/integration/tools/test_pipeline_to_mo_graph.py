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
"""Test the pipeline_to_mo_graph tool, which compiles a model with the same
flags as `max serve` and writes the generated graphs to a directory."""

import subprocess
from pathlib import Path

import python.runfiles

# Use SmolLM-135M for fast testing
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"


def _get_pipeline_to_mo_graph_binary() -> str:
    """Get the path to the pipeline_to_mo_graph binary from runfiles."""
    runfiles = python.runfiles.Create()
    assert runfiles is not None, "Unable to find runfiles tree"
    loc = runfiles.Rlocation(
        "_main/max/tests/integration/tools/pipeline_to_mo_graph"
    )
    assert loc is not None, "Unable to find pipeline_to_mo_graph binary"
    return loc


def test_pipeline_to_mo_graph_target_cuda(tmp_path: Path) -> None:
    """Test pipeline_to_mo_graph with --target cuda:sm_80 on virtual devices.

    This test verifies that:
    - The pipeline_to_mo_graph tool accepts serve-style model flags
    - Compilation succeeds without a physical GPU (virtual devices)
    - The generated MAX graph (`*.mo.mlir`) is written to --output-dir
    - Process exits cleanly with exit code 0
    """
    binary = _get_pipeline_to_mo_graph_binary()
    output_dir = tmp_path / "graphs"
    result = subprocess.run(
        [
            binary,
            "--model",
            MODEL_NAME,
            "--devices",
            "gpu:0",
            "--target",
            "cuda:sm_80",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=180,  # 3 minutes should be plenty for compilation
    )

    # Verify successful exit
    assert result.returncode == 0, (
        f"Command failed with stderr:\n{result.stderr}"
    )

    # Verify virtual device mode was enabled
    assert (
        "Compiling for target cuda:sm_80 using virtual devices" in result.stderr
    ), "Target compilation message not found"

    # Verify graph files landed in the output directory, including the
    # initial MAX graph dump.
    dumped = sorted(p.name for p in output_dir.iterdir())
    assert dumped, f"No graph files written. stderr:\n{result.stderr}"
    assert any(name.endswith(".mo.mlir") for name in dumped), (
        f"Expected a .mo.mlir MAX graph dump in {output_dir}, got {dumped}"
    )
    assert f"graph files to {output_dir}" in result.stdout, (
        f"Expected summary line in stdout, got:\n{result.stdout}"
    )


def test_pipeline_to_mo_graph_build_only(tmp_path: Path) -> None:
    """Test pipeline_to_mo_graph --build-only on virtual devices.

    This test verifies that:
    - Graphs are dumped as soon as they are built, without running the
      graph compiler (no physical GPU needed; compilation is skipped)
    - Only as-built MAX graphs (`*.mo.mlir`) land in --output-dir; no
      later-stage compiler dumps are produced
    - The dumps are MLIR text containing `mo.graph` ops
    - Ops carry Python source locations back to the building code
    - Process exits cleanly with exit code 0
    """
    binary = _get_pipeline_to_mo_graph_binary()
    output_dir = tmp_path / "graphs"
    result = subprocess.run(
        [
            binary,
            "--model",
            MODEL_NAME,
            "--devices",
            "gpu:0",
            "--target",
            "cuda:sm_80",
            "--build-only",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        f"Command failed with stderr:\n{result.stderr}"
    )

    dumped = sorted(p.name for p in output_dir.iterdir())
    assert dumped, f"No graph files written. stderr:\n{result.stderr}"
    assert all(name.endswith(".mo.mlir") for name in dumped), (
        f"Expected only .mo.mlir dumps in build-only mode, got {dumped}"
    )
    # The main model graph should be among the dumps, as real MLIR text.
    assert any(
        "mo.graph" in (output_dir / name).read_text() for name in dumped
    ), f"No dump in {dumped} contains a mo.graph op"
    # Ops should be annotated with the Python source they were built from,
    # which the materialized locations spell out as `.py` file references.
    assert any(".py" in (output_dir / name).read_text() for name in dumped), (
        f"No dump in {dumped} carries a Python source location"
    )
    assert f"graph files to {output_dir}" in result.stdout, (
        f"Expected summary line in stdout, got:\n{result.stdout}"
    )
