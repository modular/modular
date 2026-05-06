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
"""Benchmark serving dev unit tests"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from max.benchmark.benchmark_serving import parse_args


def test_benchmark_serving_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the benchmark serving help function."""
    # Mock sys.argv to simulate running with --help flag
    test_args = ["benchmark_serving.py", "--help"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            parse_args()

        # Verify it exited with code 0 (success)
        assert excinfo.value.code == 0

        # Capture and verify the help output
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()
