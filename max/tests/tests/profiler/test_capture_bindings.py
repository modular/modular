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

"""Capture bindings are harmless when no profiler plugin is available."""

from pathlib import Path

from max._core.engine import get_global_value
from max._core.profiler import (
    is_capture_recording,
    load_profiler_plugin,
    start_capture,
    stop_capture,
)


def test_capture_calls_without_plugin(tmp_path: Path) -> None:
    assert not load_profiler_plugin()

    assert not start_capture()
    assert not is_capture_recording()

    output_path = tmp_path / "capture.json"
    assert stop_capture(str(output_path))
    assert not output_path.exists()
    # The stop's temporary output path doesn't outlive it.
    assert get_global_value("max-debug.profiling-output-path") is None

    # A second start after a stop is also a no-op.
    assert not start_capture()
    assert not is_capture_recording()
    assert stop_capture(str(output_path))
