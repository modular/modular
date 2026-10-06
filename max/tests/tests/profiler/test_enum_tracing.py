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

"""Binding enums survive the method tracing profiling installs at import.

Runs with MODULAR_ENABLE_PROFILING=1 (see BUILD.bazel), which is what makes
the _core modules install that tracing; without it these pass vacuously.
"""

from max.driver import LaunchTraceEntry, Usage


def test_flag_members_combine() -> None:
    combined = Usage.STAGING | Usage.UNTRACKED
    assert Usage.STAGING in combined
    assert Usage.UNTRACKED in combined


def test_enum_nested_in_class_is_reachable() -> None:
    kind = LaunchTraceEntry.OperationKind.MEMCPY
    assert kind in LaunchTraceEntry.OperationKind
