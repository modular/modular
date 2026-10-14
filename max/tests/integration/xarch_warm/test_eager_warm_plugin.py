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

"""The eager warm plugin's ``compile_on_gpu`` marker, under an interpreter-only
default."""

import pytest
from max.experimental import executor


def test_unmarked_test_is_interpreter_only() -> None:
    assert isinstance(executor.default_executor(), executor.InterpreterExecutor)


@pytest.mark.compile_on_gpu("pinned by the plugin's own test")
def test_marked_test_compiles_only_what_the_interpreter_refuses() -> None:
    default = executor.default_executor()
    assert isinstance(default, executor._FallbackExecutor)
    assert default.refusals_only


def test_marker_restores_the_default_after_its_test() -> None:
    assert isinstance(executor.default_executor(), executor.InterpreterExecutor)
