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
"""Tests for max.experimental.support."""

from contextvars import ContextVar

from max.experimental.support import contextvar_context


def test_contextvar_context() -> None:
    var: ContextVar[str] = ContextVar("test_var")

    with contextvar_context(var, "outer") as value:
        assert value == var.get() == "outer"

        with contextvar_context(var, "inner") as value:
            assert value == var.get() == "inner"

        # reset to "outer"
        assert var.get() == "outer"

    # unset
    assert var.get(None) is None
