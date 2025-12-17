# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import unittest.mock

import pytest
from max.profiler import Tracer, traced


def test_profiling() -> None:
    """Tests that profiling functions do not error."""
    with Tracer("foo"):
        pass

    @traced(message="baz", color="red")
    def foo() -> None:
        # The span is named "baz".
        pass

    @traced
    def bar() -> None:
        # The span is named "bar".
        pass

    foo()
    bar()

    Tracer("I'm here").mark()


def test_profiling_disabled() -> None:
    with unittest.mock.patch(
        "max.profiler.tracing.is_profiling_enabled", return_value=False
    ) as m:
        test_profiling()


@pytest.mark.asyncio
async def test_async_profiling() -> None:
    """Tests that profiling async functions doesn't error."""

    async def bar() -> None:
        pass

    @traced(message="baz", color="red")
    async def foo() -> None:
        await bar()

    with Tracer("potato"):
        await foo()
