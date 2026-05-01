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
"""Functional tests for the dummy text-generation pipeline."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from max.experimental.cascade import GenerateRequest, LocalRuntime
from max.tests.tests.cascade.dummy_textgen import (
    build_dummy_textgen_pipeline,
)


@pytest.fixture()
async def runtime() -> AsyncIterator[LocalRuntime]:
    async with LocalRuntime().open() as rt:
        yield rt


@pytest.mark.asyncio
async def test_textgen_pipeline(runtime: LocalRuntime) -> None:
    pipeline = await build_dummy_textgen_pipeline()
    await pipeline.deploy(runtime)

    req = GenerateRequest(num_tokens=5)
    tokens = [token async for token in pipeline.generate(req, "hello, ")]

    assert len(tokens) == 5
    assert all(token == "A" for token in tokens)


@pytest.mark.asyncio
async def test_textgen_different_lengths(runtime: LocalRuntime) -> None:
    pipeline = await build_dummy_textgen_pipeline()
    await pipeline.deploy(runtime)

    for num_tokens in [1, 3, 10]:
        request = GenerateRequest(num_tokens=num_tokens)
        tokens = [token async for token in pipeline.generate(request, "test")]
        assert len(tokens) == num_tokens
