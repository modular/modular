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
"""Cleanup of a stream that opens with an error must run before
``_start_stream`` returns, not at finalization."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from fastapi.responses import JSONResponse
from max.pipelines.context.exceptions import InputError
from max.serve.router.openai_routes import _start_stream


@pytest.mark.asyncio
async def test_error_first_chunk_runs_generator_cleanup() -> None:
    cleaned_up = False

    async def rejects() -> AsyncGenerator[JSONResponse, None]:
        nonlocal cleaned_up
        try:
            # The handlers suspend inside an except arm, so GeneratorExit
            # arrives while that exception is still being handled.
            try:
                raise InputError("nope")
            except InputError as e:
                yield JSONResponse(status_code=400, content={"detail": str(e)})
        finally:
            cleaned_up = True

    error, stream = await _start_stream(rejects())

    assert isinstance(error, JSONResponse)
    assert error.status_code == 400
    assert cleaned_up, "the finally must run before _start_stream returns"
    # Replaying it would let a caller serialize it into the body.
    assert [chunk async for chunk in stream] == []


@pytest.mark.asyncio
async def test_successful_first_chunk_is_replayed() -> None:
    cleaned_up = False

    async def succeeds() -> AsyncGenerator[str, None]:
        nonlocal cleaned_up
        try:
            yield "first"
            yield "second"
        finally:
            cleaned_up = True

    error, stream = await _start_stream(succeeds())

    assert error is None
    assert not cleaned_up, "the non-error branch must not close the iterator"
    assert [chunk async for chunk in stream] == ["first", "second"]
    assert cleaned_up


@pytest.mark.asyncio
async def test_empty_stream_yields_nothing() -> None:

    async def empty() -> AsyncGenerator[str, None]:
        return
        yield  # unreachable; makes this an async generator

    error, stream = await _start_stream(empty())

    assert error is None
    assert [chunk async for chunk in stream] == []
