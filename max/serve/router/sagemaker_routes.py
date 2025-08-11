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

import logging
from typing import Union

from fastapi import APIRouter, Request
from fastapi.responses import Response
from max.serve.router.openai_routes import openai_create_chat_completion
from max.serve.schemas.openai import (
    CreateChatCompletionResponse,
)
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger("max.serve")

router = APIRouter()


@router.get("/ping")
async def ping() -> Response:
    """Health check."""
    return Response(status_code=200)


@router.post("/invocations", response_model=None)
async def invocations(
    request: Request,
) -> Union[CreateChatCompletionResponse, EventSourceResponse]:
    """proxy to /v1/chat/completions"""
    return await openai_create_chat_completion(request)
