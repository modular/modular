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
"""Provide a router that exposes all supported cascade inference routes."""

from fastapi import APIRouter
from max.experimental.cascade.pipelines import CascadePipeline
from max.experimental.cascade.pipelines.imgen import ImageGenInterface
from max.experimental.cascade.pipelines.textgen import TextGenInterface
from max.experimental.cascade.serve import chat_completions, open_responses


def build_router(pipeline: CascadePipeline) -> APIRouter:
    """Auto-configure routes based on the pipeline interfaces."""
    router = APIRouter()

    if isinstance(pipeline, TextGenInterface):
        router.include_router(chat_completions.build_router(pipeline))

    if isinstance(pipeline, ImageGenInterface):
        router.include_router(open_responses.build_router(pipeline))

    return router
