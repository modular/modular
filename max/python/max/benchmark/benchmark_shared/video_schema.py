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

"""Video request adapters supplied explicitly by a benchmark entrypoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .backend_names import Backend
    from .config import Endpoint
    from .datasets.types import PixelGenerationImageOptions
    from .request import PixelGenerationRequestFuncInput

VideoRequestSchema = str


class VideoRequestSchemaHandler(Protocol):
    """Adapts a video endpoint's payload and conditioning inputs."""

    def endpoint(self, backend: Backend) -> Endpoint | None:
        """Returns a schema-specific endpoint, or None for the backend default."""
        ...

    def build_payload(
        self, request: PixelGenerationRequestFuncInput
    ) -> Mapping[str, object]:
        """Builds the JSON request body."""
        ...

    def keyframe_paths(self, options: PixelGenerationImageOptions) -> list[str]:
        """Returns the generated conditioning image paths."""
        ...


_VIDEO_REQUEST_SCHEMAS: dict[str, VideoRequestSchemaHandler] = {}


def register_video_request_schema(
    name: str, handler: VideoRequestSchemaHandler
) -> None:
    """Registers a schema supplied by the benchmark entrypoint."""
    _VIDEO_REQUEST_SCHEMAS[name] = handler


def get_video_request_schema(
    name: str | None,
) -> VideoRequestSchemaHandler | None:
    """Returns the registered schema, or None for the default video body."""
    if name in (None, "default"):
        return None
    return _VIDEO_REQUEST_SCHEMAS[name]
