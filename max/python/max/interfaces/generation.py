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
"""General output interfaces for pipeline operations in the MAX API.

This module provides output structures for pipeline operations that generate
images and other content types using the OpenResponses API content format.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .request import RequestID
from .request.open_responses import OutputContent
from .status import GenerationStatus


class GenerationOutput(BaseModel):
    """Output container for image generation pipeline operations.

    This class holds a list of generated images in OpenResponses API format,
    along with request tracking and status information. It implements the
    PipelineOutput protocol by providing the required `is_done` property.

    Example:

    .. code-block:: python

        import numpy as np
        from max.interfaces.generation import GenerationOutput
        from max.interfaces.request import RequestID
        from max.interfaces.request.open_responses import OutputImageContent
        from max.interfaces.status import GenerationStatus

        # Convert numpy arrays to OutputImageContent using the factory method
        img_array1 = np.random.rand(512, 512, 3).astype(np.float32)
        img_array2 = np.random.rand(512, 512, 3).astype(np.float32)

        result = GenerationOutput(
            request_id=RequestID(value="req-123"),
            final_status=GenerationStatus.END_OF_SEQUENCE,
            output=[
                OutputImageContent.from_numpy(img_array1, format="png"),
                OutputImageContent.from_numpy(img_array2, format="jpeg"),
            ]
        )

        # Or create from URLs
        result_from_urls = GenerationOutput(
            request_id=RequestID(value="req-456"),
            final_status=GenerationStatus.END_OF_SEQUENCE,
            output=[
                OutputImageContent(
                    type="output_image",
                    image_url="https://example.com/image1.png",
                    format="png"
                )
            ]
        )

        # Check if generation is complete
        if result.is_done:
            print(f"Generated {len(result.output)} images")
    """

    model_config = ConfigDict(frozen=True)

    request_id: RequestID
    """The unique identifier for the generation request."""

    final_status: GenerationStatus
    """The final status of the generation process."""

    output: list[OutputContent]
    """List of OutputContent objects (text, images, etc.) representing generated content."""

    @property
    def is_done(self) -> bool:
        """Indicates whether the pipeline operation has completed.

        Returns:
            bool: True if the generation is done (status is not ACTIVE),
                  False otherwise.
        """
        return self.final_status.is_done
