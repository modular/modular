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

import pytest
from max.interfaces import (
    RequestID,
    PixelGenerationRequest,
    TextGenerationRequestMessage,
)


def test_pixel_generation_request_init() -> None:
    # Prompt and messages cannot be provided concurrently.
    with pytest.raises(ValueError):
        _ = PixelGenerationRequest(
            request_id=RequestID(),
            model_name="test",
            prompt="hello world",
            messages=[
                TextGenerationRequestMessage(
                    role="user",
                    content=[{"type": "text", "text": "hello world"}],
                )
            ],
        )

    _ = PixelGenerationRequest(
        request_id=RequestID(),
        model_name="test",
        prompt="hello world",
    )

    _ = PixelGenerationRequest(
        request_id=RequestID(),
        model_name="test",
        prompt=None,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[{"type": "text", "text": "hello world"}],
            )
        ],
    )

    # role not user is not supported.
    with pytest.raises(ValueError):
        _ = PixelGenerationRequest(
            request_id=RequestID(),
            model_name="test",
            prompt=None,
            messages=[
                TextGenerationRequestMessage(
                    role="not_user",
                    content=[{"type": "text", "text": "hello world"}],
                )
            ],
        )
