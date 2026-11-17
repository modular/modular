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
"""A token prompt carries media that no message part names.

When a chat completion sets ``prompt_tokens``, ``openai_routes`` sends the
token ids with ``messages=[]`` and the request's media, so the media count
cannot be checked against messages. Only a request whose tokenizer matches the
prompt against its media skips that check.
"""

from __future__ import annotations

import pytest
from max.pipelines.modeling.types import (
    ImageContentPart,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)


def test_a_validated_token_prompt_is_not_counted_against_messages() -> None:
    request = TextGenerationRequest(
        request_id=RequestID("tokens"),
        model_name="m",
        prompt=[1, 2, 3],
        images=[b"image-a", b"image-b"],
        videos=[b"video"],
        token_prompt_media_validated=True,
    )
    # Constructed although no message names the media.
    assert request.number_of_images != len(request.images)
    assert request.number_of_videos != len(request.videos)


@pytest.mark.parametrize(
    "media",
    [{"images": [b"image"]}, {"videos": [b"video"]}],
)
def test_a_token_prompt_otherwise_carries_no_media(
    media: dict[str, list[bytes]],
) -> None:
    with pytest.raises(ValueError, match="do not match messages"):
        TextGenerationRequest(
            request_id=RequestID("tokens"),
            model_name="m",
            prompt=[1, 2, 3],
            images=media.get("images", []),
            videos=media.get("videos", []),
        )


def test_the_flag_admits_no_media_beside_a_string_prompt() -> None:
    with pytest.raises(ValueError, match="string prompts cannot be provided"):
        TextGenerationRequest(
            request_id=RequestID("string"),
            model_name="m",
            prompt="describe it",
            images=[b"image"],
            token_prompt_media_validated=True,
        )


def test_messages_must_name_every_image() -> None:
    message = TextGenerationRequestMessage(
        role="user",
        content=[TextContentPart(text="what is this?"), ImageContentPart()],
    )
    with pytest.raises(ValueError, match="number of images"):
        TextGenerationRequest(
            request_id=RequestID("messages"),
            model_name="m",
            messages=[message],
            images=[b"image-a", b"image-b"],
            token_prompt_media_validated=True,
        )
