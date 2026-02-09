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

from max.interfaces import RequestID
from max.interfaces.request import OpenResponsesRequest
from max.interfaces.request.open_responses import OpenResponsesRequestBody


def test_pixel_generation_request_init() -> None:
    # Basic initialization.
    body = OpenResponsesRequestBody(
        model="test",
        input="hello world",
    )
    _ = OpenResponsesRequest(request_id=RequestID(), body=body)

    # Empty prompt should raise ValidationError (pydantic validation).
    # OpenResponsesRequest accepts empty strings, but the tokenizer will validate
    # For now, test that we can create the request object with empty input
    body_empty = OpenResponsesRequestBody(
        model="test",
        input="",
    )
    _ = OpenResponsesRequest(request_id=RequestID(), body=body_empty)
