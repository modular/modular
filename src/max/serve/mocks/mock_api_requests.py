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

"""Provides some canonical client data"""

import time


def simple_openai_request(
    model_name="gpt-3.5-turbo", content="Say this is a test!", stream=False
):
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.7,
        "stream": stream,
    }


def simple_openai_response(
    contents: str,
    model_name: str = "unnamed-model",
    response_id="response_id_0",
    timestamp=time.time_ns(),
):
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": timestamp,
        "model": model_name,
        # TODO - populate the usage statistics
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
        },
        "choices": [
            {
                # TODO - contents is a string, but may contain rich data such as roles.
                # How do we deal with this.
                # `"role":assistant"`` is required?
                "message": {"role": "assistant", "content": contents},
                "logprobs": {"content": []},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


def simple_openai_stream_request():
    """
    A simple streaming request.
    Verify via:
    curl https://api.openai.com/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" -d '{ "model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Say this is a test!"}], "stream": true}'
    """
    return {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Say This is a test!"}],
        "stream": "true",
    }


def simple_openai_stream_response(
    contents: list[str],
    model_name: str = "unnamed-model",
    response_id="response_id_0",
    timestamp=time.time_ns(),
):
    json_response = []
    # TODO this needs to be individually callable by the token gen
    # go over the message list and wrap each to the response.
    for i in range(len(contents) - 1):
        json_response.append(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model_name,
                "system_fingerprint": "null",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": contents[i]},
                    }
                ],
            }
        )
    # append the final message to indicate we are done with the response
    json_response.append(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": model_name,
            "system_fingerprint": "null",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    return json_response


def simple_kserve_request():
    return {
        "inputs": [
            {
                "name": "args_0",
                "shape": [1, 1],
                "datatype": "FP32",
                "data": [[1]],
            },
            {
                "name": "args_1",
                "shape": [1, 1],
                "datatype": "FP32",
                "data": [[1]],
            },
        ],
        "outputs": ["output_0"],
    }


def simple_kserve_response():
    return {
        "id": "infer-add",
        "model_name": "Add",
        "model_version": "v1.0.0",
        "outputs": [
            {
                "data": [[1.0]],
                "datatype": "FP32",
                "name": "output_0",
                "shape": [1, 1],
            }
        ],
    }
