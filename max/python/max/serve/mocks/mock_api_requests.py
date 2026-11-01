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

"""Provides some canonical client data."""

from typing import Any


def simple_openai_request(
    model_name: str = "gpt-3.5-turbo",
    content: str = "Say this is a test!",
    stream: bool = False,
) -> dict[str, Any]:
    """Builds a minimal OpenAI chat-completion request body."""
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.7,
        "stream": stream,
    }


def simple_kserve_request() -> dict[str, Any]:
    """Builds a minimal KServe v2 inference request body."""
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


def simple_kserve_response() -> dict[str, Any]:
    """Builds a KServe v2 inference response body."""
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
