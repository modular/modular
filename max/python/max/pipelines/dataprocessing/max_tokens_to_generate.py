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


def max_tokens_to_generate(
    prompt_size: int,
    max_length: int,
    max_new_tokens: int = -1,
) -> int:
    """Returns the max number of new tokens to generate."""
    _difference_between_max_and_prompt = max(max_length - prompt_size, 0)
    if max_new_tokens < 0:
        return _difference_between_max_and_prompt
    return min(max_new_tokens, _difference_between_max_and_prompt)
