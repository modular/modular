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

"""Provides a utility function for calculating the maximum number of tokens to generate."""


def max_tokens_to_generate(
    prompt_size: int,
    max_length: int,
    max_new_tokens: int = -1,
) -> int:
    """Returns the maximum number of new tokens to generate.

    Respects both ``max_length`` (minus ``prompt_size``) and, when non-negative,
    ``max_new_tokens``; returns the minimum of the two when both apply.

    Args:
        prompt_size: Current prompt (context) length in tokens.
        max_length: Maximum total sequence length.
        max_new_tokens: Cap on new tokens, or ``-1`` to use only ``max_length``.

    Returns:
        The effective cap on new tokens to generate.
    """
    _difference_between_max_and_prompt = max(max_length - prompt_size, 0)
    if max_new_tokens < 0:
        return _difference_between_max_and_prompt
    return min(max_new_tokens, _difference_between_max_and_prompt)
