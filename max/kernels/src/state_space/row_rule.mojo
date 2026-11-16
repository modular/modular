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
"""The mixed-verify row rule: a row longer than one real token plus its
drafts is a prefill row and commits at `row_len - num_draft_tokens`."""


@always_inline
def is_prefill_row(row_len: Int, num_draft_tokens: Int) -> Bool:
    """Whether a row carries real tokens beyond its first.

    Args:
        row_len: The row's token count.
        num_draft_tokens: The step's draft-token count per row.

    Returns:
        True when the row is a prompt chunk with placeholder drafts.
    """
    return row_len > num_draft_tokens + 1


@always_inline
def commit_len(row_len: Int, num_draft_tokens: Int) -> Int:
    """The row's tokens before its drafts: a prefill row's real tokens,
    else `row_len`.

    Args:
        row_len: The row's token count.
        num_draft_tokens: The step's draft-token count per row.

    Returns:
        `row_len - num_draft_tokens` for a prefill row, else `row_len`.
    """
    if is_prefill_row(row_len, num_draft_tokens):
        return row_len - num_draft_tokens
    return row_len


@always_inline
def is_mixed_launch(num_draft_tokens: Int, batch: Int, total_len: Int) -> Bool:
    """Whether a launch carries a prefill row beside its verify rows.

    Args:
        num_draft_tokens: The step's draft-token count per row.
        batch: The launch's row count.
        total_len: The launch's token count.

    Returns:
        True when at least one row is longer than one real token plus
        its drafts.
    """
    return num_draft_tokens > 0 and total_len > batch * (num_draft_tokens + 1)
