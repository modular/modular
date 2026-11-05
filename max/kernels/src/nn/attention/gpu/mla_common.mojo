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

"""MLA types shared by every GPU vendor's decode path."""

from std.collections import OptionalReg


@fieldwise_init
struct SparseRowLayout(TrivialRegisterPassable):
    """Describes how a sparse index producer packs one token's row.

    Every producer packs its valid entries first and pads with `-1`, so the
    number of valid entries is a closed-form function of how many keys the
    token sees. The decode mask uses it to keep `-1` slots out of the
    softmax without reading the indices.

    A k-pool producer (`kpool_expand_topk_kernel`) selects up to
    `pool_topk` pools of `kpool` keys, then, with `select_tail`, the
    `visible % kpool` keys no complete pool covers yet. `kpool = 1` is a
    plain top-k row; `pool_topk = 0` leaves its length unbounded, since the
    kernel's slot count already caps it.
    """

    var kpool: Int
    """Keys per selected pool; 1 for plain top-k."""
    var pool_topk: Int
    """Maximum number of selected pools; 0 for no bound."""
    var select_tail: Bool
    """Whether the row appends the incomplete trailing pool."""

    def __init__(out self):
        """Creates the plain top-k layout, which adds no bound."""
        self.kpool = 1
        self.pool_topk = 0
        self.select_tail = True

    def is_bounded(self) -> Bool:
        """Returns whether `valid_len` can be shorter than its argument.

        Returns:
            True when the layout bounds a row below the token's key count.
        """
        return self.kpool > 1 or self.pool_topk > 0

    @inline(.always)
    def valid_len(self, visible: Int) -> Int:
        """Returns the number of valid entries in a row.

        Args:
            visible: Number of keys visible to the token the row was
                computed for.

        Returns:
            The row's valid length; slots at or past it hold `-1`.
        """
        if self.kpool == 1:
            if self.pool_topk == 0:
                return visible
            return min(visible, self.pool_topk)
        var pools, tail = divmod(visible, self.kpool)
        var n = self.kpool * min(pools, self.pool_topk)
        if self.select_tail:
            n += tail
        return n

    @inline(.always)
    def row_limit(self, visible: Int, selection_lag: Int) -> Int:
        """Returns the valid length of a row read `selection_lag` tokens late.

        Args:
            visible: Number of keys visible to the reading token.
            selection_lag: Positions between the token the row was computed
                for and the reading token.

        Returns:
            The row's valid length, or `Int.MAX` when nothing bounds it.
        """
        if not (self.is_bounded() or selection_lag != 0):
            return Int.MAX
        return self.valid_len(max(visible - selection_lag, 0))


@inline(.always)
def _sparse_row_packed[
    sparse_row: SparseRowLayout
](
    d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]],
    indices_stride: Int,
    q_token_idx: Int,
    visible: Int,
    selection_lag: Int,
    sparse_topk: Int,
) -> Bool:
    var n = sparse_row.row_limit(visible, selection_lag)
    if not d_indices or n == Int.MAX:
        return True
    var row = d_indices.value().unsafe_offset(q_token_idx * indices_stride)
    if min(n, sparse_topk) > 0 and row[min(n, sparse_topk) - 1] < 0:
        return False
    # At or past the slot count, the scan stops before the padding.
    return n >= sparse_topk or row[n] == -1


@inline(.always)
def assert_sparse_row_packed[
    sparse_row: SparseRowLayout
](
    d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]],
    indices_stride: Int,
    q_token_idx: Int,
    visible: Int,
    selection_lag: Int,
    sparse_topk: Int,
):
    """Asserts that a token's sparse row is packed as `sparse_row` states.

    Only runs with all assertions enabled. A producer that drops a causal key
    makes the closed-form length overcount, and the softmax then scores the
    trailing `-1` slots as zero keys instead of masking them.

    The message is a literal on purpose: formatting runtime values needs a
    2 KB local buffer, and CUDA 13.1 ptxas segfaults compiling that callee
    from a warp-specialized branch after `setmaxnreg`.

    Parameters:
        sparse_row: The producer's row layout.

    Args:
        d_indices: The `[total_q_tokens, indices_stride]` sparse indices.
        indices_stride: Row stride of `d_indices`.
        q_token_idx: Global index of the token whose row is checked.
        visible: Number of keys the token sees.
        selection_lag: Positions between the token the row was computed for
            and this one.
        sparse_topk: The kernel's slot count for the sequence.
    """
    debug_assert(
        _sparse_row_packed[sparse_row](
            d_indices,
            indices_stride,
            q_token_idx,
            visible,
            selection_lag,
            sparse_topk,
        ),
        "sparse row is not packed to its closed-form valid length",
    )
