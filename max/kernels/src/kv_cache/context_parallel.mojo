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
"""Token ownership for a KV cache sharded across a context-parallel group.

Under decode context parallelism (DCP) every rank of a group receives the same
lookup table and the same *global* `cache_lengths`, but stores only the
positions it owns. Position `p` belongs to rank `(p // interleave) % size`
and lands in that rank's local slot
`(p // (interleave * size)) * interleave + p % interleave`; local slots are
addressed through the shared lookup table exactly like positions are without
CP. A rank's local slots are ordered by global position, so "keys visible to a
query at position `q`" is the prefix `[0, local_len(q + 1))`.

With `size == 1` every function reduces to the identity, so kernels can call
them unconditionally.
"""

from std.math import ceildiv


@fieldwise_init
struct ContextParallelLayout[size: Int = 1, interleave: Int = 1](
    TrivialRegisterPassable
):
    """The position-to-slot map of one rank in a context-parallel group.

    Parameters:
        size: Number of ranks sharing each request's KV positions.
        interleave: Consecutive positions a rank owns before the next rank's
            turn: `1` stripes single tokens, a page size stripes whole pages.
    """

    var rank: Int
    """This rank's index in `[0, size)`."""

    comptime enabled = Self.size > 1
    comptime _span = Self.size * Self.interleave

    @inline(.always)
    def owner(self, pos: Int) -> Int:
        """Returns the rank that stores global position `pos`."""
        comptime if not Self.enabled:
            return 0
        return (pos // Self.interleave) % Self.size

    @inline(.always)
    def owns(self, pos: Int) -> Bool:
        """Returns whether this rank stores global position `pos`."""
        comptime if not Self.enabled:
            return True
        return self.owner(pos) == self.rank

    @inline(.always)
    def local_slot(self, pos: Int) -> Int:
        """Returns the owner's local slot for global position `pos`."""
        comptime if not Self.enabled:
            return pos
        return (pos // Self._span) * Self.interleave + pos % Self.interleave

    @inline(.always)
    def global_pos(self, slot: Int) -> Int:
        """Returns the global position stored in this rank's local `slot`."""
        comptime if not Self.enabled:
            return slot
        return (
            (slot // Self.interleave) * Self._span
            + self.rank * Self.interleave
            + slot % Self.interleave
        )

    @inline(.always)
    def local_len(self, length: Int) -> Int:
        """Returns how many of the positions `[0, length)` this rank stores."""
        comptime if not Self.enabled:
            return length
        var full = (length // Self._span) * Self.interleave
        var rem = length % Self._span - self.rank * Self.interleave
        return full + max(0, min(rem, Self.interleave))

    @staticmethod
    @inline(.always)
    def max_local_len(length: Int) -> Int:
        """Returns an upper bound of `local_len(length)` over all ranks."""
        comptime if not Self.enabled:
            return length
        return ceildiv(length, Self._span) * Self.interleave
