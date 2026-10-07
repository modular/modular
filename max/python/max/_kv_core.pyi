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
"""Host-side KV cache bookkeeping shared by MAX and Mach.

A compiled extension; this stub is its public surface.

Which prefix of a hash chain a cache tree can reuse depends on how far back
each leaf's attention reads. The rules take residency alone, so they answer
for any tier: the device pools, or a KV connector reporting what an external
store holds. A connector therefore needs no notion of full / sliding-window /
SSM: it reports presence, and these rules decide what that presence is worth.

Two shapes, and they behave differently under narrowing:

* a full-attention leaf reads its whole history, so its hit is the run from
  the root and shortening an answer can never invalidate it;
* a sliding-window leaf reads only the last ``blocks_in_window`` blocks
  before wherever the sequence stops, so its hit is a SUFFIX run whose
  validity moves with the stopping point. Shortening can invalidate it.

That second property is why :func:`longest_joint_prefix_hit` iterates instead
of taking a minimum: narrowing for one leaf can invalidate another's window,
which narrows it again.
"""

from collections.abc import Sequence
from typing import TypeAlias, final

import numpy as np
import numpy.typing as npt

# Whether each block of a hash chain is held, positionally. Stub-only: the
# extension has no such attribute.
_Residency: TypeAlias = Sequence[bool] | npt.NDArray[np.bool_]

@final
class LeafShape:
    """A leaf's attention shape, which selects the rule that reads its residency.

    Built only through the static constructors.
    """

    @staticmethod
    def full() -> LeafShape:
        """Returns the shape of a leaf that reads its whole history."""

    @staticmethod
    def sliding_window(blocks_in_window: int) -> LeafShape:
        """Returns the shape of a leaf that reads its last ``blocks_in_window`` blocks.

        Raises:
            ValueError: If ``blocks_in_window`` is zero.
        """

    @staticmethod
    def recurrent() -> LeafShape:
        """Returns the shape of a leaf that holds one state per boundary.

        Any resident boundary is a complete hit, so the deepest one wins.
        """

    @staticmethod
    def scratch() -> LeafShape:
        """Returns the shape of a leaf that is never published.

        Such a leaf never narrows a hit.
        """

    def longest_hit(self, resident: _Residency) -> int:
        """Returns how much of ``resident`` a leaf of this shape can reuse.

        The answer is counted from the start even for a shape that only reads
        the tail, so shapes can be compared with one another.
        ``len(resident)`` bounds it, so narrowing means passing a shorter view.
        """

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

def longest_joint_prefix_hit(
    leaves: Sequence[tuple[LeafShape, _Residency]],
) -> int:
    """Returns the longest prefix every leaf can serve at once.

    Each leaf answers under the run the others have already allowed, so the
    run is settled once every leaf has accepted it in turn. A leaf asked
    again about a prefix it just returned has to return that same length, or
    the loop would walk the run to nothing.

    Iterating is required rather than tidy. A minimum over the leaves'
    answers is wrong twice over: a windowed leaf's answer is not a depth that
    can be compared with a full leaf's, and narrowing for one windowed leaf
    can invalidate another's window, which narrows it again.

    Args:
        leaves: One ``(shape, resident)`` pair per leaf. A sequence rather
            than a mapping, because leaves routinely share a shape (an FP8
            tree's values and scales are both full attention) and each
            still answers from its own mask.

    Returns:
        The agreed prefix length; ``0`` when there are no leaves or they
        cannot agree on any.

    Raises:
        ValueError: If the masks are not all the same width. They index one
            chain of hashes, so a short one would quietly answer about a
            different prefix than the rest.
    """

def blocks_held_of_hit(
    num_hit_blocks: int, blocks_in_window: int | None
) -> int:
    """Returns how many blocks a leaf holds of a hit ``num_hit_blocks`` deep.

    The companion to the prefix-hit rules, and not the same number: a
    windowed leaf's hit can cover a whole prefix while the leaf holds only
    the window at the end of it, because the slots below the window are never
    read. A full leaf (``blocks_in_window=None``) holds all of it.

    Every leaf's blocks END at ``num_hit_blocks``, so a leaf holding ``n`` of
    them covers ``hashes[num_hit_blocks - n : num_hit_blocks]`` and the
    ``num_hit_blocks - n`` slots below are the null block.
    """
