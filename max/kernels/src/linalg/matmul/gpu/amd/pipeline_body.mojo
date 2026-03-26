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
"""Builder for declarative pipeline body specifications.

Provides a `PipelineBody` context manager with role-typed methods and
combinators that produce `List[OpDesc]` for use with the pipeline
scheduling framework.

Role-typed methods (load/store/frag/compute) communicate programmer intent;
actual roles are stamped by `annotate_ops()` via the `TargetCostModel`.

Example — single-buffer matmul:

    with PipelineBody() as b:
        b.load(LOAD_DRAM, ch=0)
        b.store(STORE_SMEM, ch=0)
        b.barrier()
        b.fan[num_k_tiles](LOAD_FRAG, ch=0)
        b.fan[num_k_tiles](COMPUTE)
        return b.done()

Example — ping-pong half:

    with PipelineBody() as b:
        b.load(LOAD_A, ch=0, stage=os, sub=1, k=k_special)
        b.load(LOAD_A, ch=0, stage=s,  sub=0, k=k_off)
        b.load(LOAD_B, ch=1, stage=s,  sub=0, k=k_off)
        b.load(LOAD_B, ch=1, stage=s,  sub=1, k=k_off)
        b.fan[2](MMA_LOAD_A, ch=0, stage=s)
        b.fan[2](MMA_LOAD_B, ch=1, stage=s)
        b.grid[2, 2](MMA)
        return b.done()
"""

from std.collections import List

from pipeline.types import KOffsetKind, OpDesc


struct PipelineBody(Movable):
    """Builder for declarative pipeline body specifications.

    Use as a context manager: `with PipelineBody() as b:`. Accumulates
    logical `OpDesc` entries via role-typed methods and combinators.
    Call `b.done()` to finalize the builder and return the op list
    for `annotate_ops()` and subsequent scheduling.
    """

    var _ops: List[OpDesc]

    def __init__(out self):
        self._ops = List[OpDesc]()

    def __enter__(var self) -> Self:
        """Enable `with PipelineBody() as b:` syntax for scoped building."""
        return self^

    # --- Role-typed append methods ---

    def load(
        mut self,
        tag: Int,
        *,
        ch: Int = -1,
        stage: Int = 0,
        sub: Int = 0,
        k: KOffsetKind = KOffsetKind.NONE,
    ):
        """Append a global load (DRAM → LDS or DRAM → registers)."""
        self._ops.append(
            OpDesc.logical(
                tag, channel=ch, stage=stage, subtile=sub, k_offset=k
            )
        )

    def store(
        mut self,
        tag: Int,
        *,
        ch: Int = -1,
        stage: Int = 0,
        sub: Int = 0,
    ):
        """Append a shared memory store (registers → LDS)."""
        self._ops.append(
            OpDesc.logical(tag, channel=ch, stage=stage, subtile=sub)
        )

    def frag(
        mut self,
        tag: Int,
        *,
        ch: Int = -1,
        stage: Int = 0,
        sub: Int = 0,
    ):
        """Append a fragment load (LDS → registers)."""
        self._ops.append(
            OpDesc.logical(tag, channel=ch, stage=stage, subtile=sub)
        )

    def compute(mut self, tag: Int, *, stage: Int = 0, sub: Int = 0):
        """Append a compute/MMA op."""
        self._ops.append(OpDesc.logical(tag, stage=stage, subtile=sub))

    def barrier(mut self):
        """Append a barrier."""
        self._ops.append(OpDesc.barrier())

    def op(
        mut self,
        tag: Int,
        *,
        ch: Int = -1,
        stage: Int = 0,
        sub: Int = 0,
        k: KOffsetKind = KOffsetKind.NONE,
    ):
        """Append a generic logical op (escape hatch for non-standard ops)."""
        self._ops.append(
            OpDesc.logical(
                tag, channel=ch, stage=stage, subtile=sub, k_offset=k
            )
        )

    # --- Combinators ---

    def fan[
        N: Int
    ](
        mut self,
        tag: Int,
        *,
        ch: Int = -1,
        stage: Int = 0,
        k: KOffsetKind = KOffsetKind.NONE,
    ):
        """Append N ops with subtile=0..N-1 (fan-out pattern)."""
        comptime for i in range(N):
            self._ops.append(
                OpDesc.logical(
                    tag, channel=ch, stage=stage, subtile=i, k_offset=k
                )
            )

    def grid[M: Int, N: Int](mut self, tag: Int, *, ch: Int = -1):
        """Append M×N ops with stage=i, subtile=j (2D grid pattern).

        First dimension maps to stage (0..M-1), second to subtile (0..N-1).
        For flat patterns where only subtile varies, use fan[M*N] instead.
        """
        comptime for i in range(M):
            comptime for j in range(N):
                self._ops.append(
                    OpDesc.logical(tag, channel=ch, stage=i, subtile=j)
                )

    # --- Composition ---

    def extend(mut self, mut other: PipelineBody):
        """Absorb all ops from another builder (for hierarchical composition).
        """
        for i in range(len(other._ops)):
            self._ops.append(other._ops[i])

    # --- Finalize ---

    def done(self) -> List[OpDesc]:
        """Return the logical op list."""
        return self._ops.copy()
