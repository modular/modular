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
"""Defines the synchronization building blocks use on atom completions.

The layout package splits kernel code into three layers:

- Atoms (`layout.atoms`): one instruction each, plus a description of its
  operands and of how it completes.
- TileOps (`layout.tileops`): a static tiling of atoms over a tile.
- Building blocks: the code one group of threads runs, composed of TileOps.

Building blocks are the only layer that synchronizes. Atoms and TileOps
contain no synchronization of any kind: no wait, commit, fence, barrier
arrival or barrier wait. A TileOp returns once its atoms are issued, and the
building block that called it decides when to wait, so that other work can
run while the issues are in flight.

This package is the shared part of that layer: the waits and commits that
consume a completion an atom describes. Each function is keyed by the
completion's `family`, so a building block written over a generic atom `A`
waits with `wait_issues[A.Completion]()` and gets the right instruction on
every target. A family this package does not know fails to compile; its
target supplies the wait in its own building blocks.

You can import these APIs from the `layout.building_blocks` package:

```mojo
from layout.building_blocks import (
    commit_batch,
    open_batch,
    wait_batches,
    wait_issues,
)
```
"""

from layout.atoms import BatchCompletion, CompletionFamily, CounterCompletion
from layout.tmem_engine import TMemEngine

from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_all,
    async_copy_wait_group,
)


@always_inline
def wait_issues[C: CounterCompletion, pending: Int = 0]():
    """Blocks until at most `pending` of this thread's issues of `C`'s
    family are outstanding.

    Parameters:
        C: The completion of the atoms to wait for.
        pending: The issues that may stay outstanding, at most
            `C.max_pending`.
    """
    comptime assert pending >= 0, "wait_issues: pending is not negative"
    comptime assert (
        pending <= C.max_pending
    ), "wait_issues: the family cannot leave that many issues outstanding"
    comptime match C.family:
        case .tensor_memory_load:
            TMemEngine.wait_load()
        case .tensor_memory_store:
            TMemEngine.wait_store()
        case _:
            comptime assert False, (
                "wait_issues: no wait for this completion family; its target's"
                " building blocks supply it"
            )


@always_inline
def open_batch[C: BatchCompletion]():
    """Opens a batch for the issues of `C`'s family that follow.

    Parameters:
        C: The completion of the atoms the batch collects.
    """
    comptime match C.family:
        case .async_copy_group:
            # A `cp.async` group is always open, and issues join it.
            pass
        case _:
            comptime assert False, (
                "open_batch: no batch for this completion family; its target's"
                " building blocks supply it"
            )


@always_inline
def commit_batch[C: BatchCompletion]():
    """Closes the open batch of `C`'s family.

    Parameters:
        C: The completion of the atoms the batch collects.
    """
    comptime match C.family:
        case .async_copy_group:
            async_copy_commit_group()
        case _:
            comptime assert False, (
                "commit_batch: no batch for this completion family; its"
                " target's building blocks supply it"
            )


@always_inline
def wait_batches[C: BatchCompletion, pending: Int = 0]():
    """Blocks until at most `pending` committed batches of `C`'s family are
    outstanding.

    Parameters:
        C: The completion of the atoms the batches collect.
        pending: The batches that may stay outstanding, at most
            `C.max_pending`.
    """
    comptime assert pending >= 0, "wait_batches: pending is not negative"
    comptime assert (
        pending <= C.max_pending
    ), "wait_batches: the family cannot leave that many batches outstanding"
    comptime match C.family:
        case .async_copy_group:
            comptime if pending == 0:
                async_copy_wait_all()
            else:
                async_copy_wait_group(Int32(pending))
        case _:
            comptime assert False, (
                "wait_batches: no batch for this completion family; its"
                " target's building blocks supply it"
            )
