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
"""The scratch workspace a fused op's host code stages its own launches through.

A fused graph op sometimes has to hand one kernel's output to a second kernel it
runs itself, with no graph tensor in between: the grouped FP8 epilogue of
`mo.composite.distributed.allreduce_add_rms_norm_quant_fp8` writes the normed
rows and then quantizes them, and those rows are not a value the graph asked
for. The signal buffer's per-rank scratch is not available for that: the
collectives stage through it themselves, so a region a kernel writes as its
*output* would alias one they read.

The workspace is one buffer per device, created on first use, grown
geometrically past the largest request and then kept for the process's life.
It is published through the compiler runtime's global cache the way
`comm.multicast_window` publishes its window, so a pointer a captured graph
baked in stays valid: a device graph capture records the launches, and nothing
here moves or frees what they were handed.

A caller owns the workspace for one of its own launch sequences, on one device's
stream at a time. That is what makes one region enough: the two launches that
share it are enqueued back to back on the device's stream by the same
invocation, so on the device the writer has retired before the reader starts,
and no other caller of this module can be between them, because a second
invocation enqueues its own pair after this one's on that same stream. A caller
that ran two sequences concurrently on two streams of the same device, or that
lets its reader outlive the pair, would alias another's rows and needs a region
of its own: nothing here detects that.
"""

from std.ffi import _get_global_or_null, external_call
from std.memory.alloc import Layout as AllocLayout
from std.sys import size_of

from max.gpu.host import DeviceBuffer, DeviceContext


@fieldwise_init
struct StagingSlot(ImplicitlyCopyable):
    """What the global cache holds for one device: its workspace and its size.
    """

    var address: Int
    """Address of the workspace's first byte, or 0 before the first request."""

    var bytes: Int
    """Bytes the workspace was allocated with."""


def staging_workspace[
    dtype: DType
](ctx: DeviceContext, elements: Int) raises -> MutPointer[
    Scalar[dtype], MutUntrackedOrigin
]:
    """Returns this device's staging workspace, holding at least `elements`.

    The buffer outlives every caller (see the module docstring), so the pointer
    is safe to hand to a kernel and to keep in a captured graph. A request
    larger than the workspace grows it to at least twice its old size, and the
    old buffer is left mapped rather than freed: a launch still holding its
    address cannot be told apart from one that is not. Doubling bounds what a
    rising sequence of requests parks to under the final size again.

    Parameters:
        dtype: Element type the caller reads and writes the region as.

    Args:
        ctx: Device context the caller's launches run on.
        elements: Elements of `dtype` the caller needs.

    Returns:
        The workspace's first element, on `ctx`'s device, at the allocator's
        own alignment.

    Raises:
        If the device cannot allocate the workspace.
    """
    var needed = elements * size_of[dtype]()
    var name = String(t"COMM_STAGING_WORKSPACE_", ctx.device().id)

    var slot: Optional[Pointer[StagingSlot, MutUntrackedOrigin]]
    var grown = needed
    if case .Some(cached) = _get_global_or_null(name):
        var published = cached.unsafe_bitcast[StagingSlot]()
        if Int(published[].bytes) >= needed:
            return MutPointer[Scalar[dtype], MutUntrackedOrigin](
                unsafe_from_address=Int(published[].address)
            )
        grown = max(needed, 2 * Int(published[].bytes))
        slot = published
    else:
        var created = alloc(AllocLayout[StagingSlot].single()).unsafe_leak()
        created.unsafe_write(StagingSlot(address=0, bytes=0))
        external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
            StringSpan(name), created.unsafe_bitcast[NoneType]()
        )
        slot = created

    # Stream-ordered, like the staging the plain allreduce allocates for itself:
    # the launches that read and write the workspace are enqueued after this on
    # the same device's stream, and nothing here waits on the host.
    var grown_elements = (grown + size_of[dtype]() - 1) // size_of[dtype]()
    var buf = ctx.enqueue_create_buffer[dtype](grown_elements)
    var address = Int(buf.unsafe_ptr())
    # The workspace outlives every handle this module hands out, so its owning
    # handle is parked where no destructor reaches it.
    var park = alloc(AllocLayout[DeviceBuffer[dtype]].single()).unsafe_leak()
    park.unsafe_write(buf^)
    if case .Some(published) = slot:
        published[].address = address
        published[].bytes = grown_elements * size_of[dtype]()
    return MutPointer[Scalar[dtype], MutUntrackedOrigin](
        unsafe_from_address=address
    )
