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

"""This module includes utilities for working with the
tensorcore 5th generation (tcgen05) instructions."""

from sys._assembly import inlined_assembly
from sys.info import _has_blackwell_tcgen05
from memory import UnsafePointer, bitcast
from gpu.memory import AddressSpace, external_memory

alias check_blackwell_constraint = constrained[
    _has_blackwell_tcgen05(),
    (
        "The tcgen05 instructions are only applicable on nVidia Blackwell"
        " (sm_100a, sm_101a) hardware."
    ),
]


@register_passable("trivial")
struct TensorMemory:
    """A wrapper around tensor memory allocated for tcgen05 instructions."""

    var ptr: UnsafePointer[
        UInt32, address_space = AddressSpace.SHARED, alignment=16
    ]
    """Pointer to the tensor memory address."""

    var num_cols: UInt32
    """The number of columns in the tensor memory."""

    @always_inline
    fn __init__(out self, num_cols: UInt32):
        """Initialize the TensorMemory struct.

        Args:
            num_cols: The number of columns to allocate.
        """
        # Bitcast to avoid `cannot implicitly convert` error.
        self.ptr = external_memory[
            UInt32, address_space = AddressSpace.SHARED, alignment=16
        ]().bitcast[UInt32]()
        self.num_cols = num_cols


@always_inline
fn tcgen05_alloc[cta_group: Int32](mut tmem: TensorMemory):
    """Allocates tensor memory for use with tcgen05 instructions.

    Parameters:
        cta_group: The cooperative thread array (CTA) group ID.

    Args:
        tmem: TensorMemory struct to hold the allocation address and number of
              columns.

    Note:
        This function is only available on NVIDIA Blackwell GPUs (SM 100+).
    """
    check_blackwell_constraint()
    constrained[cta_group == 1 or cta_group == 2, "cta_group must be 1 or 2"]()
    inlined_assembly[
        "tcgen05.alloc.cta_group::"
        + String(cta_group)
        + ".sync.aligned.shared::cta.b32 [$0], $1;",
        NoneType,
        constraints="r,r",
        has_side_effect=True,
    ](
        UInt32(Int(tmem.ptr)),
        tmem.num_cols,
    )


@always_inline
fn tcgen05_dealloc[cta_group: Int32](mut tmem: TensorMemory):
    """Deallocates tensor memory allocated by tcgen05_alloc().

    This function deallocates tensor memory that was previously allocated using
    tcgen05_alloc(). The deallocation must be performed by the same CTA group
    that performed the allocation.

    Parameters:
        cta_group: The cooperative thread array (CTA) group ID.

    Args:
        tmem: TensorMemory struct to hold the allocation address and number of
              columns.

    Note:
        This function is only available on NVIDIA Blackwell GPUs (SM 100+).
    """
    check_blackwell_constraint()
    constrained[cta_group == 1 or cta_group == 2, "cta_group must be 1 or 2"]()
    inlined_assembly[
        "tcgen05.dealloc.cta_group::"
        + String(cta_group)
        + ".sync.aligned.b32 $0, $1;",
        NoneType,
        constraints="r,r",
        has_side_effect=True,
    ](tmem.ptr[0], tmem.num_cols)
