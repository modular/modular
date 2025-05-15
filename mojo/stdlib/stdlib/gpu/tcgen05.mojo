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

from sys import _RegisterPackType
from sys._assembly import inlined_assembly
from sys.info import _has_blackwell_tcgen05
from memory import UnsafePointer, bitcast
from gpu.memory import AddressSpace, external_memory
from gpu.mma import _str_iota  # TODO: move to a string module

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
fn tcgen05_alloc[
    cta_group: Int32
](
    ptr_tmem_addr: UnsafePointer[
        UInt32, address_space = AddressSpace.SHARED, alignment=16
    ],
    num_cols: UInt32,
):
    """Allocates tensor memory for use with tcgen05 instructions.

    Parameters:
        cta_group: The cooperative thread array (CTA) group ID.

    Args:
        ptr_tmem_addr: Shared memory pointer to hold tensor memory address.
        num_cols: The number of columns to allocate.

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
        UInt32(Int(ptr_tmem_addr)),
        num_cols,
    )


@always_inline
fn tcgen05_dealloc[cta_group: Int32](tmem_addr: UInt32, num_cols: UInt32):
    """Deallocates tensor memory allocated by tcgen05_alloc().

    This function deallocates tensor memory that was previously allocated using
    tcgen05_alloc(). The deallocation must be performed by the same CTA group
    that performed the allocation.

    Parameters:
        cta_group: The cooperative thread array (CTA) group ID.

    Args:
        tmem_addr: Address of the tensor memory to deallocate.
        num_cols: Number of columns in the tensor memory.
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
    ](tmem_addr, num_cols)


@always_inline
fn tcgen05_ld[
    *,
    datapaths: Int,
    bits: Int,
    repeat: Int,
    type: DType,
    pack: Bool,
    width: Int = (datapaths * bits * repeat) // (32 * 32),
](tmem_addr: UInt32) -> SIMD[type, width]:
    """Loads data from tensor memory into registers.

    Parameters:
        datapaths: The first dimension of the shape.
        bits: The second dimension of the shape.
        repeat: The repeat factor.
        type: The data type to load.
        pack: Whether to pack two 16-bit chunks of adjacent columns into a single 32-bit register.
        width: The nubmer elements in the result vector.

    Args:
        tmem_addr: The address of the tensor memory to load from.

    Returns:
        The SIMD register containing the loaded data.
    """
    check_blackwell_constraint()

    constrained[
        (datapaths == 16 and bits == 64)
        or (datapaths == 16 and bits == 128)
        or (datapaths == 16 and bits == 256)
        or (datapaths == 32 and bits == 32),
        "`datapaths`x`bits`b must be 16x64b, 16x128b, 16x256b or 32x32b.",
    ]()

    constrained[
        repeat in [1, 2, 4, 8, 16, 32, 64, 128],
        "`repeat` must be a power of 2 in the range [1, 128].",
    ]()

    constrained[
        width in [1, 2, 4, 8, 16, 32, 64],
        "`width` must be a power of 2 in the range [1, 64].",
    ]()

    constrained[
        width == (repeat * bits * datapaths) // (32 * 32)
        and sizeof[type]() == 4,
        (
            "Only support 4B data type and width must be equal to (num * n * m)"
            " // (32 * 32)."
        ),
    ]()

    alias shape_str = String(datapaths) + "x" + String(bits)
    alias num_str = String(repeat)
    alias pack_str = ".pack::16b" if pack else ""
    alias constraints_str = "=r," * width + "r"
    alias output_args_str = "{" + _str_iota[width, prefix="$", sep=","]() + "},"
    alias addr_str = "[$" + String(width) + "]"

    @parameter
    fn call_ld_intrinsic[pack_type: AnyTrivialRegType]() -> SIMD[type, width]:
        var r = inlined_assembly[
            "tcgen05.ld.sync.aligned."
            + shape_str
            + "b.x"
            + num_str
            + pack_str
            + ".b32 "
            + output_args_str
            + addr_str
            + ";",
            pack_type,
            constraints=constraints_str,
            has_side_effect=True,
        ](tmem_addr)
        return UnsafePointer(to=r).bitcast[SIMD[type, width]]()[]

    # fmt: off
    @parameter
    if width == 1:
        return call_ld_intrinsic[
                _RegisterPackType[UInt32]
            ]()
    elif width == 2:
        return call_ld_intrinsic[
                _RegisterPackType[UInt32, UInt32]
            ]()
    elif width == 4:
        return  call_ld_intrinsic[
                _RegisterPackType[UInt32, UInt32, UInt32, UInt32]
            ]()
    elif width == 8:
        return call_ld_intrinsic[
                _RegisterPackType[UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32]
             ]()
    elif width == 16:
        return call_ld_intrinsic[
                _RegisterPackType[UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32
            ]
        ]()
    elif width == 32:
        return call_ld_intrinsic[
                _RegisterPackType[UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32
            ]
        ]()
    else:
        return call_ld_intrinsic[
                _RegisterPackType[UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32
            ]
        ]()
    # fmt: on


@always_inline
fn tcgen05_release_allocation_lock():
    """Releases the allocation lock for the current CTA group.

    Note:
        This function is only available on NVIDIA Blackwell GPUs (SM 100+).
    """
    check_blackwell_constraint()

    inlined_assembly[
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;",
        NoneType,
        has_side_effect=True,
        constraints="",
    ]()


@always_inline
fn tcgen05_load_wait():
    """Waits for tensor memory loads to complete.

    Note:
        This function is only available on NVIDIA Blackwell GPUs (SM 100+).
    """
    check_blackwell_constraint()

    inlined_assembly[
        "tcgen05.wait::ld.sync.aligned;",
        NoneType,
        has_side_effect=True,
        constraints="",
    ]()


@always_inline
fn tcgen05_store_wait():
    """Waits for tensor memory stores to complete.

    Note:
        This function is only available on NVIDIA Blackwell GPUs (SM 100+).
    """
    check_blackwell_constraint()

    inlined_assembly[
        "tcgen05.wait::st.sync.aligned;",
        NoneType,
        has_side_effect=True,
        constraints="",
    ]()
