# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s

# Integration tests for @align decorator - verifies runtime behavior.

from sys import align_of, size_of
from memory import UnsafePointer, alloc


# Basic aligned struct
@align(64)
struct CacheAligned:
    var x: Int

    fn __init__(out self, x: Int):
        self.x = x


# @align works on single-element @register_passable structs. When @align is
# specified, the struct is NOT flattened to its element type during lowering,
# preserving the alignment metadata.
@align(32)
@register_passable
struct AlignedTrivial:
    var value: Int


# Large alignment
@align(4096)
struct PageAligned:
    var data: Int

    fn __init__(out self, data: Int):
        self.data = data


# Struct containing an aligned struct (should inherit alignment)
struct ContainsAligned:
    var inner: CacheAligned
    var other: Int


# Generic struct with alignment
@align(128)
struct AlignedGeneric[T: __TypeOfAllTypes]:
    var value: Self.T

    fn __init__(out self, value: Self.T):
        self.value = value


# Nested alignment - outer has smaller alignment than inner's requirement
@align(16)
struct OuterSmallAlign:
    var inner: CacheAligned  # CacheAligned requires 64-byte alignment


# Test cross-struct references: UsesLaterStruct is defined before LaterAlignedStruct.
# This tests that alignment lookup works for structs not yet lowered.
struct UsesLaterStruct:
    """A struct whose method creates a local of a later-defined aligned struct.
    """

    @staticmethod
    fn create_later() -> Int:
        """Create a local variable of LaterAlignedStruct and verify alignment.
        """
        var later = LaterAlignedStruct(123)
        var ptr = UnsafePointer(to=later)
        var addr = Int(ptr)
        # Return 1 if aligned, 0 if not
        return 1 if (addr & 255) == 0 else 0


@align(256)
struct LaterAlignedStruct:
    """An aligned struct defined after UsesLaterStruct."""

    var value: Int

    fn __init__(out self, value: Int):
        self.value = value


fn test_align_of() raises:
    """Test that align_of[T]() reflects the @align decorator."""
    print("=== test_align_of ===")

    # CHECK: CacheAligned alignment: 64
    print("CacheAligned alignment:", align_of[CacheAligned]())

    # CHECK: AlignedTrivial alignment: 32
    print("AlignedTrivial alignment:", align_of[AlignedTrivial]())

    # CHECK: PageAligned alignment: 4096
    print("PageAligned alignment:", align_of[PageAligned]())

    # CHECK: AlignedGeneric[Int] alignment: 128
    print("AlignedGeneric[Int] alignment:", align_of[AlignedGeneric[Int]]())


fn test_nested_alignment() raises:
    """Test that containing structs inherit alignment from aligned fields."""
    print("=== test_nested_alignment ===")

    # ContainsAligned should have at least 64-byte alignment due to CacheAligned field
    # CHECK: ContainsAligned alignment: 64
    print("ContainsAligned alignment:", align_of[ContainsAligned]())

    # OuterSmallAlign specifies @align(16) but contains CacheAligned (64-byte)
    # The actual alignment should be max(16, 64) = 64
    # CHECK: OuterSmallAlign alignment: 64
    print("OuterSmallAlign alignment:", align_of[OuterSmallAlign]())


fn test_heap_allocation_alignment() raises:
    """Test that heap-allocated aligned structs are actually aligned at runtime.

    The `alloc[T]()` function uses `align_of[T]()` as the default alignment,
    so heap allocations should respect the @align decorator.
    """
    print("=== test_heap_allocation_alignment ===")

    # Allocate on heap - should be 64-byte aligned
    var heap_ptr = alloc[CacheAligned](1)
    var heap_addr = Int(heap_ptr)
    var heap_is_aligned = (heap_addr & 63) == 0
    # CHECK: CacheAligned heap aligned: True
    print("CacheAligned heap aligned:", heap_is_aligned)
    heap_ptr.free()

    # Large alignment on heap
    var page_ptr = alloc[PageAligned](1)
    var page_addr = Int(page_ptr)
    var page_is_aligned = (page_addr & 4095) == 0
    # CHECK: PageAligned heap aligned: True
    print("PageAligned heap aligned:", page_is_aligned)
    page_ptr.free()


fn test_stack_allocation_alignment() raises:
    """Test that stack-allocated aligned structs respect @align.

    The compiler propagates alignment from @align(N) decorator to the LLVM
    alloca instructions, ensuring stack allocations are properly aligned.
    """
    print("=== test_stack_allocation_alignment ===")

    var cache_aligned = CacheAligned(42)
    var ptr = UnsafePointer(to=cache_aligned)
    var addr = Int(ptr)

    # Stack allocation should respect @align(64)
    var is_aligned = (addr & 63) == 0
    # CHECK: CacheAligned stack aligned: True
    print("CacheAligned stack aligned:", is_aligned)

    # Test large alignment on stack (4096-byte aligned)
    var page_aligned = PageAligned(99)
    var page_ptr = UnsafePointer(to=page_aligned)
    var page_addr = Int(page_ptr)
    var page_is_aligned = (page_addr & 4095) == 0
    # CHECK: PageAligned stack aligned: True
    print("PageAligned stack aligned:", page_is_aligned)


fn test_generic_alignment() raises:
    """Test that alignment works correctly with generic types."""
    print("=== test_generic_alignment ===")

    # Different instantiations should all have 128-byte alignment
    # CHECK: AlignedGeneric[Int8] alignment: 128
    print("AlignedGeneric[Int8] alignment:", align_of[AlignedGeneric[Int8]]())

    # CHECK: AlignedGeneric[Int64] alignment: 128
    print("AlignedGeneric[Int64] alignment:", align_of[AlignedGeneric[Int64]]())

    # CHECK: AlignedGeneric[SIMD[DType.float32, 4]] alignment: 128
    print(
        "AlignedGeneric[SIMD[DType.float32, 4]] alignment:",
        align_of[AlignedGeneric[SIMD[DType.float32, 4]]](),
    )


fn test_generic_stack_allocation() raises:
    """Test that generic aligned structs are properly aligned on stack."""
    print("=== test_generic_stack_allocation ===")

    # Stack-allocate a generic aligned struct
    var generic_aligned = AlignedGeneric[Int](42)
    var ptr = UnsafePointer(to=generic_aligned)
    var addr = Int(ptr)

    # Should be 128-byte aligned
    var is_aligned = (addr & 127) == 0
    # CHECK: AlignedGeneric[Int] stack aligned: True
    print("AlignedGeneric[Int] stack aligned:", is_aligned)


fn test_array_alignment() raises:
    """Test array allocation alignment behavior.

    Note: @align(N) affects the alignment of allocations but does NOT pad the
    struct size. This means arrays of aligned structs will have the base pointer
    aligned, but subsequent elements use the natural struct size as stride.

    To get each array element aligned, the struct must be explicitly padded
    (e.g., by adding padding fields) so that size_of[T]() is a multiple of
    align_of[T](). This matches how C++ alignas works with arrays.
    """
    print("=== test_array_alignment ===")

    # Allocate array - base pointer should be 64-byte aligned
    var arr = alloc[CacheAligned](4)
    var base_addr = Int(arr)
    var base_aligned = (base_addr & 63) == 0
    # CHECK: CacheAligned array base aligned: True
    print("CacheAligned array base aligned:", base_aligned)

    # Stride is size_of[CacheAligned]() = 8 (just one Int), not 64
    # This is expected - @align doesn't pad struct size
    var stride = Int(arr + 1) - Int(arr)
    # CHECK: CacheAligned array stride: 8
    print("CacheAligned array stride:", stride)

    arr.free()


fn test_cross_struct_alignment() raises:
    """Test that alignment works when a struct uses a later-defined aligned struct.

    This exercises the code path where we look up alignment from the symbol
    table (struct not yet lowered) rather than from structDecls.
    """
    print("=== test_cross_struct_alignment ===")

    # This calls a method that creates a LaterAlignedStruct (256-byte aligned)
    # The alignment lookup must work even though UsesLaterStruct is defined
    # before LaterAlignedStruct.
    var result = UsesLaterStruct.create_later()
    # CHECK: LaterAlignedStruct stack aligned in UsesLaterStruct: True
    print("LaterAlignedStruct stack aligned in UsesLaterStruct:", result == 1)


fn main() raises:
    test_align_of()
    test_nested_alignment()
    test_heap_allocation_alignment()
    test_stack_allocation_alignment()
    test_generic_alignment()
    test_generic_stack_allocation()
    test_array_alignment()
    test_cross_struct_alignment()
    # CHECK: All tests passed
    print("All tests passed")
