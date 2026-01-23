# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s

# Integration tests for @align decorator - verifies runtime behavior.

from sys import align_of, size_of
from memory import UnsafePointer, alloc
from testing import assert_equal, assert_true, TestSuite


# Basic aligned struct
@align(64)
struct CacheAligned(Movable):
    var x: Int

    fn __init__(out self, x: Int):
        self.x = x

    fn __moveinit__(out self, owned existing: Self):
        self.x = existing.x


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

    fn __init__(out self, var inner: CacheAligned, other: Int):
        self.inner = inner^
        self.other = other


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

    fn __init__(out self, var inner: CacheAligned):
        self.inner = inner^


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
    assert_equal(align_of[CacheAligned](), 64)
    assert_equal(align_of[AlignedTrivial](), 32)
    assert_equal(align_of[PageAligned](), 4096)
    assert_equal(align_of[AlignedGeneric[Int]](), 128)


fn test_nested_alignment() raises:
    """Test that containing structs inherit alignment from aligned fields."""
    # ContainsAligned should have at least 64-byte alignment due to CacheAligned field
    assert_equal(align_of[ContainsAligned](), 64)

    # OuterSmallAlign specifies @align(16) but contains CacheAligned (64-byte)
    # The actual alignment should be max(16, 64) = 64
    assert_equal(align_of[OuterSmallAlign](), 64)


fn test_heap_allocation_alignment() raises:
    """Test that heap-allocated aligned structs are actually aligned at runtime.

    The `alloc[T]()` function uses `align_of[T]()` as the default alignment,
    so heap allocations should respect the @align decorator.
    """
    # Allocate on heap - should be 64-byte aligned
    var heap_ptr = alloc[CacheAligned](1)
    var heap_addr = Int(heap_ptr)
    assert_true(
        (heap_addr & 63) == 0, "CacheAligned should be 64-byte aligned on heap"
    )
    heap_ptr.free()

    # Large alignment on heap
    var page_ptr = alloc[PageAligned](1)
    var page_addr = Int(page_ptr)
    assert_true(
        (page_addr & 4095) == 0,
        "PageAligned should be 4096-byte aligned on heap",
    )
    page_ptr.free()


fn test_stack_allocation_alignment() raises:
    """Test that stack-allocated aligned structs respect @align.

    The compiler propagates alignment from @align(N) decorator to the LLVM
    alloca instructions, ensuring stack allocations are properly aligned.
    """
    var cache_aligned = CacheAligned(42)
    var ptr = UnsafePointer(to=cache_aligned)
    var addr = Int(ptr)

    # Stack allocation should respect @align(64)
    assert_true(
        (addr & 63) == 0, "CacheAligned should be 64-byte aligned on stack"
    )

    # Test large alignment on stack (4096-byte aligned)
    var page_aligned = PageAligned(99)
    var page_ptr = UnsafePointer(to=page_aligned)
    var page_addr = Int(page_ptr)
    assert_true(
        (page_addr & 4095) == 0,
        "PageAligned should be 4096-byte aligned on stack",
    )


fn test_generic_alignment() raises:
    """Test that alignment works correctly with generic types."""
    # Different instantiations should all have 128-byte alignment
    assert_equal(align_of[AlignedGeneric[Int8]](), 128)
    assert_equal(align_of[AlignedGeneric[Int64]](), 128)
    assert_equal(align_of[AlignedGeneric[SIMD[DType.float32, 4]]](), 128)


fn test_generic_stack_allocation() raises:
    """Test that generic aligned structs are properly aligned on stack."""
    # Stack-allocate a generic aligned struct
    var generic_aligned = AlignedGeneric[Int](42)
    var ptr = UnsafePointer(to=generic_aligned)
    var addr = Int(ptr)

    # Should be 128-byte aligned
    assert_true(
        (addr & 127) == 0,
        "AlignedGeneric[Int] should be 128-byte aligned on stack",
    )


fn test_array_alignment() raises:
    """Test array allocation alignment behavior.

    Note: @align(N) affects the alignment of allocations but does NOT pad the
    struct size. This means arrays of aligned structs will have the base pointer
    aligned, but subsequent elements use the natural struct size as stride.

    To get each array element aligned, the struct must be explicitly padded
    (e.g., by adding padding fields) so that size_of[T]() is a multiple of
    align_of[T](). This matches how C++ alignas works with arrays.
    """
    # Allocate array - base pointer should be 64-byte aligned
    var arr = alloc[CacheAligned](4)
    var base_addr = Int(arr)
    assert_true(
        (base_addr & 63) == 0,
        "CacheAligned array base should be 64-byte aligned",
    )

    # Stride is size_of[CacheAligned]() = 8 (just one Int), not 64
    # This is expected - @align doesn't pad struct size
    var stride = Int(arr + 1) - Int(arr)
    assert_equal(stride, 8)

    arr.free()


fn test_cross_struct_alignment() raises:
    """Test that alignment works when a struct uses a later-defined aligned struct.

    This exercises the code path where we look up alignment from the symbol
    table (struct not yet lowered) rather than from structDecls.
    """
    # This calls a method that creates a LaterAlignedStruct (256-byte aligned)
    # The alignment lookup must work even though UsesLaterStruct is defined
    # before LaterAlignedStruct.
    var result = UsesLaterStruct.create_later()
    assert_equal(result, 1, "LaterAlignedStruct should be 256-byte aligned")


fn test_inherited_stack_alignment() raises:
    """Test that stack allocation respects alignment inherited from fields.

    This is the key test for MOCO-3165: a struct containing an @align(64) field
    should be allocated on the stack with 64-byte alignment, even if the
    containing struct has no explicit @align decorator.
    """
    # ContainsAligned has no @align decorator but contains CacheAligned which
    # has @align(64). The containing struct should inherit this alignment
    # requirement for stack allocation.
    var container = ContainsAligned(CacheAligned(42), 99)
    var ptr = UnsafePointer(to=container)
    var addr = Int(ptr)

    # The struct should be 64-byte aligned due to the CacheAligned field
    assert_true(
        (addr & 63) == 0,
        (
            "ContainsAligned should inherit 64-byte alignment from CacheAligned"
            " field"
        ),
    )

    # Also verify the inner field is aligned
    var inner_ptr = UnsafePointer(to=container.inner)
    var inner_addr = Int(inner_ptr)
    assert_true(
        (inner_addr & 63) == 0,
        "ContainsAligned.inner field should be 64-byte aligned",
    )

    # Test OuterSmallAlign which has @align(16) but contains CacheAligned (64)
    # The effective alignment should be max(16, 64) = 64
    var outer = OuterSmallAlign(CacheAligned(77))
    var outer_ptr = UnsafePointer(to=outer)
    var outer_addr = Int(outer_ptr)
    assert_true(
        (outer_addr & 63) == 0,
        "OuterSmallAlign should use max(explicit=16, inherited=64) = 64",
    )


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
