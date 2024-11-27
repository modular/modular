# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import NDBuffer


@always_inline
fn get_batch_from_row_offsets(
    row_offsets: NDBuffer[DType.uint32, 1, *_], tok_idx: Int
) -> Int:
    """Calculate the batch_idx for the given flattened token_idx using row_offsets.
    """
    var row_offsets_size = row_offsets.dim[0]()

    debug_assert(
        tok_idx >= 0 and tok_idx < int(row_offsets[row_offsets_size - 1]),
        "tok_idx is out of range of row_offsets",
    )

    var low: UInt = 0
    var high: UInt = row_offsets_size - 1
    while low + 1 != high:
        var mid = (low + high) // 2

        if tok_idx >= int(row_offsets[mid]):
            low = mid
        elif tok_idx < int(row_offsets[mid]):
            high = mid

    return Int(low)
