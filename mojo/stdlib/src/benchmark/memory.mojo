# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------===#
# clobber_memory
# ===----------------------------------------------------------------------===#


@always_inline
fn clobber_memory():
    """Forces all pending memory writes to be flushed to memory.

    This ensures that the compiler does not optimize away memory writes if it
    deems them to be not neccessary. In effect, this operation acts a barrier
    to memory reads and writes.
    """

    # This opereration corresponds to  atomic_signal_fence(memory_order_acq_rel)
    # in C++.
    __mlir_op.`pop.fence`[
        _type=None,
        syncscope = "singlethread".value,
        ordering = __mlir_attr.`#pop<atomic_ordering acq_rel>`,
    ]()
