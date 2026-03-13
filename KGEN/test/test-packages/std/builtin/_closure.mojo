# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct __ParameterClosureCaptureList[
    fn_type: __mlir_type.`!kgen.non_struct_type`, fn_ref: fn_type
](ImplicitlyCopyable, RegisterPassable):
    var value: __mlir_type.`!kgen.pointer<none>`

    # Parameter closure invariant requires this function be marked 'capturing'.
    @parameter
    @always_inline
    def __init__(out self):
        self.value = __mlir_op.`kgen.capture_list.create`[callee=Self.fn_ref]()

    @always_inline
    def __init__(out self, *, copy: Self):
        self.value = __mlir_op.`kgen.capture_list.copy`[callee=Self.fn_ref](
            copy.value
        )

    @always_inline
    def __del__(deinit self):
        __mlir_op.`pop.aligned_free`(self.value)

    @always_inline("nodebug")
    def expand(self):
        __mlir_op.`kgen.capture_list.expand`(self.value)


def __closure_wrapper_noop_dtor(self: __mlir_type.`!kgen.pointer<none>`, /):
    pass


def __closure_wrapper_noop_copy(
    *, copy: __mlir_type.`!kgen.pointer<none>`
) -> __mlir_type.`!kgen.pointer<none>`:
    return copy
