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


struct __ParameterClosureCaptureList[
    fn_type: __TypeOfAllTypes, fn_ref: fn_type
](ImplicitlyCopyable, RegisterPassable):
    var value: __mlir_type.`!kgen.pointer<none>`

    # Parameter closure invariant requires this function be marked 'capturing'.
    @parameter
    @always_inline
    fn __init__(out self):
        self.value = __mlir_op.`kgen.capture_list.create`[
            callee = Self.fn_ref
        ]()

    @always_inline
    fn __copyinit__(out self, copy: Self):
        self.value = __mlir_op.`kgen.capture_list.copy`[callee = Self.fn_ref](
            copy.value
        )

    @always_inline
    fn __del__(deinit self):
        __mlir_op.`pop.aligned_free`(self.value)

    @always_inline("nodebug")
    fn expand(self):
        __mlir_op.`kgen.capture_list.expand`(self.value)


fn __closure_wrapper_noop_dtor(self: __mlir_type.`!kgen.pointer<none>`, /):
    pass


fn __closure_wrapper_noop_copy(
    other: __mlir_type.`!kgen.pointer<none>`, /
) -> __mlir_type.`!kgen.pointer<none>`:
    return other


fn __ownership_keepalive[*Ts: AnyType](*args: *Ts):
    pass
