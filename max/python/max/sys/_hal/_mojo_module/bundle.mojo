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
"""Python projection of HAL ``RuntimeBundle``."""

from std.memory import ArcPointer, UnsafePointer, forget_deinit
from std.os import abort
from std.python import PythonObject
from _hal.context import (
    Context as HALContext,
    RuntimeBundle as HALRuntimeBundle,
)
from _hal.device import get_device_spec

from .context import Context


@fieldwise_init
struct Bundle(Movable, Writable):
    """Python projection of HAL ``RuntimeBundle``.

    A loaded device module containing one or more compiled functions.

    Holds two Arcs to defeat teardown-order hazards in the embedder:
      - ``_arc``: keeps the HAL ``RuntimeBundle`` loaded.
      - ``_ctx``: keeps the parent HAL ``Context`` alive past the
        bundle's destructor.
    """

    # TODO: generalize to multi-device — currently hardcoded to device 0.
    comptime device_spec = get_device_spec[0]()

    var _arc: ArcPointer[HALRuntimeBundle]
    var _ctx: ArcPointer[HALContext[Self.device_spec]]
    var _function_name: String

    @staticmethod
    def _self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Bundle method receiver was not a Bundle: ", e))

    @staticmethod
    def get_function_name(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
    ) raises -> PythonObject:
        return PythonObject(self_ptr[]._function_name)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Bundle(function_name=", self._function_name, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def compile_to_python_bundle[
    fn_type: TrivialRegisterPassable,
    func: fn_type,
](ctx_obj: PythonObject) raises -> PythonObject:
    """Compile a Mojo kernel and return a Python ``Bundle``.

    Instantiating this function with a kernel as the comptime ``func``
    parameter causes the Mojo compiler to emit the kernel's GPU bytecode
    and embed it into the containing shared library at build time. At
    Python runtime the resulting function loads that embedded bytecode
    onto the device, captures the compiled symbol's mangled name, and
    packages both into the returned ``Bundle``.

    Kernel modules expose one instantiation per kernel via a thin thunk
    registered with ``PythonModuleBuilder.def_function``::

        def vec_add_kernel(...): ...

        def vec_add_compiled(ctx_obj: PythonObject) raises -> PythonObject:
            return compile_to_python_bundle[
                type_of(vec_add_kernel), vec_add_kernel
            ](ctx_obj)

    ``vec_add_compiled`` is invoked from Python via
    ``ctx.compile(vec_add_compiled)``. The Python ``Context.compile``
    wrapper extracts the inner Mojo ``Context`` PyObject from its
    ``_inner`` attribute before calling the thunk, providing the value
    required by the ``downcast_value_ptr[Context]()`` call below.

    Parameters:
        fn_type: The Mojo type of the kernel function.
        func: The kernel function itself.

    Args:
        ctx_obj: The inner Mojo ``Context`` PyObject.

    Returns:
        A Python ``Bundle`` containing the loaded device module and the
        compiled function's mangled symbol name (accessible via
        ``bundle.function_name``).
    """
    var ctx_ptr = ctx_obj.downcast_value_ptr[Context]()
    var ctx_arc = ctx_ptr[]._arc
    var compile_result = ctx_arc[].compile[fn_type, func]()
    var name = String(compile_result[1].function_name)

    # Move the RuntimeBundle out of the tuple. `var (a, b) = ...`
    # would implicitly copy each element, and RuntimeBundle isn't
    # ImplicitlyCopyable — so use take_pointee + forget_deinit to
    # consume index [0] and skip the tuple's destructor (which would
    # otherwise re-drop the already-moved bundle).
    var hal_bundle = UnsafePointer(to=compile_result[0]).take_pointee()
    forget_deinit(compile_result^)

    return PythonObject(
        alloc=Bundle(
            _arc=ArcPointer(hal_bundle^),
            _ctx=ctx_arc^,
            _function_name=name,
        )
    )
