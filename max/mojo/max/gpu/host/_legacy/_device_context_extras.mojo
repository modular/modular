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

from std.builtin.device_passable import DevicePassable
from std.collections.optional import OptionalReg
from std.reflection import get_linkage_name, call_location, SourceLocation
from std.sys import size_of

from . import (
    Dim,
    LaunchAttribute,
    ConstantMemoryMapping,
    FuncAttribute,
)

from .._launch_args import _IdentityWrapper
from .device_context import (
    DeviceContext,
    DeviceContextBase,
    DeviceBuffer,
    DeviceFunction,
    DeviceExternalFunction,
    _checked,
    _DumpPath,
    _check_dim,
    _FunctionEnqueuer,
)


__extension DeviceExternalFunction:
    def _check_supports_default_call_with_pack(self):
        pass


__extension DeviceContextBase:
    comptime FunctionWrapper = _IdentityWrapper

    @inline(.always)
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        HostArgs: TypeList[Trait=RegisterPassable, ...],
        //,
        func: def(*args: *declared_arg_types) thin -> None,
        *encoded_types: DevicePassable,
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *encoded_args: *encoded_types,
        host_args: Tuple[*HostArgs] = (),
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: Optional[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device, encoding
        `DevicePassable` args and passing the trailing host arguments by host
        layout.

        You can pass the function directly to `enqueue_function`
        without compiling it first:

        ```mojo
        from max.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile it
        first to remove the overhead:

        ```mojo
        from max.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        Put capturing closures in `host_args`. The named kernel is compiled
        against those host callables; encoding them would rewrite captures to a
        synthesized `device_type` the kernel cannot read. Kernel parameters
        must be encoded args first, then the host arguments.

        Build `host_args` inline in the call with a tuple:

        ```mojo
        ctx.enqueue_function[kernel](
            buffer,
            host_args=(epilogue_fn,),
            grid_dim=1,
            block_dim=1,
        )
        ```

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            HostArgs: The host-layout types of the trailing arguments.
            func: The function to compile and launch.
            encoded_types: The `DevicePassable` argument types encoded to
                device layout.
            link_options: Additional linker flags and options as a string.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            self: The DeviceContext.
            encoded_args: The variadic `DevicePassable` arguments encoded to
                device layout.
            host_args: The trailing arguments passed by host layout.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.
            location: Source location for the function call.

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )
        var inferred_func_attribute = self._func_attribute_for_launch(
            func_attribute, shared_mem_bytes
        )
        var gpu_kernel = self.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            link_options=link_options,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=inferred_func_attribute)
        gpu_kernel._call_with_pack_checked(
            self,
            *encoded_args,
            host_args=host_args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @inline(.always)
    def enqueue_function[
        *Ts: DevicePassable
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: Optional[SourceLocation] = None,
    ) raises:
        """Enqueues a pre-compiled checked function for execution on this device.

        This overload requires a `DeviceFunction` that was compiled with
        type checking enabled (via `compile_function`). The function
        will verify that the argument types match the declared types at
        compile time.

        Parameters:
            Ts: Argument dtypes.

        Args:
            self: The DeviceContext.
            f: The compiled function to execute.
            args: Arguments to pass to the function.
            grid_dim: Dimensions of the compute grid, made up of thread
                blocks.
            block_dim: Dimensions of each thread block in the grid.
            cluster_dim: Dimensions of clusters (if the thread blocks are
                grouped into clusters).
            shared_mem_bytes: Amount of shared memory per thread block.
            attributes: Launch attributes.
            constant_memory: Constant memory mapping.
            location: Source location for the function call.

        ```mojo
        from max.gpu.host import DeviceContext

        def kernel(x: Int):
            print("Value:", x)

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compiled_func, 42, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        f._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )
