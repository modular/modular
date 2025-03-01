# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Implementation of the C++ backed DeviceContext in Mojo

from collections import List, Optional
from collections.string import StaticString
from pathlib import Path
from sys import env_get_int, env_get_string, external_call, is_defined, sizeof
from sys.compile import DebugLevel, OptimizationLevel
from sys.info import _get_arch, has_nvidia_gpu_accelerator, is_triple
from sys.param_env import _is_bool_like

from builtin._location import __call_location, _SourceLocation
from compile.compile import Info
from gpu.host._compile import (
    _compile_code,
    _compile_code_asm,
    _cross_compilation,
    _get_gpu_target,
    _ptxas_compile,
    _to_sass,
)
from memory import stack_allocation

from utils import Variant

from .info import DEFAULT_GPU


# Create empty structs to ensure type checking when using the C++ handles.
struct _DeviceContextCpp:
    pass


struct _DeviceBufferCpp:
    pass


struct _DeviceFunctionCpp:
    pass


struct _DeviceStreamCpp:
    pass


struct _DeviceTimerCpp:
    pass


alias _DeviceContextPtr = UnsafePointer[_DeviceContextCpp]
alias _DeviceBufferPtr = UnsafePointer[_DeviceBufferCpp]
alias _DeviceFunctionPtr = UnsafePointer[_DeviceFunctionCpp]
alias _DeviceStreamPtr = UnsafePointer[_DeviceStreamCpp]
alias _DeviceTimerPtr = UnsafePointer[_DeviceTimerCpp]
alias _CharPtr = UnsafePointer[UInt8]
alias _IntPtr = UnsafePointer[Int32]
alias _VoidPtr = UnsafePointer[NoneType]
alias _SizeT = UInt

# Define helper methods to call AsyncRT bindings.


@always_inline
fn _checked(
    err: _CharPtr,
    *,
    msg: String = "",
    location: OptionalReg[_SourceLocation] = None,
) raises:
    if err:
        _raise_checked_impl(err, msg, location.or_else(__call_location[2]()))


@no_inline
fn _raise_checked_impl(
    err_msg: _CharPtr, msg: String, location: _SourceLocation
) raises:
    var err = String(StaticString(unsafe_from_utf8_ptr=err_msg))
    # void AsyncRT_DeviceContext_strfree(const char* ptr)
    external_call["AsyncRT_DeviceContext_strfree", NoneType, _CharPtr](err_msg)
    raise Error(location.prefix(err + ((" " + msg) if msg else "")))


struct _DeviceTimer:
    var _handle: _DeviceTimerPtr

    @implicit
    fn __init__(out self, ptr: _DeviceTimerPtr):
        self._handle = ptr

    fn __del__(owned self):
        # void AsyncRT_DeviceTimer_release(const DviceTimer *timer)
        external_call["AsyncRT_DeviceTimer_release", NoneType, _DeviceTimerPtr](
            self._handle
        )


@value
struct _DeviceBufferMode:
    var _mode: Int

    alias _SYNC = _DeviceBufferMode(0)
    alias _ASYNC = _DeviceBufferMode(1)
    alias _HOST = _DeviceBufferMode(2)

    fn __eq__(self, other: Self) -> Bool:
        return self._mode == other._mode


struct DeviceBuffer[
    type: DType,
    address_space: AddressSpace = AddressSpace.GENERIC,
    mut: Bool = True,
    origin: Origin[mut] = Origin[mut].cast_from[MutableAnyOrigin].result,
](Sized):
    """Represents a block of device-resident storage. For GPU devices, a device
    buffer is allocated in the device's global memory.

    To allocate a `DeviceBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_buffer()`](/mojo/stdlib/gpu/host/device_context/DeviceContext#enqueue_create_buffer).

    Parameters:
        type: Data type to be stored in the buffer.
        address_space: The address space of the underlying pointer.
        mut: The mutability of the underlying pointer.
        origin: The origin of the underlying pointer.
    """

    alias _DevicePtr = UnsafePointer[
        Scalar[type], address_space=address_space, mut=mut, origin=origin
    ]
    alias _HostMappedBufferType = _HostMappedBuffer[
        type, address_space=address_space, mut=mut, origin=origin
    ]

    # _device_ptr must be the first word in the struct to enable passing of
    # DeviceBuffer to kernels. The first word is passed to the kernel and
    # it needs to contain the value registered with the driver.
    var _device_ptr: Self._DevicePtr
    var _handle: _DeviceBufferPtr

    @doc_private
    @always_inline
    fn __init__(
        mut self,
        ctx: DeviceContext,
        size: Int,
        mode: _DeviceBufferMode,
    ) raises:
        """This init takes in a constructed `DeviceContext` and schedules an
        owned buffer allocation using the stream in the device context.
        """
        alias elem_size = sizeof[type]()
        var cpp_handle = _DeviceBufferPtr()
        var device_ptr = Self._DevicePtr()

        if mode == _DeviceBufferMode._SYNC:
            # const char *AsyncRT_DeviceContext_createBuffer_sync(const DeviceBuffer **result, void **device_ptr, const DeviceContext *ctx, size_t len, size_t elem_size)
            _checked(
                external_call[
                    "AsyncRT_DeviceContext_createBuffer_sync",
                    _CharPtr,
                    UnsafePointer[_DeviceBufferPtr],
                    UnsafePointer[Self._DevicePtr],
                    _DeviceContextPtr,
                    _SizeT,
                    _SizeT,
                ](
                    UnsafePointer.address_of(cpp_handle),
                    UnsafePointer.address_of(device_ptr),
                    ctx._handle,
                    size,
                    elem_size,
                )
            )
        elif mode == _DeviceBufferMode._ASYNC:
            # const char *AsyncRT_DeviceContext_createBuffer_async(const DeviceBuffer **result, void **device_ptr, const DeviceContext *ctx, size_t len, size_t elem_size)
            _checked(
                external_call[
                    "AsyncRT_DeviceContext_createBuffer_async",
                    _CharPtr,
                    UnsafePointer[_DeviceBufferPtr],
                    UnsafePointer[Self._DevicePtr],
                    _DeviceContextPtr,
                    _SizeT,
                    _SizeT,
                ](
                    UnsafePointer.address_of(cpp_handle),
                    UnsafePointer.address_of(device_ptr),
                    ctx._handle,
                    size,
                    elem_size,
                )
            )
        else:
            # const char *AsyncRT_DeviceContext_createHostBuffer(const DeviceBuffer **result, void **device_ptr, const DeviceContext *ctx, size_t len, size_t elem_size)
            _checked(
                external_call[
                    "AsyncRT_DeviceContext_createHostBuffer",
                    _CharPtr,
                    UnsafePointer[_DeviceBufferPtr],
                    UnsafePointer[Self._DevicePtr],
                    _DeviceContextPtr,
                    _SizeT,
                    _SizeT,
                ](
                    UnsafePointer.address_of(cpp_handle),
                    UnsafePointer.address_of(device_ptr),
                    ctx._handle,
                    size,
                    elem_size,
                )
            )

        self._device_ptr = device_ptr
        self._handle = cpp_handle

    @doc_private
    fn __init__(
        mut self, handle: _DeviceBufferPtr, device_ptr: Self._DevicePtr
    ):
        self._device_ptr = device_ptr
        self._handle = handle

    @doc_private
    fn __init__(
        mut self,
        ctx: DeviceContext,
        ptr: Self._DevicePtr,
        size: Int,
        *,
        owning: Bool,
    ):
        alias elem_size = sizeof[type]()
        var cpp_handle = _DeviceBufferPtr()
        # void AsyncRT_DeviceContext_createBuffer_owning(
        #     const DeviceBuffer **result, const DeviceContext *ctx,
        #     void *device_ptr, size_t len, size_t elem_size, bool owning)
        external_call[
            "AsyncRT_DeviceContext_createBuffer_owning",
            NoneType,
            UnsafePointer[_DeviceBufferPtr],
            _DeviceContextPtr,
            Self._DevicePtr,
            _SizeT,
            _SizeT,
            Bool,
        ](
            UnsafePointer.address_of(cpp_handle),
            ctx._handle,
            ptr,
            size,
            elem_size,
            owning,
        )

        self._device_ptr = ptr
        self._handle = cpp_handle

    fn __copyinit__(out self, existing: Self):
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceBuffer_retain(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_retain",
            NoneType,
            _DeviceBufferPtr,
        ](existing._handle)
        self._device_ptr = existing._device_ptr
        self._handle = existing._handle

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    fn __moveinit__(out self, owned existing: Self):
        self._device_ptr = existing._device_ptr
        self._handle = existing._handle

    @always_inline
    fn __del__(owned self):
        """This function schedules an owned buffer free using the stream in the
        device context.
        """
        # void AsyncRT_DeviceBuffer_release(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_release", NoneType, _DeviceBufferPtr
        ](
            self._handle,
        )

    fn __len__(self) -> Int:
        # int64_t AsyncRT_DeviceBuffer_bytesize(const DeviceBuffer *buffer)
        return (
            external_call[
                "AsyncRT_DeviceBuffer_bytesize", Int, _DeviceBufferPtr
            ](self._handle)
            // sizeof[type]()
        )

    @always_inline
    fn create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> DeviceBuffer[view_type]:
        alias elem_size = sizeof[view_type]()
        var new_handle = _DeviceBufferPtr()
        var new_device_ptr = UnsafePointer[Scalar[view_type]]()
        # const char *AsyncRT_DeviceBuffer_createSubBuffer(
        #     const DeviceBuffer **result, void **device_ptr,
        #     const DeviceBuffer *buf, size_t offset, size_t len, size_t elem_size)
        _checked(
            external_call[
                "AsyncRT_DeviceBuffer_createSubBuffer",
                _CharPtr,
                UnsafePointer[_DeviceBufferPtr],
                UnsafePointer[UnsafePointer[Scalar[view_type]]],
                _DeviceBufferPtr,
                _SizeT,
                _SizeT,
                _SizeT,
            ](
                UnsafePointer.address_of(new_handle),
                UnsafePointer.address_of(new_device_ptr),
                self._handle,
                offset,
                size,
                elem_size,
            )
        )
        return DeviceBuffer[view_type](new_handle, new_device_ptr)

    fn enqueue_copy_to(self, dst: DeviceBuffer[type, **_]) raises:
        # const char * AsyncRT_DeviceBuffer_copyTo(const DeviceBuffer* src, const DeviceBuffer *dst)
        _checked(
            external_call[
                "AsyncRT_DeviceBuffer_copyTo",
                _CharPtr,
                _DeviceBufferPtr,
                _DeviceBufferPtr,
            ](self._handle, dst._handle)
        )

    fn enqueue_fill(self, val: Scalar[type]) raises -> Self:
        self.context().enqueue_memset(self, val)
        return self

    fn reassign_ownership_to(self, ctx: DeviceContext) raises:
        # const char * AsyncRT_DeviceBuffer_reassignOwnershipTo(const DeviceBuffer *buf, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceBuffer_reassignOwnershipTo",
                _CharPtr,
                _DeviceBufferPtr,
                _DeviceContextPtr,
            ](self._handle, ctx._handle)
        )

    fn take_ptr(
        owned self,
    ) -> Self._DevicePtr:
        # void AsyncRT_DeviceBuffer_release_ptr(const DeviceBuffer *buffer)
        external_call[
            "AsyncRT_DeviceBuffer_release_ptr", NoneType, _DeviceBufferPtr
        ](self._handle)
        var result = self._device_ptr
        self._device_ptr = Self._DevicePtr()
        return result

    fn unsafe_ptr(
        self,
    ) -> Self._DevicePtr:
        return self._device_ptr

    fn context(self) raises -> DeviceContext:
        # const DeviceContext *AsyncRT_DeviceBuffer_context(const DeviceBuffer *buffer)
        var ctx_ptr: _DeviceContextPtr = external_call[
            "AsyncRT_DeviceBuffer_context", _DeviceContextPtr, _DeviceBufferPtr
        ](self._handle)
        return DeviceContext(ctx_ptr)

    fn map_to_host(
        self,
    ) raises -> Self._HostMappedBufferType:
        return Self._HostMappedBufferType(self.context(), self)

    fn __getitem__(self, idx: Int) -> Scalar[type]:
        return self._device_ptr[idx]

    fn __setitem__(
        self: DeviceBuffer[type, mut=True], idx: Int, val: Scalar[type]
    ):
        self._device_ptr[idx] = val


@doc_private
struct DeviceStream:
    var _handle: _DeviceStreamPtr

    @always_inline
    fn __init__(out self, ctx: DeviceContext) raises:
        var result = _DeviceStreamPtr()
        # const char *AsyncRT_DeviceContext_stream(const DeviceStream **result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stream",
                _CharPtr,
                UnsafePointer[_DeviceStreamPtr],
                _DeviceContextPtr,
            ](UnsafePointer.address_of(result), ctx._handle)
        )
        self._handle = result

    fn __copyinit__(out self, existing: Self):
        # void AsyncRT_DeviceStream_retain(const DeviceStream *stream)
        external_call[
            "AsyncRT_DeviceStream_retain",
            NoneType,
            _DeviceStreamPtr,
        ](existing._handle)
        self._handle = existing._handle

    fn __moveinit__(out self, owned existing: Self):
        self._handle = existing._handle

    @always_inline
    fn __del__(owned self):
        # void AsyncRT_DeviceStream_release(const DeviceStream *stream)
        external_call[
            "AsyncRT_DeviceStream_release", NoneType, _DeviceStreamPtr
        ](
            self._handle,
        )

    @always_inline
    fn synchronize(self) raises:
        # const char *AsyncRT_DeviceStream_synchronize(const DeviceStream *stream)
        _checked(
            external_call[
                "AsyncRT_DeviceStream_synchronize",
                _CharPtr,
                _DeviceStreamPtr,
            ](self._handle)
        )


fn _is_nvidia_gpu[target: __mlir_type.`!kgen.target`]() -> Bool:
    return is_triple["nvptx64-nvidia-cuda", target]()


fn _is_path_like(val: String) -> Bool:
    # Ideally we want to use `val.start_with` but we hit a compiler bug if we do
    # that. So, instead we implement the function inline, since we only care
    # about whether the string starts with a `/`, `~`, or "./".
    var ss = val.as_string_slice()
    if len(ss) == 0:
        return False
    if len(ss) >= 1:
        if ss[0] == "/" or ss[0] == "~":
            return True
    if len(ss) >= 2:
        if ss[0] == "." and ss[1] == "/":
            return True
    return False


struct DeviceFunction[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    *,
    target: __mlir_type.`!kgen.target` = _get_gpu_target(),
    _ptxas_info_verbose: Bool = False,
]:
    """Represents a compiled device function. From inside a MAX custom
    operation, use
    [`DeviceContext.compile_function()`](/mojo/stdlib/gpu/host/device_context/DeviceContext#compile_function)
    to compile a function and use
    [`enqueue_function()`](/mojo/stdlib/gpu/host/device_context/DeviceContext#enqueue_function)
    to launch it on the accelerator.
    """

    # emit asm if cross compiling for nvidia gpus.
    alias _emission_kind = "asm" if (
        _cross_compilation() and _is_nvidia_gpu[target]()
    ) else "object"
    var _handle: _DeviceFunctionPtr
    var _func_impl: Info[func_type, func]

    fn __copyinit__(out self, existing: Self):
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceFunction_retain(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_retain",
            NoneType,
            _DeviceFunctionPtr,
        ](existing._handle)
        self._handle = existing._handle
        self._func_impl = existing._func_impl

    fn __moveinit__(out self, owned existing: Self):
        self._handle = existing._handle
        self._func_impl = existing._func_impl

    fn __del__(owned self):
        # Decrement the reference count held by this struct.
        #
        # void AsyncRT_DeviceFunction_release(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_release",
            NoneType,
            _DeviceFunctionPtr,
        ](self._handle)

    @doc_private
    @always_inline
    fn __init__(
        mut self,
        ctx: DeviceContext,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        var max_dynamic_shared_size_bytes: Int32 = -1
        if func_attribute:
            if (
                func_attribute.value().attribute
                == Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            ):
                max_dynamic_shared_size_bytes = func_attribute.value().value
            else:
                raise Error(
                    "the function attribute '",
                    func_attribute.value().attribute,
                    "' is not currently supported",
                )

        # const char *AsyncRT_DeviceContext_loadFunction(
        #     const DeviceFunction **result, const DeviceContext *ctx,
        #     const char *moduleName, const char *functionName, const char *data,
        #     size_t dataLen, int32_t maxDynamicSharedBytes, const char *debugLevel,
        #     int32_t optimizationLevel)
        var result = _DeviceFunctionPtr()
        self._func_impl = _compile_code[
            func,
            emission_kind = self._emission_kind,
            target=target,
        ]()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_loadFunction",
                _CharPtr,
                UnsafePointer[_DeviceFunctionPtr],
                _DeviceContextPtr,
                _CharPtr,
                _CharPtr,
                _CharPtr,
                _SizeT,
                Int32,
                _CharPtr,
                Int32,
            ](
                UnsafePointer.address_of(result),
                ctx._handle,
                self._func_impl.module_name.unsafe_ptr(),
                self._func_impl.function_name.unsafe_ptr(),
                self._func_impl.asm.unsafe_ptr(),
                len(self._func_impl.asm),
                max_dynamic_shared_size_bytes,
                String(DebugLevel).unsafe_cstr_ptr().bitcast[UInt8](),
                Int(OptimizationLevel),
            )
        )
        self._handle = result

    @always_inline
    fn _copy_to_constant_memory(self, mapping: ConstantMemoryMapping) raises:
        # const char *AsyncRT_DeviceFunction_copyToConstantMemory(const DeviceFunction *func, const char *name,
        #                                                         const void *data, size_t byte_size)
        _checked(
            external_call[
                "AsyncRT_DeviceFunction_copyToConstantMemory",
                _CharPtr,
                _DeviceFunctionPtr,
                _CharPtr,
                _VoidPtr,
                _SizeT,
            ](
                self._handle,
                mapping.name.unsafe_cstr_ptr().bitcast[UInt8](),
                mapping.ptr,
                mapping.byte_count,
            )
        )

    @staticmethod
    fn _dump_q[
        name: StringLiteral, val: Variant[Bool, Path, fn () capturing -> Path]
    ]() -> (Bool, Variant[Bool, Path, fn () capturing -> Path]):
        alias name_upper = StringLiteral.get[String(name).upper()]()
        alias env_var = "DUMP_GPU_" + name_upper

        @parameter
        if is_defined[env_var]():
            alias env_val = env_get_string[env_var]()

            @parameter
            if _is_bool_like[env_val]():
                alias env_bool_val = env_get_bool[env_var]()
                return env_bool_val, Variant[
                    Bool, Path, fn () capturing -> Path
                ](env_bool_val)

            @parameter
            if _is_path_like(env_val):
                return True, Variant[Bool, Path, fn () capturing -> Path](
                    Path(env_val)
                )

            constrained[
                False,
                String(
                    "the environment variable '",
                    env_var,
                    (
                        "' is not a valid value. The value should either be"
                        " a boolean value or a path like value, but got '"
                    ),
                    env_val,
                    "'",
                ),
            ]()
            return False, val

        @parameter
        if val.isa[Bool]():
            return val.unsafe_get[Bool](), val

        @parameter
        if val.isa[Path]():
            return val.unsafe_get[Path]() != Path(""), val

        return val.isa[fn () capturing -> Path](), val

    @staticmethod
    fn _cleanup_asm(s: StringLiteral) -> String:
        return (
            String(s)
            .replace("\t// begin inline asm\n", "")
            .replace("\t// end inline asm\n", "")
            .replace("\t;;#ASMSTART\n", "")
            .replace("\t;;#ASMEND\n", "")
        )

    fn _expand_path(self, path: Path) -> Path:
        """If the path contains a `%` character, it is replaced with the module
        name. This allows one to dump multiple kernels which are disambiguated
        by the module name.
        """
        return String(path).replace("%", self._func_impl.module_name)

    @no_inline
    fn dump_rep[
        dump_asm: Variant[Bool, Path, fn () capturing -> Path] = False,
        dump_llvm: Variant[Bool, Path, fn () capturing -> Path] = False,
        _dump_sass: Variant[Bool, Path, fn () capturing -> Path] = False,
    ](self) raises:
        fn get_asm() -> StringLiteral:
            @parameter
            if Self._emission_kind == "asm":
                return self._func_impl.asm
            return _compile_code_asm[
                func,
                emission_kind="asm",
                target=target,
            ]()

        @parameter
        if _ptxas_info_verbose:
            print(_ptxas_compile[target](get_asm(), options="-v"))

        alias dump_asm_tup = Self._dump_q["asm", dump_asm]()
        alias do_dump_asm = dump_asm_tup[0]
        alias dump_asm_val = dump_asm_tup[1]

        @parameter
        if do_dump_asm:
            var asm = self._cleanup_asm(get_asm())

            @parameter
            if dump_asm_val.isa[fn () capturing -> Path]():
                alias dump_asm_fn = dump_asm_val.unsafe_get[
                    fn () capturing -> Path
                ]()
                dump_asm_fn().write_text(asm)
            elif dump_asm_val.isa[Path]():
                self._expand_path(dump_asm_val.unsafe_get[Path]()).write_text(
                    asm
                )
            else:
                print(asm)

        alias dump_sass_tup = Self._dump_q["sass", _dump_sass]()
        alias do_dump_sass = dump_sass_tup[0]
        alias dump_sass_val = dump_sass_tup[1]

        @parameter
        if do_dump_sass:
            var ptx = Self._cleanup_asm(self._func_impl.asm)
            var sass = _to_sass[target](ptx)

            @parameter
            if dump_sass_val.isa[fn () capturing -> Path]():
                alias _dump_sass_fn = dump_sass_val.unsafe_get[
                    fn () capturing -> Path
                ]()
                _dump_sass_fn().write_text(sass)
            elif dump_sass_val.isa[Path]():
                self._expand_path(dump_sass_val.unsafe_get[Path]()).write_text(
                    sass
                )
            else:
                print(sass)

        alias dump_llvm_tup = Self._dump_q["llvm", dump_llvm]()
        alias do_dump_llvm = dump_llvm_tup[0]
        alias dump_llvm_val = dump_llvm_tup[1]

        @parameter
        if do_dump_llvm:
            var llvm = _compile_code_asm[
                Self.func,
                emission_kind="llvm-opt",
                target=target,
            ]()

            @parameter
            if dump_llvm_val.isa[fn () capturing -> Path]():
                alias dump_llvm_fn = dump_llvm_val.unsafe_get[
                    fn () capturing -> Path
                ]()
                dump_llvm_fn().write_text(llvm)
            elif dump_llvm_val.isa[Path]():
                self._expand_path(dump_llvm_val.unsafe_get[Path]()).write_text(
                    llvm
                )
            else:
                print(llvm)

    @always_inline
    @parameter
    fn _call_with_pack[
        *Ts: AnyType
    ](
        self,
        ctx: DeviceContext,
        args: VariadicPack[_, AnyType, *Ts],
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        alias num_args = len(VariadicList(Ts))
        var num_captures = self._func_impl.num_captures
        alias populate = __type_of(self._func_impl).populate
        alias num_captures_static = 16

        var dense_args_addrs = stack_allocation[
            num_captures_static + num_args, UnsafePointer[NoneType]
        ]()

        if num_captures > num_captures_static:
            dense_args_addrs = UnsafePointer[UnsafePointer[NoneType]].alloc(
                num_captures + num_args
            )

        @parameter
        for i in range(num_args):
            var first_word_addr = UnsafePointer.address_of(args[i])
            dense_args_addrs[i] = first_word_addr.bitcast[NoneType]()

        if cluster_dim:
            attributes.append(
                LaunchAttribute.from_cluster_dim(cluster_dim.value())
            )

        if constant_memory:
            for i in range(len(constant_memory)):
                self._copy_to_constant_memory(constant_memory[i])

        # const char *AsyncRT_DeviceContext_enqueueFunctionDirect(const DeviceContext *ctx, const DeviceFunction *func,
        #                                                         uint32_t gridX, uint32_t gridY, uint32_t gridZ,
        #                                                         uint32_t blockX, uint32_t blockY, uint32_t blockZ,
        #                                                         uint32_t sharedMemBytes, void *attrs, uint32_t num_attrs,
        #                                                         void **args)

        if num_captures > 0:
            # Call the populate function to initialize the captured values in the arguments array.
            # The captured values are always at the end of the argument list.
            # This function (generated by the compiler) has to be inlined here
            # and be in the same scope as the user of dense_args_addr
            # (i.e. the following external_call).
            # Because this closure uses stack allocated ptrs
            # to store the captured values in dense_args_addrs, they need to
            # not go out of the scope before dense_args_addr is being use.
            var capture_args_start = dense_args_addrs.offset(num_args)
            populate(
                rebind[UnsafePointer[NoneType]](
                    capture_args_start.bitcast[NoneType]()
                )
            )

            _checked(
                external_call[
                    "AsyncRT_DeviceContext_enqueueFunctionDirect",
                    _CharPtr,
                    _DeviceContextPtr,
                    _DeviceFunctionPtr,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UnsafePointer[LaunchAttribute],
                    UInt32,
                    UnsafePointer[UnsafePointer[NoneType]],
                ](
                    ctx._handle,
                    self._handle,
                    grid_dim.x(),
                    grid_dim.y(),
                    grid_dim.z(),
                    block_dim.x(),
                    block_dim.y(),
                    block_dim.z(),
                    shared_mem_bytes.or_else(0),
                    attributes.unsafe_ptr(),
                    len(attributes),
                    dense_args_addrs,
                )
            )
        else:
            _checked(
                external_call[
                    "AsyncRT_DeviceContext_enqueueFunctionDirect",
                    _CharPtr,
                    _DeviceContextPtr,
                    _DeviceFunctionPtr,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UInt32,
                    UnsafePointer[LaunchAttribute],
                    UInt32,
                    UnsafePointer[UnsafePointer[NoneType]],
                ](
                    ctx._handle,
                    self._handle,
                    grid_dim.x(),
                    grid_dim.y(),
                    grid_dim.z(),
                    block_dim.x(),
                    block_dim.y(),
                    block_dim.z(),
                    shared_mem_bytes.or_else(0),
                    attributes.unsafe_ptr(),
                    len(attributes),
                    dense_args_addrs,
                )
            )

        if num_captures > num_captures_static:
            dense_args_addrs.free()

    @always_inline
    fn get_attribute(self, attr: Attribute) raises -> Int:
        var result: Int32 = 0
        # const char *AsyncRT_DeviceFunction_getAttribute(int32_t *result, const DeviceFunction *func, int32_t attr_code)
        _checked(
            external_call[
                "AsyncRT_DeviceFunction_getAttribute",
                _CharPtr,
                UnsafePointer[Int32],
                _DeviceFunctionPtr,
                Int32,
            ](
                UnsafePointer.address_of(result),
                self._handle,
                attr.code,
            )
        )
        return Int(result)


struct DeviceExternalFunction[
    func_type: AnyTrivialRegType, //, func: func_type
]:
    var _handle: _DeviceFunctionPtr

    fn __copyinit__(out self, existing: Self):
        # Increment the reference count before copying the handle.
        #
        # void AsyncRT_DeviceFunction_retain(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_retain",
            NoneType,
            _DeviceFunctionPtr,
        ](existing._handle)
        self._handle = existing._handle

    fn __moveinit__(out self, owned existing: Self):
        self._handle = existing._handle

    fn __del__(owned self):
        # Decrement the reference count held by this struct.
        #
        # void AsyncRT_DeviceFunction_release(const DeviceFunction *ctx)
        external_call[
            "AsyncRT_DeviceFunction_release",
            NoneType,
            _DeviceFunctionPtr,
        ](self._handle)

    @doc_private
    @always_inline
    fn __init__(
        mut self,
        ctx: DeviceContext,
        *,
        function_name: String,
        asm: String,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        var max_dynamic_shared_size_bytes: Int32 = -1
        if func_attribute:
            if (
                func_attribute.value().attribute
                == Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            ):
                max_dynamic_shared_size_bytes = func_attribute.value().value
            else:
                raise Error(
                    "the function attribute '",
                    func_attribute.value().attribute,
                    "' is not currently supported",
                )

        # const char *AsyncRT_DeviceContext_loadFunction(
        #     const DeviceFunction **result, const DeviceContext *ctx,
        #     const char *moduleName, const char *functionName, const char *data,
        #     size_t dataLen, int32_t maxDynamicSharedBytes, const char *debugLevel,
        #     int32_t optimizationLevel)
        var module_name: String = ""
        var result = _DeviceFunctionPtr()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_loadFunction",
                _CharPtr,
                UnsafePointer[_DeviceFunctionPtr],
                _DeviceContextPtr,
                _CharPtr,
                _CharPtr,
                _CharPtr,
                _SizeT,
                Int32,
                _CharPtr,
                Int32,
            ](
                UnsafePointer.address_of(result),
                ctx._handle,
                module_name.unsafe_ptr(),
                function_name.unsafe_ptr(),
                asm.unsafe_ptr(),
                len(asm),
                max_dynamic_shared_size_bytes,
                String(DebugLevel).unsafe_cstr_ptr().bitcast[UInt8](),
                Int(OptimizationLevel),
            )
        )
        self._handle = result

    @always_inline
    fn _copy_to_constant_memory(self, mapping: ConstantMemoryMapping) raises:
        # const char *AsyncRT_DeviceFunction_copyToConstantMemory(const DeviceFunction *func, const char *name,
        #                                                         const void *data, size_t byte_size)
        _checked(
            external_call[
                "AsyncRT_DeviceFunction_copyToConstantMemory",
                _CharPtr,
                _DeviceFunctionPtr,
                _CharPtr,
                _VoidPtr,
                _SizeT,
            ](
                self._handle,
                mapping.name.unsafe_cstr_ptr().bitcast[UInt8](),
                mapping.ptr,
                mapping.byte_count,
            )
        )

    @always_inline
    @parameter
    fn _call_with_pack[
        *Ts: AnyType
    ](
        self,
        ctx: DeviceContext,
        args: VariadicPack[_, AnyType, *Ts],
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        alias num_args = len(VariadicList(Ts))

        var dense_args_addrs = stack_allocation[
            num_args, UnsafePointer[NoneType]
        ]()

        @parameter
        for i in range(num_args):
            var first_word_addr = UnsafePointer.address_of(args[i])
            dense_args_addrs[i] = first_word_addr.bitcast[NoneType]()

        if cluster_dim:
            attributes.append(
                LaunchAttribute.from_cluster_dim(cluster_dim.value())
            )

        if constant_memory:
            for i in range(len(constant_memory)):
                self._copy_to_constant_memory(constant_memory[i])

        # const char *AsyncRT_DeviceContext_enqueueFunctionDirect(const DeviceContext *ctx, const DeviceFunction *func,
        #                                                         uint32_t gridX, uint32_t gridY, uint32_t gridZ,
        #                                                         uint32_t blockX, uint32_t blockY, uint32_t blockZ,
        #                                                         uint32_t sharedMemBytes, void *attrs, uint32_t num_attrs,
        #                                                         void **args)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_enqueueFunctionDirect",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceFunctionPtr,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UInt32,
                UnsafePointer[LaunchAttribute],
                UInt32,
                UnsafePointer[UnsafePointer[NoneType]],
            ](
                ctx._handle,
                self._handle,
                grid_dim.x(),
                grid_dim.y(),
                grid_dim.z(),
                block_dim.x(),
                block_dim.y(),
                block_dim.z(),
                shared_mem_bytes.or_else(0),
                attributes.unsafe_ptr(),
                len(attributes),
                dense_args_addrs,
            )
        )

    @always_inline
    fn get_attribute(self, attr: Attribute) raises -> Int:
        var result: Int32 = 0
        # const char *AsyncRT_DeviceFunction_getAttribute(int32_t *result, const DeviceFunction *func, int32_t attr_code)
        _checked(
            external_call[
                "AsyncRT_DeviceFunction_getAttribute",
                _CharPtr,
                UnsafePointer[Int32],
                _DeviceFunctionPtr,
                Int32,
            ](
                UnsafePointer.address_of(result),
                self._handle,
                attr.code,
            )
        )
        return Int(result)


@register_passable
struct DeviceContext:
    """Represents a single stream of execution on a particular accelerator
    (GPU). A `DeviceContext` serves as the low-level interface to the
    accelerator inside a MAX [custom operation](/max/custom-ops/) and provides
    methods for allocating buffers on the device, copying data between host and
    device, and for compiling and running functions (also known as kernels) on
    the device.

    The device context can be used as a
    [context manager](/mojo/manual/errors#use-a-context-manager). For example:

    ```mojo
    from gpu.host import DeviceContext
    from gpu import thread_idx

    fn kernel():
        print("hello from thread:", thread_idx.x, thread_idx.y, thread_idx.z)

    with DeviceContext() as ctx:
        ctx.enqueue_function[kernel](grid_dim=1, block_dim=(2, 2, 2))
        ctx.synchronize()
    ```

    A custom operation receives an opaque `DeviceContextPtr`, which provides
    a `get_device_context()` method to retrieve the device context:

    ```mojo
    from runtime.asyncrt import DeviceContextPtr

    @register("custom_op")
    struct CustomOp:
        @staticmethod
        fn execute(ctx_ptr: DeviceContextPtr) raises:
            var ctx = ctx_ptr.get_device_context()
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=(2, 2, 2))
            ctx.synchronize()
    ```
    """

    alias device_info = DEFAULT_GPU
    """`gpu.info.Info` object for the default accelerator."""

    alias device_api = Self.device_info.api
    """Device API for the default accelerator (for example, "cuda" or
    "hip")."""

    var _handle: _DeviceContextPtr

    @always_inline
    fn __init__(
        mut self,
        device_id: Int = 0,
        *,
        api: String = Self.device_api,
        buffer_cache_size: UInt = 0,
    ) raises:
        """Constructs a `DeviceContext` for the specified device.

        Args:
            device_id: ID of the accelerator device. If not specified, uses
                the default accelerator.
            api: Device API, for example, "cuda" for an NVIDIA GPU, or "gpu" for
                the currently available accelerator.
            buffer_cache_size: Amount of space to pre-allocate for device buffers,
                in bytes.
        """
        # const char *AsyncRT_DeviceContext_create(const DeviceContext **result, const char *api, int id)
        var result = _DeviceContextPtr()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_create",
                _CharPtr,
                UnsafePointer[_DeviceContextPtr],
                _CharPtr,
                Int32,
                _SizeT,
            ](
                UnsafePointer.address_of(result),
                api.unsafe_ptr(),
                device_id,
                buffer_cache_size,
            )
        )
        self._handle = result

    fn _retain(self):
        # Increment the reference count.
        #
        # void AsyncRT_DeviceContext_retain(const DeviceContext *ctx)
        external_call[
            "AsyncRT_DeviceContext_retain",
            NoneType,
            _DeviceContextPtr,
        ](self._handle)

    @doc_private
    @implicit
    fn __init__(out self, handle: UnsafePointer[NoneType]):
        """Create a Mojo DeviceContext from a pointer to an existing C++ object.
        """
        self._handle = handle.bitcast[_DeviceContextCpp]()
        self._retain()

    @doc_private
    @implicit
    fn __init__(out self, ctx_ptr: _DeviceContextPtr):
        """Create a Mojo DeviceContext from a pointer to an existing C++ object.
        """
        self._handle = ctx_ptr
        self._retain()

    fn __copyinit__(out self, existing: Self):
        """Copy the `DeviceContext`."""
        # Increment the reference count before copying the handle.
        existing._retain()
        self._handle = existing._handle

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    fn __del__(owned self):
        # Decrement the reference count held by this struct.
        #
        # void AsyncRT_DeviceContext_release(const DeviceContext *ctx)
        external_call[
            "AsyncRT_DeviceContext_release",
            NoneType,
            _DeviceContextPtr,
        ](self._handle)

    fn __enter__(owned self) -> Self:
        return self^

    fn name(self) -> String:
        """Returns the device name, an ASCII string identifying this device,
        defined by the native device API."""
        # const char *AsyncRT_DeviceContext_deviceName(const DeviceContext *ctx)
        var name_ptr = external_call[
            "AsyncRT_DeviceContext_deviceName",
            _CharPtr,
            _DeviceContextPtr,
        ](
            self._handle,
        )
        result = String(StaticString(unsafe_from_utf8_ptr=name_ptr))
        # void AsyncRT_DeviceContext_strfree(const char* ptr)
        external_call["AsyncRT_DeviceContext_strfree", NoneType, _CharPtr](
            name_ptr
        )
        return result

    fn api(self) -> String:
        """Returns the name of the API used to program the device.

        Possible values are:

        - "cpu": Generic host device (CPU).
        - "cuda": NVIDIA GPUs.
        - "hip": AMD GPUs.
        """
        # void AsyncRT_DeviceContext_deviceApi(llvm::StringRef *result, const DeviceContext *ctx)
        var api_ptr = StaticString(ptr=UnsafePointer[Byte](), length=0)
        external_call[
            "AsyncRT_DeviceContext_deviceApi",
            NoneType,
            UnsafePointer[StaticString],
            _DeviceContextPtr,
        ](
            UnsafePointer.address_of(api_ptr),
            self._handle,
        )
        return String(api_ptr)

    @always_inline
    fn malloc_host[
        type: AnyType
    ](self, size: Int) raises -> UnsafePointer[type]:
        """Allocates a block of _pinned_ memory on the host.

        Pinned memory is guaranteed to remain resident in the host's RAM, not be
        paged/swapped out to disk. Memory allocated normally (for example, using
        [`UnsafePointer.alloc()`](/mojo/stdlib/memory/unsafe_ptr/UnsafePointer#alloc))
        is pageable—individual pages of memory can be moved to secondary storage
        (disk/SSD) when main memory fills up.

        Using pinned memory allows devices to make fast transfers
        between host memory and device memory, because they can use direct
        memory access (DMA) to transfer data without relying on the CPU.

        Allocating too much pinned memory can cause performance issues, since it
        reduces the amount of memory available for other processes.

        Parameters:
            type: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            A pointer to the newly-allocated memory.
        """
        # const char *AsyncRT_DeviceContext_mallocHost(void **result, const DeviceContext *ctx, size_t size)
        alias elem_size = sizeof[type]()
        var result = UnsafePointer[type]()
        _checked(
            external_call[
                "AsyncRT_DeviceContext_mallocHost",
                _CharPtr,
                UnsafePointer[UnsafePointer[type]],
                _DeviceContextPtr,
                _SizeT,
            ](
                UnsafePointer.address_of(result),
                self._handle,
                size * elem_size,
            )
        )
        return result

    @always_inline
    fn free_host[type: AnyType](self, ptr: UnsafePointer[type]) raises:
        """Frees a previously-allocated block of pinned memory.

        Parameters:
            type: The data type stored in the allocated memory.

        Args:
            ptr: Pointer to the data block to free.
        """
        # const char * AsyncRT_DeviceContext_freeHost(const DeviceContext *ctx, void *ptr)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_freeHost",
                _CharPtr,
                _DeviceContextPtr,
                UnsafePointer[type],
            ](
                self._handle,
                ptr,
            )
        )

    fn enqueue_create_buffer[
        type: DType
    ](self, size: Int) raises -> DeviceBuffer[type]:
        """Enqueues a buffer creation using the `DeviceBuffer` constructor.

        For GPU devices, the space is allocated in the device's global memory.

        Parameters:
            type: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            The allocated buffer.
        """
        return DeviceBuffer[type](self, size, _DeviceBufferMode._ASYNC)

    fn create_buffer_sync[
        type: DType
    ](self, size: Int) raises -> DeviceBuffer[type]:
        """Creates a buffer synchronously using the `DeviceBuffer` constructor.

        Parameters:
            type: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            The allocated buffer.
        """
        var result = DeviceBuffer[type](self, size, _DeviceBufferMode._SYNC)
        self.synchronize()
        return result

    fn enqueue_create_host_buffer[
        type: DType
    ](self, size: Int) raises -> DeviceBuffer[type]:
        """Enqueues a the creation of a host memory DeviceBuffer."""
        return DeviceBuffer[type](self, size, _DeviceBufferMode._HOST)

    @always_inline
    fn compile_function[
        func_type: AnyTrivialRegType, //,
        func: func_type,
        *,
        dump_asm: Variant[Bool, Path, fn () capturing -> Path] = False,
        dump_llvm: Variant[Bool, Path, fn () capturing -> Path] = False,
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            target = Self.device_info.target(),
            _ptxas_info_verbose=False,
        ],
    ) raises:
        """Compiles the provided function for execution on this device.

        Parameters:
            func_type: Type of the function.
            func: The function to compile.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.

        Args:
            func_attribute: An attribute to use when compiling the code (such
                as maximum shared memory size).

        Returns:
            The compiled function.
        """
        result = self.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=False,
            _target = Self.device_info.target(),
            _ptxas_info_verbose=False,
        ](func_attribute=func_attribute)

    @doc_private
    @always_inline
    fn compile_function[
        func_type: AnyTrivialRegType, //,
        func: func_type,
        *,
        dump_asm: Variant[Bool, Path, fn () capturing -> Path] = False,
        dump_llvm: Variant[Bool, Path, fn () capturing -> Path] = False,
        _dump_sass: Variant[Bool, Path, fn () capturing -> Path] = False,
        _target: __mlir_type.`!kgen.target` = Self.device_info.target(),
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceFunction[
            func,
            target=_target,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ],
    ) raises:
        """Private version of `compile_function()`, which includes debugging
        params."""
        debug_assert(
            not func_attribute
            or func_attribute.value().attribute
            != Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            or func_attribute.value().value
            <= self.device_info.shared_memory_per_multiprocessor,
            "Requested more than available shared memory.",
        )
        alias result_type = __type_of(result)
        result = result_type(
            self,
            func_attribute=func_attribute,
        )

        result.dump_rep[
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
        ]()

    fn load_function[
        func_type: AnyTrivialRegType, //,
        func: func_type,
    ](
        self,
        *,
        function_name: String,
        asm: String,
        func_attribute: OptionalReg[FuncAttribute] = None,
        out result: DeviceExternalFunction[func],
    ) raises:
        alias result_type = __type_of(result)
        result = result_type(
            self,
            function_name=function_name,
            asm=asm,
            func_attribute=func_attribute,
        )

    @parameter
    @always_inline
    fn enqueue_function[
        func_type: AnyTrivialRegType, //,
        func: func_type,
        *Ts: AnyType,
        dump_asm: Variant[Bool, Path, fn () capturing -> Path] = False,
        dump_llvm: Variant[Bool, Path, fn () capturing -> Path] = False,
    ](
        self,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device.

        Parameters:
            func_type: The type of the function to launch.
            func: The function to launch.
            Ts: The types of the arguments being passed to the function.
            dump_asm: Pass `True` or a `Path` to dump the assembly.
            dump_llvm: Pass `True` or a `Path` to dump the LLVM IR.
        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from gpu.host import DeviceContext

        fn kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile it
        first to remove the overhead:

        ```mojo
        with DeviceContext() as ctx:
            var compile_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compile_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```
        """

        var gpu_kernel = self.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=False,
            _target = self.device_info.target(),
            _ptxas_info_verbose=False,
        ](func_attribute=func_attribute)

        self._enqueue_function(
            gpu_kernel,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn enqueue_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        """Enqueues a compiled function for execution on this device.

        Parameters:
            Ts: Argument types.

        Args:
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

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from gpu.host import DeviceContext

        fn kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile
        the function first to remove the overhead:

        ```mojo
        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        """
        self._enqueue_function(
            f,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn _enqueue_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceFunction,
        args: VariadicPack[_, AnyType, *Ts],
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        f._call_with_pack(
            self,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn enqueue_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceExternalFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        self._enqueue_external_function(
            f,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @parameter
    @always_inline
    fn _enqueue_external_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceExternalFunction,
        args: VariadicPack[_, AnyType, *Ts],
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        owned attributes: List[LaunchAttribute] = List[LaunchAttribute](),
        owned constant_memory: List[ConstantMemoryMapping] = List[
            ConstantMemoryMapping
        ](),
    ) raises:
        f._call_with_pack(
            self,
            args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    @always_inline
    fn execution_time[
        func: fn (Self) raises capturing [_] -> None
    ](self, num_iters: Int) raises -> Int:
        var timer_ptr = _DeviceTimerPtr()
        # const char* AsyncRT_DeviceContext_startTimer(const DeviceTimer **result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_startTimer",
                _CharPtr,
                UnsafePointer[_DeviceTimerPtr],
                _DeviceContextPtr,
            ](
                UnsafePointer.address_of(timer_ptr),
                self._handle,
            )
        )
        var timer = _DeviceTimer(timer_ptr)
        for _ in range(num_iters):
            func(self)
        var elapsed_nanos: Int = 0
        # const char *AsyncRT_DeviceContext_stopTimer(int64_t *result, const DeviceContext *ctx, const DeviceTimer *timer)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stopTimer",
                _CharPtr,
                UnsafePointer[Int],
                _DeviceContextPtr,
                _DeviceTimerPtr,
            ](
                UnsafePointer.address_of(elapsed_nanos),
                self._handle,
                timer._handle,
            )
        )
        return elapsed_nanos

    @always_inline
    fn execution_time[
        func: fn () raises capturing [_] -> None
    ](self, num_iters: Int) raises -> Int:
        var timer_ptr = _DeviceTimerPtr()
        # const char* AsyncRT_DeviceContext_startTimer(const DeviceTimer **result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_startTimer",
                _CharPtr,
                UnsafePointer[_DeviceTimerPtr],
                _DeviceContextPtr,
            ](
                UnsafePointer.address_of(timer_ptr),
                self._handle,
            )
        )
        var timer = _DeviceTimer(timer_ptr)
        for _ in range(num_iters):
            func()
        var elapsed_nanos: Int = 0
        # const char *AsyncRT_DeviceContext_stopTimer(int64_t *result, const DeviceContext *ctx, const DeviceTimer *timer)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stopTimer",
                _CharPtr,
                UnsafePointer[Int],
                _DeviceContextPtr,
                _DeviceTimerPtr,
            ](
                UnsafePointer.address_of(elapsed_nanos),
                self._handle,
                timer._handle,
            )
        )
        return elapsed_nanos

    @always_inline
    fn execution_time_iter[
        func: fn (Self, Int) raises capturing [_] -> None
    ](self, num_iters: Int) raises -> Int:
        var timer_ptr = _DeviceTimerPtr()
        # const char* AsyncRT_DeviceContext_startTimer(const DeviceTimer **result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_startTimer",
                _CharPtr,
                UnsafePointer[_DeviceTimerPtr],
                _DeviceContextPtr,
            ](
                UnsafePointer.address_of(timer_ptr),
                self._handle,
            )
        )
        var timer = _DeviceTimer(timer_ptr)
        for i in range(num_iters):
            func(self, i)
        var elapsed_nanos: Int = 0
        # const char *AsyncRT_DeviceContext_stopTimer(int64_t *result, const DeviceContext *ctx, const DeviceTimer *timer)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_stopTimer",
                _CharPtr,
                UnsafePointer[Int],
                _DeviceContextPtr,
                _DeviceTimerPtr,
            ](
                UnsafePointer.address_of(elapsed_nanos),
                self._handle,
                timer._handle,
            )
        )
        return elapsed_nanos

    @always_inline
    fn enqueue_copy[
        type: DType
    ](
        self, dst_buf: DeviceBuffer[type], src_ptr: UnsafePointer[Scalar[type]]
    ) raises:
        """Enqueues an async copy from the host to the provided device
        buffer. The number of bytes copied is determined by the size of the
        device buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_ptr: Host pointer to copy from.
        """
        # const char * AsyncRT_DeviceContext_HtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const void *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_HtoD_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                UnsafePointer[Scalar[type]],
            ](
                self._handle,
                dst_buf._handle,
                src_ptr,
            )
        )

    @always_inline
    fn enqueue_copy[
        type: DType
    ](
        self, dst_ptr: UnsafePointer[Scalar[type]], src_buf: DeviceBuffer[type]
    ) raises:
        """Enqueues an async copy from the device to the host. The
        number of bytes copied is determined by the size of the device buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_buf: Device buffer to copy from.
        """
        # const char * AsyncRT_DeviceContext_DtoH_async(const DeviceContext *ctx, void *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoH_async",
                _CharPtr,
                _DeviceContextPtr,
                UnsafePointer[Scalar[type]],
                _DeviceBufferPtr,
            ](
                self._handle,
                dst_ptr,
                src_buf._handle,
            )
        )

    @always_inline
    fn enqueue_copy[
        type: DType
    ](
        self,
        dst_ptr: UnsafePointer[Scalar[type]],
        src_ptr: UnsafePointer[Scalar[type]],
        size: Int,
    ) raises:
        """Enqueues an async copy of `size` elements from a device pointer to
        another device pointer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_ptr: Device pointer to copy from.
            size: Number of elements (of the specified `DType`) to copy.
        """
        # Not directly implemented on DeviceContext, wrap in buffers first
        var dst_buf = DeviceBuffer[
            mut = dst_ptr.mut,
            origin = dst_ptr.origin,
            address_space = dst_ptr.address_space,
        ](self, dst_ptr, size, owning=False)
        var src_buf = DeviceBuffer[
            mut = src_ptr.mut,
            origin = src_ptr.origin,
            address_space = src_ptr.address_space,
        ](self, src_ptr, size, owning=False)
        self.enqueue_copy[type](dst_buf, src_buf)

    @always_inline
    fn enqueue_copy[
        type: DType
    ](
        self, dst_buf: DeviceBuffer[type, **_], src_buf: DeviceBuffer[type, **_]
    ) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.
        """
        # const char * AsyncRT_DeviceContext_DtoD_async(const DeviceContext *ctx, const DeviceBuffer *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoD_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                _DeviceBufferPtr,
            ](
                self._handle,
                dst_buf._handle,
                src_buf._handle,
            )
        )

    @always_inline
    fn copy[
        type: DType
    ](
        self, dst_buf: DeviceBuffer[type], src_ptr: UnsafePointer[Scalar[type]]
    ) raises:
        """Copies data from the host to the provided device
        buffer. The number of bytes copied is determined by the size of the
        device buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_ptr: Host pointer to copy from.
        """
        # const char * AsyncRT_DeviceContext_HtoD_sync(const DeviceContext *ctx, const DeviceBuffer *dst, const void *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_HtoD_sync",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                UnsafePointer[Scalar[type]],
            ](
                self._handle,
                dst_buf._handle,
                src_ptr,
            )
        )

    @always_inline
    fn copy[
        type: DType
    ](
        self, dst_ptr: UnsafePointer[Scalar[type]], src_buf: DeviceBuffer[type]
    ) raises:
        """Copies data from the device to the host. The
        number of bytes copied is determined by the size of the device buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_buf: Device buffer to copy from.
        """
        # const char * AsyncRT_DeviceContext_DtoH_sync(const DeviceContext *ctx, void *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoH_sync",
                _CharPtr,
                _DeviceContextPtr,
                UnsafePointer[Scalar[type]],
                _DeviceBufferPtr,
            ](
                self._handle,
                dst_ptr,
                src_buf._handle,
            )
        )

    @always_inline
    fn copy[
        type: DType
    ](self, dst_buf: DeviceBuffer[type], src_buf: DeviceBuffer[type]) raises:
        """Copies data from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            type: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.
        """
        # const char * AsyncRT_DeviceContext_DtoD_sync(const DeviceContext *ctx, const DeviceBuffer *dst, const DeviceBuffer *src)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoD_sync",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                _DeviceBufferPtr,
            ](
                self._handle,
                dst_buf._handle,
                src_buf._handle,
            )
        )

    @always_inline
    fn enqueue_memset[
        type: DType
    ](self, dst: DeviceBuffer[type, **_], val: Scalar[type]) raises:
        """Enqueues an async memset operation, setting all of the elements in
        the destination device buffer to the specified value.

        Parameters:
            type: Type of the data stored in the buffer.

        Args:
            dst: Destination buffer.
            val: Value to set all elements of `dst` to.
        """
        alias bitwidth = bitwidthof[type]()
        constrained[
            bitwidth == 8 or bitwidth == 16 or bitwidth == 32 or bitwidth == 64,
            "bitwidth of memset type must be one of [8,16,32,64]",
        ]()
        var value: UInt64

        @parameter
        if bitwidth == 8:
            value = UInt64(Int(bitcast[DType.uint8, 1](val)))
        elif bitwidth == 16:
            value = UInt64(Int(bitcast[DType.uint16, 1](val)))
        elif bitwidth == 32:
            value = UInt64(bitcast[DType.uint32, 1](val))
        else:
            value = bitcast[DType.uint64, 1](val)

        # const char *AsyncRT_DeviceContext_setMemory_async(const DeviceContext *ctx, const DeviceBuffer *dst, uint64_t val, size_t val_size)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_setMemory_async",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                UInt64,
                _SizeT,
            ](
                self._handle,
                dst._handle,
                value,
                sizeof[type](),
            )
        )

    @always_inline
    fn memset_sync[
        type: DType
    ](self, dst: DeviceBuffer[type], val: Scalar[type]) raises:
        """Synchronously sets all of the elements in
        the destination device buffer to the specified value.

        Parameters:
            type: Type of the data stored in the buffer.

        Args:
            dst: The destination buffer.
            val: Value to set all elements of `dst` to.
        """
        alias bitwidth = bitwidthof[type]()
        constrained[
            bitwidth == 8 or bitwidth == 16 or bitwidth == 32 or bitwidth == 64,
            "bitwidth of memset type must be one of [8,16,32, 64]",
        ]()
        var value: UInt64

        @parameter
        if bitwidth == 8:
            value = UInt64(Int(bitcast[DType.uint8, 1](val)))
        elif bitwidth == 16:
            value = UInt64(Int(bitcast[DType.uint16, 1](val)))
        elif bitwidth == 32:
            value = UInt64(bitcast[DType.uint32, 1](val))
        else:
            value = bitcast[DType.uint64, 1](val)

        # const char *AsyncRT_DeviceContext_setMemory_sync(const DeviceContext *ctx, const DeviceBuffer *dst, uint64_t val, size_t val_size)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_setMemory_sync",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceBufferPtr,
                UInt64,
                _SizeT,
            ](
                self._handle,
                dst._handle,
                value,
                sizeof[type](),
            )
        )

    fn memset[
        type: DType
    ](self, dst: DeviceBuffer[type], val: Scalar[type]) raises:
        """Enqueues an async memset operation, setting all of the elements in
        the destination device buffer to the specified value.

        Parameters:
            type: Type of the data stored in the buffer.

        Args:
            dst: Destination buffer.
            val: Value to set all elements of `dst` to.
        """
        self.enqueue_memset[type](dst, val)

    @doc_private
    fn stream(self) raises -> DeviceStream:
        return DeviceStream(self)

    @always_inline
    fn synchronize(self) raises:
        """Blocks until all asynchronous calls on the stream associated with
        this device context have completed.

        This should never be necessary when writing a custom operation."""
        # const char * AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_synchronize",
                _CharPtr,
                _DeviceContextPtr,
            ](
                self._handle,
            )
        )

    fn enqueue_wait_for(self, other: DeviceContext) raises:
        """Enqueue a wait for other to be processed."""
        # const char * AsyncRT_DeviceContext_enqueue_wait_for_context(const DeviceContext *ctx, const DeviceContext *other)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_enqueue_wait_for_context",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceContextPtr,
            ](self._handle, other._handle)
        )

    @always_inline
    fn get_driver_version(self) raises -> Int:
        """Returns the driver version associated with this device."""
        var value: Int32 = 0
        # const char * AsyncRT_DeviceContext_getDriverVersion(int *result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_getDriverVersion",
                _CharPtr,
                _IntPtr,
                _DeviceContextPtr,
            ](
                UnsafePointer.address_of(value),
                self._handle,
            )
        )
        return Int(value)

    @always_inline
    fn get_attribute(self, attr: DeviceAttribute) raises -> Int:
        """Returns the specified attribute for this device.

        Args:
            attr: The device attribute to query.

        Returns:
            The value for `attr` on this device.
        """
        var value: Int32 = 0
        # const char * AsyncRT_DeviceContext_getAttribute(int *result, const DeviceContext *ctx, int attr)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_getAttribute",
                _CharPtr,
                _IntPtr,
                _DeviceContextPtr,
                Int,
            ](
                UnsafePointer.address_of(value),
                self._handle,
                Int(attr._value),
            )
        )
        return Int(value)

    @always_inline
    fn is_compatible(self) raises:
        """Returns True if this device is compatible with MAX."""
        # const char * AsyncRT_DeviceContext_isCompatible(const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_isCompatible",
                _CharPtr,
                _DeviceContextPtr,
            ](
                self._handle,
            )
        )

    @always_inline
    fn id(self) raises -> Int64:
        """Returns the ID associated with this device."""
        # int64_t AsyncRT_DeviceContext_id(const DeviceContext *ctx)
        return external_call[
            "AsyncRT_DeviceContext_id", Int64, _DeviceContextPtr
        ](self._handle)

    @doc_private
    @always_inline
    fn compute_capability(self) raises -> Int:
        var compute_capability: Int32 = 0
        # const char * AsyncRT_DeviceContext_computeCapability(int32_t *result, const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_computeCapability",
                _CharPtr,
                _IntPtr,
                _DeviceContextPtr,
            ](UnsafePointer.address_of(compute_capability), self._handle)
        )
        return Int(compute_capability)

    @always_inline
    fn get_memory_info(self) raises -> (_SizeT, _SizeT):
        """Returns the free and total memory size for this device.

        Returns:
            A tuple of (free memory, total memory) in bytes.
        """
        var free = _SizeT(0)
        var total = _SizeT(0)
        # const char *AsyncRT_DeviceContext_getMemoryInfo(const DeviceContext *ctx, size_t *free, size_t *total)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_getMemoryInfo",
                _CharPtr,
                _DeviceContextPtr,
                UnsafePointer[_SizeT],
                UnsafePointer[_SizeT],
            ](
                self._handle,
                UnsafePointer.address_of(free),
                UnsafePointer.address_of(total),
            )
        )

        return (free, total)

    @always_inline
    fn can_access(self, peer: DeviceContext) raises -> Bool:
        """Returns True if this device can access the identified peer device.

        Args:
            peer: The peer device.
        """
        var result: Bool = False
        # const char *AsyncRT_DeviceContext_canAccess(bool *result, const DeviceContext *ctx, const DeviceContext *peer)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_canAccess",
                _CharPtr,
                UnsafePointer[Bool],
                _DeviceContextPtr,
                _DeviceContextPtr,
            ](
                UnsafePointer.address_of(result),
                self._handle,
                peer._handle,
            )
        )
        return result

    @always_inline
    fn enable_peer_access(self, peer: DeviceContext) raises:
        """Enables access to the peer device.

        Args:
            peer: The peer device.
        """
        # const char *AsyncRT_DeviceContext_enablePeerAccess(const DeviceContext *ctx, const DeviceContext *peer)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_enablePeerAccess",
                _CharPtr,
                _DeviceContextPtr,
                _DeviceContextPtr,
            ](
                self._handle,
                peer._handle,
            )
        )

    @staticmethod
    @always_inline
    fn number_of_devices(*, api: String = Self.device_api) -> Int:
        """Returns the number of devices available that support the specified
        API.

        Args:
            api: Requested device API (for example, "cuda" or "hip").
        """
        # int32_t *AsyncRT_DeviceContext_numberOfDevices(const char* kind)
        return Int(
            external_call[
                "AsyncRT_DeviceContext_numberOfDevices",
                Int32,
                _CharPtr,
            ](
                api.unsafe_ptr(),
            )
        )

    fn map_to_host[
        type: DType
    ](self, buf: DeviceBuffer[type]) raises -> _HostMappedBuffer[type]:
        """Allows for temporary access to the device buffer by the host
        from within a `with` statement.

        ```mojo
        var in_dev = ctx.enqueue_create_buffer[DType.float32](length)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

        # Initialize the input and output with known values.
        with ctx.map_to_host(in_dev) as in_host, ctx.map_to_host(out_dev) as out_host:
            for i in range(length):
                in_host[i] = i
                out_host[i] = 255
        ```

        Values modified inside the `with` statement are updated on the
        device when the `with` statement exits.
        """
        return _HostMappedBuffer[type](self, buf)


struct _HostMappedBuffer[
    type: DType,
    address_space: AddressSpace = AddressSpace.GENERIC,
    mut: Bool = True,
    origin: Origin[mut] = Origin[mut].cast_from[MutableAnyOrigin].result,
]:
    var _ctx: DeviceContext
    alias _DevBuf = DeviceBuffer[
        type, address_space=address_space, mut=mut, origin=origin
    ]
    var _dev_buf: Self._DevBuf
    var _cpu_buf: DeviceBuffer[type, **_]

    fn __init__(
        mut self,
        ctx: DeviceContext,
        buf: Self._DevBuf,
    ) raises:
        var cpu_buf = ctx.enqueue_create_host_buffer[type](len(buf))
        self._ctx = ctx
        self._dev_buf = buf
        self._cpu_buf = cpu_buf

    fn __del__(owned self):
        pass

    fn __enter__(mut self) raises -> DeviceBuffer[type]:
        self._dev_buf.enqueue_copy_to(self._cpu_buf)
        self._ctx.synchronize()
        return self._cpu_buf

    fn __exit__(mut self) raises:
        self._ctx.synchronize()
        self._cpu_buf.enqueue_copy_to(self._dev_buf)
        self._ctx.synchronize()
