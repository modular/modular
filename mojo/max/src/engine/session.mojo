# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._context import *
from ._status import *
from ._compilation import *
from .model import *
from .tensor_map import *
from ._engine_impl import _EngineImpl, _get_engine_path

from max.graph import Module

from collections.optional import Optional
from memory.anypointer import AnyPointer
from os.atomic import Atomic
from sys.ffi import DLHandle


struct _InferenceSessionImpl(Movable):
    var engine: _EngineImpl
    var context: RuntimeContext
    var ref_count: Atomic[DType.int64]

    fn __init__(
        inout self,
        lib_path: StringRef,
        device: StringRef = "cpu",
        num_threads: Optional[Int] = None,
    ):
        self.engine = _EngineImpl(lib_path)
        let config = RuntimeConfig(self.engine.lib, device, num_threads)
        self.context = RuntimeContext(config ^, self.engine.lib)
        self.ref_count = 1

    fn __moveinit__(inout self, owned existing: Self):
        self.engine = existing.engine ^
        self.context = existing.context ^
        self.ref_count = existing.ref_count.fetch_add(0)

    fn _compile_model_from_config(
        self,
        config: CompileConfig,
        owned session: InferenceSession,
    ) raises -> CompiledModel:
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to compile model"

        let status = Status(self.engine.lib)
        let compiled_model_ptr = call_dylib_func[CCompiledModel](
            self.engine.lib,
            CompiledModel.CompileModelFnName,
            context,
            config.borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()

        return CompiledModel(compiled_model_ptr, self.engine.lib, session ^)

    fn compile_model(
        self,
        model_path: StringRef,
        user_defined_ops_path: Optional[StringRef],
        owned session: InferenceSession,
    ) raises -> CompiledModel:
        """
        Compiles the model given model path.

        Raises: an Error if the path to the model file does
        not exist on the filesystem.
        """
        let config = CompileConfig(self.engine.lib)
        if user_defined_ops_path:
            config.set_replace_ops_path(user_defined_ops_path.value())
        config.set_model_path(model_path)

        return self._compile_model_from_config(config, session ^)

    fn compile_model(
        self,
        module: Module,
        user_defined_ops_path: Optional[StringRef],
        owned session: InferenceSession,
    ) raises -> CompiledModel:
        """
        Compiles the model given an opaque model source representation.
        """
        # Set compilation config options.
        let config = CompileConfig(self.engine.lib)
        if user_defined_ops_path:
            config.set_replace_ops_path(user_defined_ops_path.value())
        config.set_model_source(
            ModelSource(module.m.c.ptr, FrameworkFormat.MAXGraph)
        )

        return self._compile_model_from_config(config, session ^)

    fn _init_model(
        self,
        owned compiled_model: CompiledModel,
        owned session: InferenceSession,
    ) raises -> Model:
        let status = Status(self.engine.lib)
        let model_ptr = call_dylib_func[CModel](
            self.engine.lib,
            Model._InitModelFnName,
            self.context.borrow_ptr(),
            compiled_model.borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()

        model_ptr.await_model(self.engine.lib)

        let model = Model(
            self.context.borrow_ptr(),
            model_ptr,
            self.engine.lib,
            session ^,
            compiled_model ^,
        )
        return model ^

    fn load_model(
        self,
        module: Module,
        user_defined_ops_path: Optional[StringRef],
        owned session: InferenceSession,
    ) raises -> Model:
        """
        Compiles and initializes the model given a MAX graph `Module`.
        """
        let compiled_model = self.compile_model(
            module, user_defined_ops_path, session.copy()
        )

        return self._init_model(compiled_model ^, session ^)

    fn load_model(
        self,
        model_path: Path,
        user_defined_ops_path: Optional[StringRef],
        owned session: InferenceSession,
    ) raises -> Model:
        """
        Compiles and initializes model given path.
        """
        let compiled_model = self.compile_model(
            model_path.path._strref_dangerous(),
            user_defined_ops_path,
            session.copy(),
        )
        model_path.path._strref_keepalive()

        return self._init_model(compiled_model ^, session ^)

    fn get_as_engine_tensor_spec(
        self,
        name: StringRef,
        spec: TensorSpec,
        owned session: InferenceSession,
    ) raises -> EngineTensorSpec:
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create tensor spec"
        return EngineTensorSpec(name, spec, self.engine.lib, session ^)

    fn get_as_engine_tensor_spec(
        self,
        name: StringRef,
        shape: Optional[DynamicVector[Optional[Int64]]],
        dtype: DType,
        owned session: InferenceSession,
    ) raises -> EngineTensorSpec:
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create tensor spec"
        return EngineTensorSpec(name, shape, dtype, self.engine.lib, session ^)

    fn new_tensor_map(
        self, owned session: InferenceSession
    ) raises -> TensorMap:
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create tensor map"
        return TensorMap(self.context.borrow_ptr(), self.engine.lib, session ^)


@value
@register_passable
struct InferenceSession:
    var ptr: AnyPointer[_InferenceSessionImpl]

    fn __init__(
        device: StringRef = "cpu", num_threads: Optional[Int] = None
    ) raises -> Self:
        let path = _get_engine_path()
        let self = Self._allocateAndInit(
            path._strref_dangerous(), device, num_threads
        )
        path._strref_keepalive()
        return Self {ptr: self}

    @staticmethod
    fn _allocateAndInit(
        lib_path: StringRef,
        device: StringRef,
        num_threads: Optional[Int],
    ) raises -> AnyPointer[_InferenceSessionImpl]:
        let ptr = AnyPointer[_InferenceSessionImpl].alloc(1)
        __get_address_as_uninit_lvalue(ptr.value).__init__(
            lib_path, device, num_threads
        )
        return ptr

    fn copy(self) -> Self:
        _ = __get_address_as_lvalue(self.ptr.value).ref_count.fetch_add(1)
        return Self {ptr: self.ptr}

    fn load_model(
        self,
        model_path: Path,
        user_defined_ops_path: Optional[StringRef] = None,
    ) raises -> Model:
        """
        Replaces Modular ops with user-defined ops using the provided
        path-to-mojopkg and then compiles and initializes the model
        using the provided model path.
        """
        return __get_address_as_lvalue(self.ptr.value).load_model(
            model_path, user_defined_ops_path, self.copy()
        )

    fn load_model(
        self,
        module: Module,
        user_defined_ops_path: Optional[StringRef] = None,
    ) raises -> Model:
        """
        Replaces Modular ops with user-defined ops using the provided
        path-to-mojopkg and then compiles and initializes the model
        using the provided MAX graph module.
        """
        return __get_address_as_lvalue(self.ptr.value).load_model(
            module, user_defined_ops_path, self.copy()
        )

    fn get_as_engine_tensor_spec(
        self, name: StringRef, spec: TensorSpec
    ) raises -> EngineTensorSpec:
        """Gets a TensorSpec compatible with Max Engine.

        Args:
            name: Name of the Tensor.
            spec: Tensor specification in Mojo TensorSpec format.

        Returns:
           EngineTensorSpec to be used with Max Engine APIs.

        """
        return __get_address_as_lvalue(
            self.ptr.value
        ).get_as_engine_tensor_spec(name, spec, self.copy())

    fn get_as_engine_tensor_spec(
        self,
        name: StringRef,
        shape: Optional[DynamicVector[Optional[Int64]]],
        dtype: DType,
    ) raises -> EngineTensorSpec:
        """Gets a TensorSpec compatible with Max Engine.

        Args:
            name: Name of the Tensor.
            shape: Shape of the Tensor.
                   Dynamic Dimensions can be represented with None and for
                   Dynamic Rank Tensor use None as value for shape.
            dtype: DataType of the Tensor.

        Returns:
            EngineTensorSpec to be used with Max Engine APIs.
        """
        return __get_address_as_lvalue(
            self.ptr.value
        ).get_as_engine_tensor_spec(name, shape, dtype, self.copy())

    fn new_tensor_map(self) raises -> TensorMap:
        return __get_address_as_lvalue(self.ptr.value).new_tensor_map(
            self.copy()
        )

    fn __del__(owned self):
        if __get_address_as_lvalue(self.ptr.value).ref_count.fetch_sub(1) != 1:
            # There are others holding reference to this session. Keep the
            # session alive and let other reference holders deal with
            # managing it.
            return

        # Now we know that there is only active reference to this session
        # and that is held by us. So go ahead and call destructor on it.
        _ = __get_address_as_owned_value(self.ptr.value)
        self.ptr.free()
