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
        lib_path: String,
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
        owned config: LoadOptions,
        owned session: InferenceSession,
    ) raises -> CompiledModel:
        let context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to compile model"

        var compile_config = CompileConfig(self.engine.lib)

        let model_path = config._model_path
        if model_path:
            let path = model_path.value()
            compile_config.set_model_path(path.path._strref_dangerous())
            path.path._strref_keepalive()

        let custom_ops_path = config._custom_ops_path
        if custom_ops_path:
            let path = custom_ops_path.value()
            compile_config.set_replace_ops_path(path.path._strref_dangerous())
            path.path._strref_keepalive()

        let model_source = config._source
        if model_source and model_path:
            raise "give either module source or path"

        if model_source:
            compile_config.set_model_source(model_source.value())

        let status = Status(self.engine.lib)
        let compile_ptr = compile_config.borrow_ptr()
        let compiled_model_ptr = call_dylib_func[CCompiledModel](
            self.engine.lib,
            CompiledModel.CompileModelFnName,
            context,
            compile_ptr,
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()

        let model = CompiledModel(
            compiled_model_ptr, self.engine.lib, session ^
        )
        _ = compile_config ^

        # We could borrow config and don't do this,
        # but internally C APi will take ownership of compile_config ptr
        # and mutates it to null. This will convey the intention that we are
        # mutating something we own. There is no need for caller of this
        # to have the mutated value now. That negates the need for inout.
        _ = config ^

        return model ^

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
        owned config: LoadOptions,
        owned session: InferenceSession,
    ) raises -> Model:
        """
        Compiles and initializes the model.
        """
        let compiled_model = self._compile_model_from_config(
            config ^, session.copy()
        )

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
struct LoadOptions(CollectionElement):
    var _source: Optional[ModelSource]
    var _model_path: Optional[Path]
    var _custom_ops_path: Optional[Path]

    fn __init__(inout self):
        self._source = None
        self._model_path = None
        self._custom_ops_path = None

    fn _set_model_source(inout self, module: Module):
        """Specifies the Max Graph Module to load model from.
           Use either this function or `set_model_path` function
           to specify model source.

        Args:
            module: Max Graph module.
        """
        self._source = ModelSource(module.m.c.ptr, FrameworkFormat.MAXGraph)

    fn _set_model_path(inout self, path: Path):
        """Specifies the loaction in filesystem to load model from.
           Use either this function or `set_model_source` function
           to specify model source.

        Args:
            path: Path of the model on disk.

        """
        self._model_path = path

    fn set_custom_ops_path(inout self, path: Path) raises:
        """Replace Modular kernels in given model with user-defined kernels.

        Args:
            path: Path to mojo custom op package.
        """
        self._custom_ops_path = path


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
        lib_path: String,
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
        self, path: Path, config: Optional[LoadOptions] = None
    ) raises -> Model:
        """Compile and Initialize AI model with Max
           engine with given path and config.

        Args:
            path: Location of model in filesystem.
            config: Configurations need for compiling model.

        Returns:
            Initialized model ready for inference.

        """
        var load_config: LoadOptions
        if config:
            load_config = config.value()
        else:
            load_config = LoadOptions()
        load_config._set_model_path(path)
        return __get_address_as_lvalue(self.ptr.value).load_model(
            load_config ^, self.copy()
        )

    fn load_model(
        self, module: Module, config: Optional[LoadOptions] = None
    ) raises -> Model:
        """Compile and Initialize AI model with Max
           engine with given Max Graph module and config.

        Args:
            module: Max Graph module.
            config: Configurations need for compiling model.

        Returns:
            Initialized model ready for inference.

        """
        var load_config: LoadOptions
        if config:
            load_config = config.value()
        else:
            load_config = LoadOptions()
        load_config._set_model_source(module)
        return __get_address_as_lvalue(self.ptr.value).load_model(
            load_config ^, self.copy()
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
