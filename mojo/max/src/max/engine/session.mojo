# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Defines the `InferenceSession` type that serves as an entry point to
MAX Engine.
"""

from collections import List
from collections.optional import Optional
from os.atomic import Atomic
from pathlib import Path
from sys.ffi import _get_global_or_null

from max._utils import call_dylib_func
from max.driver import Accelerator, Device, cpu
from max.graph import Graph
from max.tensor import Tensor, TensorSpec
from memory import ArcPointer, UnsafePointer

from ._compilation import (
    CCompiledModel,
    CompileConfig,
    CompiledModel,
    FrameworkFormat,
    ModelSource,
)
from ._context import PrintStyle, RuntimeConfig, RuntimeContext
from ._engine_impl import _EngineImpl, _get_engine_path
from ._model_impl import CModel
from ._status import Status


struct _InferenceSessionImpl(Movable):
    var engine: _EngineImpl
    var context: RuntimeContext
    var device: Device

    fn __init__(out self, lib_path: String, device: Device):
        self.engine = _EngineImpl(lib_path)
        self.device = device
        var config = RuntimeConfig(
            self.engine.lib,
            device,
            max_context=_get_global_or_null["MaxContext"](),
        )
        self.context = RuntimeContext(config^, self.engine.lib)

    fn __moveinit__(out self, owned existing: Self):
        self.engine = existing.engine^
        self.context = existing.context^
        self.device = existing.device^

    fn _compile_model_from_config(
        self,
        owned config: _TorchLoadOptions,
        owned session: InferenceSession,
    ) raises -> CompiledModel:
        var context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to compile model"

        var compile_config = CompileConfig(self.engine.lib)

        var model_path = config._model_path
        if model_path:
            var path = model_path.value()
            compile_config.set_model_path(path.path)

        var pipeline_name = config._pipeline_name
        if pipeline_name:
            compile_config.set_pipeline_name(pipeline_name.value())

        var custom_ops_paths = config._custom_ops_paths
        # TODO: Use a direct for loop (#38478).
        for i in range(len(custom_ops_paths)):
            var path = custom_ops_paths[i]
            compile_config.set_replace_ops_path(path.path)

        var model_source = config._source
        if model_source and model_path:
            raise "give either module source or path"

        if model_source:
            compile_config.set_model_source(model_source.value())

        var spec_count = len(config._input_specs)
        for i in range(spec_count):
            var _spec = config._input_specs[i]
            if _spec._static:
                compile_config.add_input_spec(_spec._static.value())
            else:
                var dtype = _spec._dtype
                compile_config.add_input_spec(_spec._dynamic, dtype)

        compile_config.set_torch_input_specs()

        var status = Status(self.engine.lib)
        var compile_ptr = compile_config.borrow_ptr()
        var compiled_model_ptr = call_dylib_func[CCompiledModel](
            self.engine.lib,
            CompiledModel.CompileModelFnName,
            context,
            compile_ptr,
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()

        var model = CompiledModel(compiled_model_ptr, self.engine.lib, session^)
        _ = compile_config^

        # We could borrow config and don't do this,
        # but internally C APi will take ownership of compile_config ptr
        # and mutates it to null. This will convey the intention that we are
        # mutating something we own. There is no need for caller of this
        # to have the mutated value now. That negates the need for mut.
        _ = config^

        return model^

    fn _init_model(
        self,
        owned compiled_model: CompiledModel,
        owned session: InferenceSession,
    ) raises -> Model:
        var status = Status(self.engine.lib)
        var model_ptr = call_dylib_func[CModel](
            self.engine.lib,
            Model._InitModelFnName,
            self.context.borrow_ptr(),
            compiled_model.borrow_ptr(),
            UnsafePointer[NoneType](),  # Pass null weights registry.
            status.borrow_ptr(),
        )
        if status:
            raise status.__str__()

        model_ptr.await_model(self.engine.lib)

        var model = Model(
            self.context.borrow_ptr(),
            model_ptr,
            self.engine.lib,
            session^,
            compiled_model^,
            self.device,
        )
        return model^

    fn load(
        self,
        owned config: _TorchLoadOptions,
        owned session: InferenceSession,
    ) raises -> Model:
        """
        Compiles and initializes the model.
        """
        var compiled_model = self._compile_model_from_config(config^, session)

        return self._init_model(compiled_model^, session^)

    fn get_as_engine_tensor_spec(
        self,
        name: String,
        spec: TensorSpec,
        owned session: InferenceSession,
    ) raises -> EngineTensorSpec:
        var context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create tensor spec"
        return EngineTensorSpec(name, spec, self.engine.lib, session^)

    fn get_as_engine_tensor_spec(
        self,
        name: String,
        shape: Optional[List[Optional[Int64]]],
        dtype: DType,
        owned session: InferenceSession,
    ) raises -> EngineTensorSpec:
        var context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create tensor spec"
        return EngineTensorSpec(name, shape, dtype, self.engine.lib, session^)

    fn new_tensor_map(
        self, owned session: InferenceSession
    ) raises -> TensorMap:
        var context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create tensor map"
        return TensorMap(self.context.borrow_ptr(), self.engine.lib, session^)

    fn new_borrowed_tensor_value[
        type: DType
    ](
        self, owned session: InferenceSession, tensor: Tensor[type]
    ) raises -> Value:
        """Create a new Value representing data read-only from given tensor."""
        var context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create tensor value"
        return Value._new_borrowed_tensor[type](
            self.context.borrow_ptr(), self.engine.lib, session^, tensor
        )

    fn new_bool_value(
        self, owned session: InferenceSession, value: Bool
    ) raises -> Value:
        var context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create bool value"
        return Value._new_bool(
            self.context.borrow_ptr(), self.engine.lib, session^, value
        )

    fn new_list_value(self, owned session: InferenceSession) raises -> Value:
        var context = self.context.borrow_ptr()
        if not context.ptr:
            raise "failed to create list value"
        return Value._new_list(
            self.context.borrow_ptr(), self.engine.lib, session^
        )


@value
struct InputSpec(CollectionElement):
    """
    Specifies a model's input shape and data type (required for TorchScript).

    When loading a TorchScript model, you must specify the shape and data type
    for each input with an `InputSpec`, which you then pass to
    [`InferenceSession.load()`](/max/api/mojo/engine/session/InferenceSession#load).
    For example:

    ```mojo
    var batch = 1
    var seqlen = 128
    var input_ids_spec = TensorSpec(DType.int64, batch, seqlen)
    var attention_mask_spec = TensorSpec(DType.int64, batch, seqlen)

    var session = engine.InferenceSession()
    var model = session.load(
        "roberta.torchscript",
        input_specs=List[InputSpec](
            InputSpec(input_ids_spec), InputSpec(attention_mask_spec)
        ),
    )
    ```
    """

    var _static: Optional[TensorSpec]

    alias _legacy_dynamic_type = Optional[List[Optional[Int64]]]
    alias _dynamic_type = Optional[List[ShapeElement]]
    var _dynamic: Self._dynamic_type
    var _dtype: DType

    @implicit
    fn __init__(out self, spec: TensorSpec):
        """
        Create input specifications for one input tensor, as a
        [`TensorSpec`](/mojo/stdlib/tensor/tensor_spec/TensorSpec).
        Only applicable for TorchScript models.

        Args:
            spec: Spec for the input. This is the standard library
                  [`TensorSpec`](/mojo/stdlib/tensor/tensor_spec/TensorSpec).
        """
        self._static = spec
        self._dynamic = None
        self._dtype = spec.dtype()

    fn __init__(out self, spec: Optional[List[Optional[Int64]]], dtype: DType):
        """
        Create specifications for one input tensor, as a list of integers.
        Only applicable for TorchScript models.

        If an input supports dynamic shapes, use `None` for that dimension
        size.

        Args:
            spec: Shape of the input, as a list of integers.
            dtype: Datatype of the input, from the standard library
                   [`DType`](/mojo/stdlib/builtin/dtype/DType).
        """
        self._static = None
        if spec:
            var dyn_spec = List[ShapeElement]()
            for item in spec.value():
                if item[]:
                    dyn_spec.append(item[].value())
                else:
                    dyn_spec.append(None)
            self._dynamic = dyn_spec^
        else:
            self._dynamic = None
        self._dtype = dtype

    fn __init__(out self, spec: Optional[List[ShapeElement]], dtype: DType):
        """
        Create specifications for one input tensor, as a list of shape
        elements.  Only applicable for TorchScript models.

        If an input supports dynamic shapes, use `None` or a string dimension
        name for that dimension size.

        Args:
            spec: Shape of the input, as a list of
                  [`ShapeElement`](/max/api/mojo/engine/shape_element/ShapeElement)
                  values.
            dtype: Datatype of the input, from the standard library
                   [`DType`](/mojo/stdlib/builtin/dtype/DType).
        """
        self._static = None
        self._dynamic = spec
        self._dtype = dtype

    fn __init__(out self, spec: NoneType, dtype: DType):
        """
        Create a specification for a dynamic-rank input.  Only applicable for
        TorchScript models.

        Args:
            spec: Always `None`.
            dtype: Datatype of the input, from the standard library
                   [`DType`](/mojo/stdlib/builtin/dtype/DType).
        """
        self._static = None
        self._dynamic = None
        self._dtype = dtype


@value
struct _TorchLoadOptions(CollectionElement):
    """
    Configuration options to load PyTorch models with MAX Engine.

    This is used only internally.  To provide load options, pass them as
    keyword arguments to
    [`InferenceSession.load()`](/max/api/mojo/engine/session/InferenceSession#load).
    """

    var _source: Optional[ModelSource]
    var _model_path: Optional[Path]
    var _pipeline_name: Optional[String]
    var _custom_ops_paths: List[Path]
    var _input_specs: List[InputSpec]

    fn __init__(out self):
        """Creates a new _TorchLoadOptions object."""
        self._source = None
        self._model_path = None
        self._pipeline_name = None
        self._custom_ops_paths = List[Path]()
        self._input_specs = List[InputSpec]()

    fn set_model_source(mut self, graph: Graph) raises:
        """Specifies the MAX Graph to load model from.
           Use either this function or `set_model_path` function
           to specify model source.

        Args:
            graph: MAX Graph.
        """
        self._source = ModelSource(
            UnsafePointer(graph._module().c.ptr),
            FrameworkFormat.MAXGraph,
        )

    fn set_model_path(mut self, path: Path):
        """Specifies the loaction in filesystem to load model from.
           Use either this function or `set_model_source` function
           to specify model source.

        Args:
            path: Path of the model on disk.

        """
        self._model_path = path

    fn set_pipeline_name(mut self, graph: Graph) raises:
        """Specifies the given MAX Graph name i.e. llama3.
        Used for telemetry.

        Args:
            graph: MAX Graph.
        """
        self._pipeline_name = graph._name()

    fn set_custom_ops_paths(mut self, paths: List[Path]) raises:
        """Replace Modular kernels in given model with user-defined kernels.

        Args:
            paths: List of paths to mojo custom op packages.
        """
        self._custom_ops_paths = paths

    fn set_input_specs(mut self, specs: List[InputSpec]):
        """Set input specs to the given list of specs.

        Args:
            specs: The list of specs to replace the current list of input specs.
        """
        self._input_specs = specs


@value
struct SessionOptions:
    """
    Configuration options for InferenceSession.
    """

    var _device: Optional[Device]

    fn __init__(out self):
        """Creates a new SessionOptions object."""
        self._device = None

    @implicit
    fn __init__(out self, device: Device):
        """Creates a new SessionOptions object with a device set."""
        self._device = device

    fn _set_device(mut self, device: Device):
        self._device = device


@value
@deprecated(
    "the Mojo max.engine API has been deprecated in favor of the Python API. It"
    " will be open sourced in a future patch prior to being removed."
)
struct InferenceSession:
    """
    Holds the context for MAX Engine in which you can load and run models.

    For example, you can load a model like this:

    ```mojo
    var session = engine.InferenceSession()
    var model = session.load("bert-base-uncased")
    ```
    """

    var _ptr: ArcPointer[_InferenceSessionImpl]

    fn __init__(out self, options: SessionOptions = SessionOptions()) raises:
        """Creates a new inference session.

        Args:
            options: Session options to configure how session is created.

        """
        var device = options._device.or_else(cpu())
        if "cuda" in String(device):
            # This should eventually be a method on the device itself so we can
            # avoid having `session.mojo` depend on CUDA.
            if not Accelerator.check_compute_capability(device):
                raise "Accelerator is not supported by MAX"
        var path = _get_engine_path()
        self._ptr = ArcPointer(_InferenceSessionImpl(path, device))

    fn load(
        self,
        path: Path,
        *,
        custom_ops_paths: List[Path] = List[Path](),
        input_specs: Optional[List[InputSpec]] = None,
    ) raises -> Model:
        """Compile and initialize a model in MAX Engine, with the given
           model path and config.

        Note: PyTorch models must be in TorchScript format.

        If you're loading a TorchScript model, you must specify the `input_specs`
        argument with a list of
        [`InputSpec`](/max/api/mojo/engine/session/InputSpec) objects
        that specify the model's input specs (which may have dynamic shapes).
        For details, see how to [specify input
        specs](/max/model-formats#specify-torchscript-input-specs).

        Args:
            path: Location of model in filesystem. You may pass a string here
                  because the [`Path`](/mojo/stdlib/pathlib/path/Path) object
                  supports implicit casting from a string.
            custom_ops_paths:
                List of paths to Mojo custom op packages, to replace Modular kernels in
                models with user-defined kernels.
            input_specs:
                Provide shapes and dtypes for model inputs.  Required for
                TorchScript models, optional for other input formats.

        Returns:
            Initialized model ready for inference.

        """
        var load_config = _TorchLoadOptions()
        load_config.set_model_path(path)
        load_config.set_custom_ops_paths(custom_ops_paths)
        if input_specs:
            load_config.set_input_specs(input_specs.value())
        return self._ptr[].load(load_config^, self)

    fn load(
        self,
        graph: Graph,
        *,
        custom_ops_paths: List[Path] = List[Path](),
        input_specs: Optional[List[InputSpec]] = None,
    ) raises -> Model:
        """Compile and initialize a model in MAX Engine, with the given
           [`Graph`](/max/api/mojo/graph/graph/Graph) and config.

        Args:
            graph: MAX Graph.
            custom_ops_paths:
                List of paths to Mojo custom op packages, to replace Modular kernels in
                models with user-defined kernels.
            input_specs:
                Provide shapes and dtypes for model inputs.  Required for
                TorchScript models, optional for other input formats.

        Returns:
            Initialized model ready for inference.

        """
        var load_config = _TorchLoadOptions()
        load_config.set_model_source(graph)
        load_config.set_custom_ops_paths(custom_ops_paths)
        load_config.set_pipeline_name(graph)
        if input_specs:
            load_config.set_input_specs(input_specs.value())
        return self._ptr[].load(load_config^, self)

    fn get_as_engine_tensor_spec(
        self, name: String, spec: TensorSpec
    ) raises -> EngineTensorSpec:
        """Gets a TensorSpec compatible with MAX Engine.

        Args:
            name: Name of the Tensor.
            spec: Tensor specification in Mojo TensorSpec format.

        Returns:
           EngineTensorSpec to be used with MAX Engine APIs.

        """
        return self._ptr[].get_as_engine_tensor_spec(name, spec, self)

    fn get_as_engine_tensor_spec(
        self,
        name: String,
        shape: Optional[List[Optional[Int64]]],
        dtype: DType,
    ) raises -> EngineTensorSpec:
        """Gets a TensorSpec compatible with MAX Engine.

        Args:
            name: Name of the Tensor.
            shape: Shape of the Tensor.
                   Dynamic Dimensions can be represented with None and for
                   Dynamic Rank Tensor use None as value for shape.
            dtype: DataType of the Tensor.

        Returns:
            EngineTensorSpec to be used with MAX Engine APIs.
        """
        return self._ptr[].get_as_engine_tensor_spec(name, shape, dtype, self)

    fn new_tensor_map(self) raises -> TensorMap:
        """Gets a new TensorMap. This can be used to pass inputs to model.

        Returns:
            A new instance of TensorMap.
        """
        return self._ptr[].new_tensor_map(self)

    fn new_borrowed_tensor_value[
        type: DType
    ](self, tensor: Tensor[type]) raises -> Value:
        """Create a new Value representing data read-only from given tensor.

        The user must ensure the tensor stays live through the lifetime of the
        value.

        Parameters:
            type: Data type of the tensor to turn into a Value.

        Args:
            tensor: Tensor to borrow into a value.

        Returns:
            A value borrowing the tensor.
        """
        return self._ptr[].new_borrowed_tensor_value(self, tensor)

    fn new_bool_value(self, value: Bool) raises -> Value:
        """Create a new Value representing a Bool.

        Args:
            value: Boolean to wrap into a value.

        Returns:
            Value representing the given boolean.
        """
        return self._ptr[].new_bool_value(self, value)

    fn new_list_value(self) raises -> Value:
        """Create a new Value representing an empty list.

        Returns:
            A new value containing an empty list.
        """
        return self._ptr[].new_list_value(self)

    fn set_debug_print_options(
        mut self,
        style: PrintStyle = PrintStyle.COMPACT,
        precision: UInt = 6,
        output_directory: String = "",
    ):
        """Sets the debug print options on the context.

        This affects debug printing across all model execution using the same InferenceSession.

        Warning: Even with style set to `NONE`, debug print ops in the graph can stop optimizations.
        If you see performance issues, try fully removing debug print ops.

        Args:
            style: How the values will be printed.
            precision: If the style is `FULL`, the digits of precision in the output.
            output_directory: If the style is `BINARY`, the directory to store output tensors.
        """
        self._ptr[].context.set_debug_print_options(
            style, precision, output_directory
        )
