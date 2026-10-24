//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
// Engine module: uses Buffer from driver (renamed from Tensor)

#include "GraphCompiler/Driver/Compile/CompileDriver.h"
#include "GraphCompiler/MEFModel/MEFModel.h"
#include "GraphCompiler/Support/ModelTensorSpec.h"
#include "Mojo/KGENDialect/KGENCompilationContext.h"
#include "Support/Buffer.h"
#include "Support/Configuration.h"
#include "Support/ML/SizeUtils.h"         // maxAttributeAlignment
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "_core/AsyncRT/Bindings.h"       // IWYU pragma: keep (PyAsyncValue)
#include "_core/AsyncRT/TypeCasters.h" // IWYU pragma: keep (AsyncValueRef caster)
#include "_core/Engine/DebugConfig.h"
#include "_core/Engine/InferenceSession.h"
#include "_core/Graph/Bindings.h"
#include "_core/Profiler/Trace.h"
#include "nanobind/make_iterator.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/filesystem.h"    // IWYU pragma: keep (type casters)
#include "nanobind/stl/optional.h"      // IWYU pragma: keep (type casters)
#include "nanobind/stl/shared_ptr.h"    // IWYU pragma: keep (type casters)
#include "nanobind/stl/string.h"        // IWYU pragma: keep (type casters)
#include "nanobind/stl/unordered_map.h" // IWYU pragma: keep (type casters)
#include "nanobind/stl/variant.h"       // IWYU pragma: keep (type casters)
#include "nanobind/stl/vector.h"        // IWYU pragma: keep (type casters)
#include <cstring>

// This should be included in one compilation unit.
// See section on intrusive reference counting.
// https://nanobind.readthedocs.io/en/latest/ownership_adv.html#the-solution
#include "nanobind/intrusive/counter.inl"

namespace nb = nanobind;

// For named argument shorthand.
// See:
// https://nanobind.readthedocs.io/en/stable/basics.html#keyword-and-default-arguments
using namespace nanobind::literals;
using namespace M;

template <typename T>
static void setMojoDefine(M::Core::InferenceSession &self, const char *key,
                          T value) {
  auto &ctx = self.getRuntimeContext().maxContext;
  auto &options = ctx->emplaceIfMissing<KGEN::CompilationContext>();
  options.mojoDefines[key] = value;
}

void M::Graph::Python::bindEngineModule(nb::module_ m) {
  m.doc() = R"nbdoc(
        Modular framework Python bindings.
    )nbdoc";

  // Also a `ValueError`, which these errors were before, so existing
  // handlers keep catching them.
  nb::tuple executionErrorBases = nb::make_tuple(nb::handle(PyExc_RuntimeError),
                                                 nb::handle(PyExc_ValueError));
  nb::exception<M::Core::ModelExecutionError> executionError(
      m, "ModelExecutionError", executionErrorBases);
  executionError.doc() =
      "Raised when ``Model.execute`` fails, often in one of the model's "
      "kernels. The message names the kernel and, with "
      "``source_tracebacks``, the lines of your model it runs. It is a "
      "``RuntimeError``, and a ``ValueError`` for code written before it.";

  nb::class_<ModelTensorSpec>(m, "TensorSpec", R"nbdoc(
        Defines the properties of a tensor, including its name, shape and data type.

        For usage examples, see :obj:`Model.input_metadata`.
      )nbdoc")
      // Properties
      .def_prop_ro(
          "dtype",
          [](ModelTensorSpec &self) { return self.getSpec().getEltType(); },
          "A tensor data type.")
      .def_prop_ro("name", &ModelTensorSpec::getName, "A tensor name.")
      .def_prop_ro(
          "shape",
          [](ModelTensorSpec &self)
              -> std::optional<std::vector<std::optional<std::size_t>>> {
            const auto &spec = self.getSpec();
            if (!spec.hasRank())
              return std::nullopt;

            std::vector<std::optional<std::size_t>> result;
            result.reserve(spec.getRank());
            for (const auto dim : spec) {
              if (dim < 0)
                result.push_back(std::nullopt);
              else
                result.push_back(dim);
            }
            return result;
          },
          R"nbdoc(
        The shape of the tensor as a list of integers.

        If a dimension size is unknown/dynamic (such as the batch size), its
        value is ``None``.
      )nbdoc")
      // Dunders
      .def("__repr__",
           [](ModelTensorSpec &) -> nb::str {
             return {}; /* Replaced in Python*/
           })
      .def("__str__", [](ModelTensorSpec &) -> nb::str {
        return {}; /* Replaced in Python*/
      });

  nb::class_<M::Model>(m, "ModelMetadata", R"nbdoc(
        Input and output metadata for a compiled model function.
      )nbdoc")
      .def_prop_ro("name",
                   [](const M::Model &self) { return self.getName().str(); })
      .def_prop_ro("input_metadata",
                   [](const M::Model &self) {
                     return M::Core::getModelTensorSpecs(self.getInputNames(),
                                                         self.getInputSpecs());
                   })
      .def_prop_ro("output_metadata", [](const M::Model &self) {
        return M::Core::getModelTensorSpecs(self.getOutputNames(),
                                            self.getOutputSpecs());
      });

  nb::class_<M::CompiledModels>(m, "CompiledModels", R"nbdoc(
        A compiled model artifact containing one or more submodels.
      )nbdoc")
      .def("__len__",
           [](const M::CompiledModels &self) { return self.getNumModels(); })
      .def(
          "__getitem__",
          [](const M::CompiledModels &self, ssize_t index) -> const M::Model * {
            ssize_t numModels = static_cast<ssize_t>(self.getNumModels());
            if (index < 0)
              index += numModels;
            if (index < 0 || index >= numModels)
              throw nb::index_error();
            return self.getModel(static_cast<size_t>(index));
          },
          nb::rv_policy::reference, nb::keep_alive<0, 1>())
      .def("__iter__",
           [](const M::CompiledModels &self) {
             auto models =
                 llvm::map_range(self.getModels(),
                                 [](const std::unique_ptr<const M::Model> &m) {
                                   return m.get();
                                 });
             return nb::make_iterator<nb::rv_policy::reference_internal>(
                 nb::type<M::CompiledModels>(), "CompiledModelsIterator",
                 models.begin(), models.end());
           })
      .def_prop_ro("names",
                   [](const M::CompiledModels &self) {
                     return llvm::map_to_vector(
                         self.getModels(),
                         [](const std::unique_ptr<const M::Model> &m) {
                           return m->getName().str();
                         });
                   })
      .def(
          "export_mef",
          [](M::CompiledModels &self, const char *path) {
            BufferRef buffer = self.getBuffer();
            if (!buffer)
              throw std::invalid_argument(
                  "compiled model buffer is not available");
            // TODO MSDK-1026: Delete the `remove` call below.
            // Remove the file if it exists. Currently, getFile maps the file
            // to be written to. But if the new content is shorter than the
            // original file content, there will be leftover data from the old
            // version. A better fix is to provide a createFile in
            // WriteableBuffer.
            (void)llvm::sys::fs::remove(path);
            auto writeOr =
                WriteableBuffer::getFile(path, buffer->getBufferSize(), 0);
            if (writeOr.isError())
              throw std::invalid_argument(writeOr.getError());
            WriteableBufferRef os = writeOr.takeValue();
            *os << buffer->getBuffer();
          },
          "path"_a,
          R"nbdoc(
        Exports the compiled model as MEF bytes to the given file.

        Args:
            path: The filesystem path to write the MEF to.
      )nbdoc");

  m.def(
      "get_config_value",
      [](const std::string &key) -> std::string {
        auto configOr = M::Config::open();
        if (configOr.isError())
          throw std::runtime_error(configOr.getError());
        std::optional<llvm::StringRef> value = configOr->maybeGetValue(key);
        if (!value)
          throw nb::key_error(key.c_str());
        return value->str();
      },
      "key"_a,
      R"nbdoc(
      Reads a config value.

      Global overrides take priority, then the ``MODULAR_<KEY>``
      environment variable, then ``modular.cfg``.

      Raises:
          KeyError: If the key is not set in any source.
          RuntimeError: If the config fails to open.
      )nbdoc");
  m.def(
      "get_global_value",
      [](const std::string &key) -> std::optional<std::string> {
        return M::Config::getGlobalValueIfSet(key);
      },
      "key"_a,
      R"nbdoc(
      Returns the process-wide override for ``key``, or ``None`` if unset.

      Ignores environment variables and ``modular.cfg``.
      )nbdoc");
  m.def(
      "set_global_value",
      [](const std::string &key, const std::string &value) {
        M::Config::setGlobalValue(key, value);
      },
      "key"_a, "value"_a,
      R"nbdoc(
      Sets a process-wide config override.

      Overrides take priority over environment variables and
      ``modular.cfg`` for every consumer in the process. They are not
      inherited by subprocesses.
      )nbdoc");
  m.def(
      "unset_global_value",
      [](const std::string &key) { M::Config::unsetGlobalValue(key); }, "key"_a,
      R"nbdoc(
      Removes an override set by :func:`set_global_value`.
      )nbdoc");
  m.def(
      "max_cache_dir",
      []() -> std::optional<std::filesystem::path> {
        return M::getMaxCacheDir();
      },
      R"nbdoc(
      Returns the directory the engine caches compiled models (``.mef``) in.

      Resolved by the compiler itself.

      Returns:
          pathlib.Path | None: the cache directory, or None if unresolvable.
    )nbdoc");

  m.def(
       "read",
       [](std::filesystem::path path) {
         nb::gil_scoped_release release;
         auto bufferOr = M::Buffer::getFile(path);
         if (bufferOr.isError())
           throw std::runtime_error(bufferOr.getError());
         auto modelsOr = M::CompiledModels::read(bufferOr.takeValue());
         if (modelsOr.isError())
           throw std::runtime_error(modelsOr.getError());
         return modelsOr.takeValue();
       },
       "path"_a,
       R"nbdoc(
      Reads a compiled-model artifact (``.mef``) from a file path.

      Returns:
          The artifact, ready to be initialized on any session via
          ``InferenceSession._load_all``.

      Raises:
          RuntimeError: If the file is missing or is not a valid MEF
              for this engine build.
      )nbdoc")
      .def(
          "read",
          [](nb::bytes data) {
            // Copy into an attribute-aligned owning buffer while the GIL
            // still pins `data`; MEF sections require maxAttributeAlignment
            // and Python bytes make no alignment promise.
            auto buffer = M::WriteableBuffer::get(
                data.size(), M::maxAttributeAlignment().value());
            std::memcpy(buffer->getBufferStart(), data.c_str(), data.size());
            nb::gil_scoped_release release;
            auto modelsOr = M::CompiledModels::read(std::move(buffer));
            if (modelsOr.isError())
              throw std::runtime_error(modelsOr.getError());
            return modelsOr.takeValue();
          },
          "data"_a,
          R"nbdoc(
      Reads a compiled-model artifact (``.mef``) from bytes.

      Returns:
          The artifact, ready to be initialized on any session via
          ``InferenceSession._load_all``.

      Raises:
          RuntimeError: If the bytes are not a valid MEF for this
              engine build.
      )nbdoc");

  nb::class_<M::Core::ModelHandle>(m, "Model", R"nbdoc(
        A loaded model that you can execute.

        Do not instantiate this class directly. Instead, create it with
        :meth:`InferenceSession.load` or :meth:`InferenceSession.init`.

        A :class:`Model` is callable. Calling it directly (``model(inputs...)``)
        accepts tensors as positional or keyword arguments and dispatches
        to :meth:`execute`. You can also call :meth:`execute` directly, which
        accepts positional arguments only.

        When using keyword arguments, the names must match the model's input
        metadata (see :attr:`input_metadata`). Calling the model raises
        ``TypeError`` if a keyword argument doesn't match a model input, if a
        positional and keyword argument refer to the same parameter, or if the
        number of inputs doesn't match.

        For supported input types and execution errors, see :meth:`execute`.

        For example, load a graph, list its inputs, and run it:

        .. code-block:: python

            import numpy as np

            from max.driver import CPU
            from max.dtype import DType
            from max.engine import InferenceSession
            from max.graph import DeviceRef, Graph, TensorType

            with Graph(
                "add_one",
                input_types=[
                    TensorType(DType.float32, ["batch"], device=DeviceRef.CPU())
                ],
            ) as graph:
                (x,) = graph.inputs
                graph.output(x + 1.0)

            session = InferenceSession(devices=[CPU()])
            model = session.load(graph)
            for tensor in model.input_metadata:
                print(
                    f"name: {tensor.name}, shape: {tensor.shape},"
                    f" dtype: {tensor.dtype}"
                )
            outputs = model.execute(np.ones((2,), np.float32))
      )nbdoc")
      // Properties
      .def_prop_ro("devices", &M::Core::ModelHandle::getDevices,
                   "The device objects used by the model.")
      .def_ro("input_devices", &M::Core::ModelHandle::inputDevicesPy,
              "The devices of the model's input tensors, as a list of "
              ":class:`Device` objects.")
      .def_prop_ro("input_metadata", &M::Core::ModelHandle::getInputMetadata,
                   R"nbdoc(
        Metadata about the model's input tensors, as a list of :class:`TensorSpec` objects.

        For example, you can print the input tensor names, shapes, and dtypes:

        .. code-block:: python

            from max.driver import CPU
            from max.dtype import DType
            from max.engine import InferenceSession
            from max.graph import DeviceRef, Graph, TensorType

            device = CPU()
            device_ref = DeviceRef.from_device(device)
            input_type = TensorType(DType.float32, [2, 3], device=device_ref)
            with Graph("metadata", input_types=[input_type]) as graph:
                graph.output(graph.inputs[0])

            session = InferenceSession(devices=[device])
            model = session.load(graph)
            for tensor in model.input_metadata:
                print(f'name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}')
      )nbdoc")
      .def_ro("output_devices", &M::Core::ModelHandle::outputDevicesPy,
              "The devices of the model's output tensors, as a list of "
              ":class:`Device` objects.")
      .def_prop_ro("output_metadata", &M::Core::ModelHandle::getOutputMetadata,
                   R"nbdoc(
        Metadata about the model's output tensors, as a list of :class:`TensorSpec` objects.

        For example, you can print the output tensor names, shapes, and dtypes:

        .. code-block:: python

            from max.driver import CPU
            from max.dtype import DType
            from max.engine import InferenceSession
            from max.graph import DeviceRef, Graph, TensorType

            device = CPU()
            device_ref = DeviceRef.from_device(device)
            input_type = TensorType(DType.float32, [2, 3], device=device_ref)
            with Graph("metadata", input_types=[input_type]) as graph:
                graph.output(graph.inputs[0])

            session = InferenceSession(devices=[device])
            model = session.load(graph)
            for tensor in model.output_metadata:
                print(f'name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}')
      )nbdoc")
      .def_prop_ro("kernel_summaries",
                   &M::Core::ModelHandle::getKernelSummaries,
                   R"nbdoc(
        Kernel fusion summaries from the compiled model.

        Returns a list of strings, one per ``mgp.generic.execute`` kernel in
        the compiled graph. Each string describes the fused kernel
        composition, such as
        ``"Epilogue(custom__kv_rope, custom__kv_cache_store)"``.
      )nbdoc")
      .def_prop_ro("name", &M::Core::ModelHandle::getName,
                   R"nbdoc(
        The symbol name of this model.

        Mirrors the ``sym_name`` of the model's ``mo.graph`` op, preserved
        through MEF serialization. Used by
        :meth:`InferenceSession.load_all` to key the returned dict by graph
        name.
      )nbdoc")
      .def_prop_ro(
          "signature", [](M::Core::ModelHandle &) { /* Replaced in Python */ },
          nb::sig("def signature(self) -> inspect.Signature"),
          "The input signature for the model.")
      // Methods
      .def(
          "execute", [](M::Core::ModelHandle &) { /* Replaced in Python */ },
          nb::sig("def execute(self, *args: InputType) -> list[Buffer]"),
          R"nbdoc(
        Executes the model with the provided input and returns the outputs.

        For example, compiling a small graph and executing it:

        .. code-block:: python

            from max.driver import CPU
            from max.dtype import DType
            from max.engine import InferenceSession
            from max.graph import DeviceRef, Graph, ops

            device = CPU()
            device_ref = DeviceRef.from_device(device)
            with Graph("add") as graph:
                graph.output(
                    ops.add(
                        ops.constant([1.0, 2.0], DType.float32, device=device_ref),
                        ops.constant([3.0, 4.0], DType.float32, device=device_ref),
                    )
                )

            session = InferenceSession(devices=[device])
            model = session.load(graph)
            result = model.execute()[0]

        .. invisible-code-block: python

            import numpy as np

            np.testing.assert_allclose(result.to_numpy(), [4.0, 6.0])

        Args:
            args: A list of input tensors. The following input types are
              supported:

              * Any tensors implementing the DLPack protocol, such as
                :obj:`np.ndarray` or :obj:`torch.Tensor`.
              * MAX driver buffers, such as :obj:`max.driver.Buffer`.
              * Scalar inputs, such as :obj:`bool`, :obj:`float`, :obj:`int`,
                or :obj:`np.generic`.

              All inputs are copied to the device that the model is resident on
              prior to executing.

        Returns:
            A list of output tensors. The output tensors are resident on the
            execution device.

        Raises:
            RuntimeError: If the given input tensors' shapes don't match what
              the model expects.

            TypeError: If the given input tensors' dtype can't be cast to what
              the model expects.

            ValueError: If positional inputs aren't one of the supported
              types.
      )nbdoc")
      // Dunders
      .def(
          "__call__", [](M::Core::ModelHandle &) { /* Replaced in Python */ },
          nb::sig("def __call__(self, *args: InputType, **kwargs: InputType) "
                  "-> list[Buffer]"),
          "Executes the model. See :class:`Model` for details.")
      .def(
          "__repr__", [](M::Core::ModelHandle &) { /* Replaced in Python */ },
          nb::sig("def __repr__(self) -> str"))
      .def(
          "capture", [](M::Core::ModelHandle &) { /* Replaced in Python */ },
          nb::sig("def capture(self, graph_keys: int | Sequence[int], *inputs: "
                  "Buffer) -> "
                  "list[Buffer]"),
          R"nbdoc(
        Captures execution into a device graph for the caller-provided key.

        Capture is best-effort and model-dependent. It records the current
        execution path; models that perform unsupported operations during
        capture (for example, host-device synchronization) fail to capture.
        Callers should decide which phases are safe to capture (such as
        decode-only phases in serving).
      )nbdoc")
      .def(
          "replay", [](M::Core::ModelHandle &) { /* Replaced in Python */ },
          nb::sig("def replay(self, graph_keys: int | Sequence[int], *inputs: "
                  "Buffer) -> None"),
          "Replays the captured device graph for the provided key.")
      .def(
          "debug_verify_replay",
          [](M::Core::ModelHandle &) { /* Replaced in Python */ },
          nb::sig("def debug_verify_replay(self, graph_keys: int | "
                  "Sequence[int], "
                  "*inputs: "
                  "Buffer) -> None"),
          R"nbdoc(
        Executes eagerly and verifies the launch trace matches the captured graph.

        This method validates that graph capture correctly represents eager
        execution by running the model and comparing kernel launch traces.
        Useful for debugging graph capture issues.

        Args:
            graph_keys: One graph key per participating device stream.
            inputs: The input buffers matching the captured input signature.

        Raises:
            RuntimeError: If no graph was captured or trace verification
                fails.
          )nbdoc")
      .def(
          "release_captured_graph",
          [](M::Core::ModelHandle &) { /* Replaced in Python */ },
          nb::sig("def release_captured_graph(self, graph_keys: int | "
                  "Sequence[int]) -> None"),
          R"nbdoc(
        Releases a previously captured device graph.

        Drops the device-side graph and its working memory once the last
        reference held by the runtime is released. Releasing a key that was
        never captured is a no-op.

        Args:
            graph_keys: The caller-provided graph key (or per-device keys)
                identifying the captured graph to release.
          )nbdoc")
      // Private methods
      .def("_execute_device_tensors",
           &M::Core::ModelHandle::executeDeviceTensors, "tensors"_a)
      .def("_capture", &M::Core::ModelHandle::capture, "graph_keys"_a,
           "inputs"_a, "Capture execution into a device graph.")
      .def("_replay", &M::Core::ModelHandle::replay, "graph_keys"_a, "inputs"_a,
           "Replay the captured device graph.")
      .def("_debug_verify_replay", &M::Core::ModelHandle::debugVerifyReplay,
           "graph_keys"_a, "inputs"_a,
           "Debug verify replay against captured graph.")
      .def("_await_device_graphs", &M::Core::ModelHandle::awaitDeviceGraphs,
           "Await all pending device graph instantiations.")
      .def("_release_captured_graph",
           &M::Core::ModelHandle::releaseCapturedGraphs, "graph_keys"_a,
           "Release captured device graphs for the given keys.")
      .def("_export_mef", &M::Core::ModelHandle::export_mef, "path"_a, R"nbdoc(
        Exports the compiled model as a mef to a file.

        Args:
          path: The filename where the mef is exported to.
      )nbdoc")
      .def("reload", &M::Core::ModelHandle::reload, "weights_registry"_a,
           nb::sig("def reload(self, weights_registry: dict, /) -> None"),
           R"nbdoc(
        Reloads weights into this compiled model in-place.

        Reuses the compiled graph and replaces the weight buffers in-place.

        Args:
            weights_registry: Model weight names mapped to their new values.
      )nbdoc")
      .def("release_weights", &M::Core::ModelHandle::releaseWeights, R"nbdoc(
        Drops the host-side weight references held by this model.

        Releases the weights registry and the owning references, so the host
        weight memory can be freed once the caller drops its own references.
        Safe only when every weight was copied to its execution device during
        model init: reading a host weight after this call is undefined
        behavior. ``reload`` remains usable afterwards.
      )nbdoc");

  nb::class_<M::Core::DebugConfig>(m, "DebugConfig", R"nbdoc(
       Unified debug configuration for MAX inference.

       ``DebugConfig`` is a process-wide singleton accessed through
       :attr:`InferenceSession.debug`. It controls model debugging features
       such as ``NaN`` checks, synchronous GPU execution, stack traces, and IR
       dumping.

       There are two ways to configure debugging options:

       * Set the ``MODULAR_DEBUG`` environment variable to a list of
         kebab-case property names separated by commas. Boolean properties
         can be enabled with just the name; others use ``name=value`` form.
         For example: ``MODULAR_DEBUG=nan-check,assert-level=all``.
       * Set properties directly with the Python API, for example
         ``InferenceSession.debug.<property> = <value>``. Options are
         class-level on :class:`InferenceSession` because they affect
         globally shared infrastructure.

       For the environment variable and config file, the name ``sensible``
       enables a curated default set defined in :attr:`sensible_mode`.
      )nbdoc")
      .def_prop_rw("nan_check", &M::Core::DebugConfig::getNanCheck,
                   &M::Core::DebugConfig::setNanCheck,
                   "A boolean that, when ``True``, triggers MAX to "
                   "insert runtime checks after each compiled op that "
                   "abort if any output contains ``NaN``. Takes effect "
                   "at model build time.")
      .def_prop_rw("uninitialized_read_check",
                   &M::Core::DebugConfig::getUninitializedReadCheck,
                   &M::Core::DebugConfig::setUninitializedReadCheck,
                   "A boolean that, when ``True``, triggers MAX to "
                   "instrument buffer reads to detect reads of "
                   "uninitialized memory. Takes effect at model build "
                   "time.")
      .def_prop_rw("uninitialized_read_mode",
                   &M::Core::DebugConfig::getUninitializedReadMode,
                   &M::Core::DebugConfig::setUninitializedReadMode,
                   "A string that sets what :attr:`uninitialized_read_check` "
                   "does on a match. One of ``''``, ``'abort'``, "
                   "``'report'``. ``''`` (unset) behaves as ``'abort'``, "
                   "which stops at the first match. ``'report'`` prints each "
                   "match and keeps running so a single run enumerates every "
                   "offending load, with no bound on its output. On Apple "
                   "GPUs, ``'report'`` currently aborts at the first match "
                   "without printing. Ignored unless "
                   ":attr:`uninitialized_read_check` is set. Takes effect at "
                   "model build time.")
      .def_prop_rw("device_sync_mode", &M::Core::DebugConfig::getDeviceSyncMode,
                   &M::Core::DebugConfig::setDeviceSyncMode,
                   "A boolean that, when ``True``, triggers MAX to "
                   "force synchronous GPU execution so every device "
                   "operation waits for completion. This surfaces async "
                   "errors at their call site but serializes the "
                   "pipeline. Takes effect at run time.")
      .def_prop_rw("stack_trace_on_error",
                   &M::Core::DebugConfig::getStackTraceOnError,
                   &M::Core::DebugConfig::setStackTraceOnError,
                   "A boolean that, when ``True``, triggers MAX to "
                   "print a C++ stack trace whenever a runtime error "
                   "is raised. Takes effect at run time.")
      .def_prop_rw("stack_trace_on_crash",
                   &M::Core::DebugConfig::getStackTraceOnCrash,
                   &M::Core::DebugConfig::setStackTraceOnCrash,
                   "A boolean that, when ``True``, triggers MAX to "
                   "print a C++ stack trace on fatal signals such as "
                   "``SIGSEGV`` or ``SIGABRT``. Takes effect at run time.")
      .def_prop_rw("source_tracebacks",
                   &M::Core::DebugConfig::getSourceTracebacks,
                   &M::Core::DebugConfig::setSourceTracebacks,
                   "A boolean that, when ``True``, triggers MAX to "
                   "capture Python source locations during graph "
                   "construction so runtime errors can be traced back "
                   "to user code. Takes effect at graph build time "
                   "and is typically set using "
                   "``Graph.debug.source_tracebacks``.")
      .def_prop_rw("source_tracebacks_hide_internal",
                   &M::Core::DebugConfig::getSourceTracebacksHideInternal,
                   &M::Core::DebugConfig::setSourceTracebacksHideInternal,
                   "A boolean that, when ``True`` (the default), cuts the "
                   "tracebacks that :attr:`source_tracebacks` captures at "
                   "the call into the graph API, leaving out MAX's "
                   "implementation of each op, so errors show the code "
                   "that created it. Set it to ``False`` to keep every "
                   "frame. Takes effect at graph build time.")
      .def_prop_rw("pre_jit", &M::Core::DebugConfig::getPreJit,
                   &M::Core::DebugConfig::setPreJit,
                   "A boolean that, when ``True``, stops graph "
                   "compilation once the Mojo for the graph has been "
                   "emitted (see ``ir_output_dir``), skipping the kernel "
                   "JIT. ``InferenceSession.compile`` then raises "
                   ":class:`~max.engine.CompilationStopped`. Takes "
                   "effect at model build time.")
      .def_prop_rw("op_log_level", &M::Core::DebugConfig::getOpLogLevel,
                   &M::Core::DebugConfig::setOpLogLevel,
                   "A string that sets the log level for per-op "
                   "tracing. One of ``''``, ``'NOTSET'``, ``'TRACE'``, "
                   "``'DEBUG'``, ``'INFO'``, ``'WARNING'``, "
                   "``'ERROR'``, ``'CRITICAL'``. Takes effect at "
                   "model build time.")
      .def_prop_rw("assert_level", &M::Core::DebugConfig::getAssertLevel,
                   &M::Core::DebugConfig::setAssertLevel,
                   "A string that sets the Mojo assertion level for "
                   "compiled kernels. One of ``''``, ``'none'``, "
                   "``'warn'``, ``'safe'``, ``'all'``. Higher levels "
                   "enable more runtime checks at a performance cost. "
                   "Takes effect at "
                   "model build time.")
      .def_prop_rw(
          "print_style",
          [](const M::Core::DebugConfig &self) -> ResultOutputStyle {
            const std::string s = self.getPrintStyle();
            if (s == "compact")
              return ResultOutputStyle::kCompact;
            if (s == "full")
              return ResultOutputStyle::kFull;
            if (s == "binary")
              return ResultOutputStyle::kBinary;
            if (s == "binary_max_checkpoint")
              return ResultOutputStyle::kBinaryMaxCheckpoint;
            return ResultOutputStyle::kNone;
          },
          [](M::Core::DebugConfig &self, ResultOutputStyle v) {
            switch (v) {
            case ResultOutputStyle::kCompact:
              self.setPrintStyle("compact");
              return;
            case ResultOutputStyle::kFull:
              self.setPrintStyle("full");
              return;
            case ResultOutputStyle::kBinary:
              self.setPrintStyle("binary");
              return;
            case ResultOutputStyle::kBinaryMaxCheckpoint:
              self.setPrintStyle("binary_max_checkpoint");
              return;
            case ResultOutputStyle::kNone:
              self.setPrintStyle("none");
              return;
            }
          },
          "A :class:`PrintStyle` value that sets the format for tensor "
          "debug printing. Takes effect at run time.")
      .def_prop_rw("ir_output_dir", &M::Core::DebugConfig::getIrOutputDir,
                   &M::Core::DebugConfig::setIrOutputDir,
                   "A string path to the directory into which MAX "
                   "dumps intermediate compiler IR for inspection. "
                   "Empty string disables dumping. Takes effect at "
                   "model build time.")
      .def_prop_rw("sensible_mode", &M::Core::DebugConfig::getSensibleMode,
                   &M::Core::DebugConfig::setSensibleMode,
                   "A boolean that, when ``True``, triggers MAX to enable a "
                   "curated default debugging set, including ``nan_check``, "
                   "``assert_level='all'``, ``device_sync_mode``, "
                   "``stack_trace_on_error``, ``stack_trace_on_crash``, and "
                   "``source_tracebacks``. You can override the defaults using "
                   "individual properties.")
      .def("reset", &M::Core::DebugConfig::reset,
           "Resets all debug options to their defaults.");

  // InferenceSession is intrusively reference counted, allowing a single
  // reference count between C++ and Python. This requires a bit of additional
  // setup, see
  // https://nanobind.readthedocs.io/en/latest/ownership_adv.html#intrusive-reference-counting
  nb::class_<M::Core::InferenceSession>(
      m, "InferenceSession",
      nb::intrusive_ptr<M::Core::InferenceSession>(
          [](M::Core::InferenceSession *d, PyObject *po) noexcept {
            d->set_self_py(po);
          }),
      R"nbdoc(
      Manages compilation and execution of MAX models.

      An inference session holds device configuration and compiles graphs
      into executable :class:`Model` objects. It also manages custom
      extensions and debug options.

      .. code-block:: python

          from max import driver
          from max._core.engine import InferenceSession

          devices = [driver.CPU()]
          session = InferenceSession(devices, custom_extensions=[])

      Args:
          devices: The devices used for compilation and execution.
          custom_extensions: The paths to custom Mojo extension libraries.
          num_threads: The number of execution threads. Defaults to 0,
              which lets the runtime choose automatically.
  )nbdoc")
      .def(nb::init<std::vector<Driver::Python::PyDeviceTy> &&,
                    std::vector<std::filesystem::path> &&, size_t>(),
           "devices"_a, "custom_extensions"_a, "num_threads"_a = 0)
      .def(
          "_load_all",
          [](M::Core::InferenceSession &session,
             AsyncRT::AsyncValueRef<M::CompiledModels> &compiled,
             const nb::dict &weightsDict) {
            // Refcount-share — leaves the caller's Python AsyncValue
            // handle intact so the same compile result can be initialized
            // any number of times.
            return session.loadAll(compiled.copy(), weightsDict);
          },
          // No keep_alive needed: each returned ModelHandle holds an
          // nb::ref<InferenceSession> internally (see ModelHandle's
          // post-load constructor in InferenceSession.h).
          "compiled"_a, "weights_registry"_a)
      .def("compile",
           nb::overload_cast<std::filesystem::path,
                             std::vector<std::filesystem::path> &&>(
               &M::Core::InferenceSession::compile),
           // The CompiledModels references framework context owned by the
           // session; keep the session alive while any AsyncValue is alive.
           nb::keep_alive<0, 1>(), "model_path"_a, "custom_extension_paths"_a,
           R"nbdoc(
      Compiles a model from a file path.

      Args:
          model_path: The path to the compiled model file (for example, a
              ``.mef`` file).
          custom_extension_paths: The paths to custom Mojo extension
              libraries.

      Returns:
          A handle to the compiled artifact, ready to be initialized with
          weights via ``_load_all``.
      )nbdoc")
      .def("compile",
           nb::overload_cast<const nb::capsule &,
                             std::vector<std::filesystem::path> &&,
                             std::string &&, bool>(
               &M::Core::InferenceSession::compile),
           nb::keep_alive<0, 1>(), "model"_a, "custom_extensions"_a,
           "pipeline_name"_a, "tile_based_fusion"_a = false,
           R"nbdoc(
      Compiles a model from an in-memory capsule object.

      Args:
          model: A capsule containing the compiled model object.
          custom_extensions: The paths to custom Mojo extension libraries.
          pipeline_name: The name identifier for the compiled pipeline.
          tile_based_fusion: When ``True``, compiles the graph under the
              tile-based programming model. Defaults to ``False``.

      Returns:
          A handle to the compiled artifact, ready to be initialized with
          weights via ``_load_all``.
      )nbdoc")
      .def(
          "_wrap_compiled",
          [](M::Core::InferenceSession &self, M::CompiledModels &&models) {
            auto result = AsyncValueRef<M::CompiledModels>::allocate(
                self.getRuntimeContext().getCPUDevice());
            result.copy().emplace<M::CompiledModels>(std::move(models));
            return result;
          },
          nb::keep_alive<0, 1>(), "models"_a,
          R"nbdoc(
      Wraps an already-read ``CompiledModels`` in a resolved async handle.

      Consumes ``models``. The handle is allocated on this session's
      runtime and can be passed to :meth:`_load_all`.
      )nbdoc")
      .def(
          "set_debug_print_options",
          [](M::Core::InferenceSession &self, ResultOutputStyle style,
             unsigned int precision, const char *directory) {
            auto &ctx = self.getRuntimeContext().maxContext;
            DebugTensorPrintOptions &options =
                ctx->emplaceIfMissing<DebugTensorPrintOptions>();
            options.style = style;
            options.precision = precision;
            options.binaryDir = directory;
          },
          "style"_a, "precision"_a, "directory"_a,
          R"nbdoc(
      Sets debug output options for tensor printing during execution.

      Args:
          style: The output format style.
          precision: The number of decimal places for floating-point
              values.
          directory: The directory path for binary output files.
      )nbdoc")
      .def("set_mojo_define", &setMojoDefine<bool>, "key"_a, "value"_a,
           R"nbdoc(
      Sets a compile-time Mojo define to a boolean value.

      Args:
          key: The define name.
          value: The boolean value to assign.
      )nbdoc")
      .def("set_mojo_define", &setMojoDefine<int>, "key"_a, "value"_a,
           R"nbdoc(
      Sets a compile-time Mojo define to an integer value.

      Args:
          key: The define name.
          value: The integer value to assign.
      )nbdoc")
      .def("set_mojo_define", &setMojoDefine<const char *>, "key"_a, "value"_a,
           R"nbdoc(
      Sets a compile-time Mojo define to a string value.

      Args:
          key: The define name.
          value: The string value to assign.
      )nbdoc")
      .def_prop_ro("devices", &M::Core::InferenceSession::getDevices,
                   "The list of devices used by this session.")
      .def_prop_ro_static(
          "debug",
          [](nb::handle) -> M::Core::DebugConfig & {
            return M::Core::DebugConfig::instance();
          },
          nb::rv_policy::reference, "The unified debug configuration.");

  nb::enum_<ResultOutputStyle>(m, "PrintStyle", R"nbdoc(
      Controls the output format for debug tensor printing.

      Pass one of these values to :meth:`InferenceSession.set_debug_print_options`
      to configure how tensors are printed during execution.
  )nbdoc")
      .value("COMPACT", ResultOutputStyle::kCompact,
             "Compact human-readable format.")
      .value("FULL", ResultOutputStyle::kFull,
             "Full verbose format with all tensor details.")
      .value("BINARY", ResultOutputStyle::kBinary, "Raw binary format.")
      .value("BINARY_MAX_CHECKPOINT", ResultOutputStyle::kBinaryMaxCheckpoint,
             "Binary checkpoint format compatible with MAX.")
      .value("NONE", ResultOutputStyle::kNone, "Disables debug output.");

  M::Profiler::Python::addTracingToNanobindModuleClassMethods(
      m, "max._core.engine.");

  // https://nanobind.readthedocs.io/en/latest/ownership_adv.html#the-solution
  nb::intrusive_init(
      [](PyObject *o) noexcept {
        nb::gil_scoped_acquire guard;
        Py_INCREF(o);
      },
      [](PyObject *o) noexcept {
        nb::gil_scoped_acquire guard;
        Py_DECREF(o);
      });
}
