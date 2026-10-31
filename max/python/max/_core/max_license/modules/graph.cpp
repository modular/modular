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

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "GraphCompiler/GraphCompiler.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/RMODialect/RMOAttrs.h"
#include "Init/Init.h"
#include "Mojo/KGENDialect/KGENOps.h" // IWYU pragma: keep (provides definition of M::KGEN::GeneratorOp)
#include "Support/Context.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/DType.h"
#include "Support/Nanobind/Pointer.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "_core/Driver/Buffer.h"
#include "_core/Graph/Bindings.h"
#include "_core/Graph/InferContext.h"
#include "_core/Profiler/Trace.h"

// Include type casters so that nanobind methods can take MLIR C API types.
#include "mlir/Bindings/Python/NanobindAdaptors.h" // IWYU pragma: keep (type casters)
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/raw_ostream.h"

#include "nanobind/nanobind.h"
#include "nanobind/stl/filesystem.h" // IWYU pragma: keep (type casters)
#include "nanobind/stl/shared_ptr.h" // IWYU pragma: keep (type casters)
#include <cstring>
#include <stdexcept>

// Backs Graph profile scope's _location() wrap loop: wraps `child` in a
// MOGG::ProfileScopeLocationAttr naming the active scope.
MlirLocation profile_scope_location(MlirContext ctx, const std::string &name,
                                    MlirLocation child,
                                    std::optional<std::string> color);

namespace nb = nanobind;
using namespace M::Graph;

// For named argument shorthand.
// See:
// https://nanobind.readthedocs.io/en/stable/basics.html#keyword-and-default-arguments
using namespace nanobind::literals;

namespace {
void load_modular_dialects(MlirDialectRegistry registry) {
  M::MO::loadDialects(*unwrap(registry));
}

MlirOperation next_operation(MlirOperation op) {
  mlir::Operation *x = unwrap(op);
  return wrap(x->getNextNode());
}

MlirOperation prev_operation(MlirOperation op) {
  mlir::Operation *x = unwrap(op);
  return wrap(x->getPrevNode());
}

MlirOperation last_operation(MlirBlock block) {
  mlir::Block *x = unwrap(block);

  if (x->empty())
    return wrap((mlir::Operation *)nullptr);
  return wrap(&x->back());
}

/// Initializes M::Context when no global context exists, then registers it on
/// the given MLIR context so loadContext() succeeds for graph compilation and
/// kernel analysis.
///
/// Reuses the existing global M::Context when present (e.g. after
/// InferenceSession) if Init::Options match; otherwise creates it via
/// Init::getOrCreateContext with cpuDevice options aligned with
/// Engine::Context::create's default session (see comments in-body).
static void _init_and_register_max_context(MlirContext mlirCtx) {
  M::Init::Options options;
  M::AsyncRT::CPUDeviceOptions rtOptions;
  // AsyncRT::CPUDeviceOptions defaults mainWillDonate to true; Engine's
  // getOrCreateCPUDeviceOptions forces false.
  rtOptions.mainWillDonate = false;
  options.withCPUDeviceOptions(rtOptions);
  // Engine::Config defaults isCrashLogEnabled() to false;
  // Engine::Context::create passes withForceDisableCrashReporting(!that).
  options.withForceDisableCrashReporting(true);

  auto ctxOr = M::Init::getOrCreateContext("max-graph", options);
  if (ctxOr.isError())
    throw std::runtime_error(ctxOr.takeError().get());
  M::ContextRef maxContext = ctxOr.takeValue();

  M::registerContext(*unwrap(mlirCtx), maxContext);
}

M::ArrayElementsAttr array_attr(M::Driver::Python::Buffer &buffer,
                                M::MO::TensorType &type) {
  // Create an ND array from data, taking the size and shape from the tensor
  // type argument.
  auto spec = buffer.getSpec();

  llvm::ArrayRef<uint8_t> bytesArray(llvm::bit_cast<uint8_t *>(buffer.data()),
                                     spec.getSizeInBytes());

  auto dims = llvm::SmallVector<int64_t>(spec.begin(), spec.end());
  auto shapedType =
      mlir::RankedTensorType::get(type.getIntDims(), type.getElementType());

  return M::ArrayElementsAttr::get({bytesArray.data(), bytesArray.size()},
                                   shapedType);
}

char *frame_traceback(void *frame_ptr) {
  // This is simply
  // `''.join(traceback.format_list(frame))`
  nb::gil_scoped_acquire acquire{};
  auto frame = nb::handle(reinterpret_cast<PyObject *>(frame_ptr));
  nb::object os = nb::module_::import_("traceback");
  nb::object formatStack = os.attr("format_list");
  auto empty = nb::str("");
  nb::object emptyJoin = empty.attr("join");

  auto outStr = nb::cast<std::string>(emptyJoin(formatStack(frame)));

  char *out = new char[outStr.size() + 1]{};
  // Copy to a string owned by the caller.
  outStr.copy(out, outStr.size());
  return out;
}

void frame_decref(void *frame_ptr) {
  // Destruction is always on the main thread controlled by python.
  // It will always have the gil.
  auto frame = nb::handle(reinterpret_cast<PyObject *>(frame_ptr));
  frame.dec_ref();
}

void frame_incref(void *frame_ptr) {
  // Destruction is always on the main thread controlled by python.
  // It will always have the gil.
  auto frame = nb::handle(reinterpret_cast<PyObject *>(frame_ptr));
  frame.inc_ref();
}

MlirLocation frame_loc(MlirContext ctx, nb::handle frame) {
  // The resulting location handles incrementing and decrementing the object's
  // reference count, as the object will be interned on the MLIR context.
  mlir::Location loc = M::RMO::PythonFrameAttr::getOpaque(
      unwrap(ctx), {.ptr = frame.ptr(),
                    .to_str = &frame_traceback,
                    .incref = &frame_incref,
                    .decref = &frame_decref});
  return wrap(loc);
}

// Serializes op and its descendants to MLIR bytecode with each op's
// Python-frame source location spelled out. The frames are attached as
// OpaqueLocs that hold a live callback and do not survive serialization, so
// they are materialized into serializable FusedLocs first. The materialization
// runs on a clone, so the caller's operation keeps its live OpaqueLocs.
//
// Bytecode rather than assembly text because this extension static-links its
// own LLVM: its textual float printer matches APFloat semantics by pointer
// against its own copy of the APFloatBase::sem* statics, so it mangles f64
// constants and aborts on any f32 that MLIR spells as a hex literal. Callers
// reparse this through max.mlir, which shares libmax's copy, and print there.
// See GEX-4052.
nb::bytes source_locations_bytecode(mlir::OpState &op) {
  mlir::OwningOpRef<mlir::Operation *> clone(op.getOperation()->clone());
  M::RMO::materializePythonFrameLocations(clone.get());
  std::string result;
  llvm::raw_string_ostream os(result);
  if (mlir::failed(mlir::writeBytecodeToFile(clone.get(), os)))
    throw std::runtime_error(
        "Failed to serialize operation with source locations to bytecode");
  return nb::bytes(result.data(), result.size());
}

/// Creates a Buffer directly from an MLIR constant attribute.
///
/// Supports:
/// - ArrayElementsAttr (#M.dense_array)
/// - AlignedBytesAttr (#M.aligned_bytes)
/// - DenseResourceElementsAttr (external blob)
///
/// Always copies data into a new buffer. Handles splatting when source
/// has 1 element but target shape requires more.
M::Driver::Python::Buffer
buffer_from_constant_attr(mlir::ElementsAttr attr, M::DType dtype,
                          const std::vector<size_t> &shape,
                          M::Driver::Python::PyDeviceTy device) {
  // Extract raw bytes based on attribute type
  llvm::ArrayRef<uint8_t> rawData;

  if (auto arrayAttr = llvm::dyn_cast<M::ArrayElementsAttr>(attr)) {
    rawData = arrayAttr.getRawData();
  } else if (auto alignedAttr = llvm::dyn_cast<M::AlignedBytesAttr>(attr)) {
    rawData = alignedAttr.getData();
  } else if (auto denseResource =
                 llvm::dyn_cast<mlir::DenseResourceElementsAttr>(attr)) {
    auto blob = denseResource.getRawHandle().getBlob();
    if (!blob)
      throw std::runtime_error("DenseResourceElementsAttr has no blob data");
    auto blobData = blob->getData();
    rawData = llvm::ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(blobData.data()), blobData.size());
  } else {
    throw std::runtime_error(
        "Unsupported constant attribute type for buffer creation");
  }

  // Calculate element counts
  size_t targetNumElements = 1;
  for (size_t dim : shape)
    targetNumElements *= dim;

  size_t elementSize = dtype.getSizeInBytes();
  size_t srcNumElements = rawData.size() / elementSize;

  // Validate for splatting
  if (srcNumElements != targetNumElements && srcNumElements != 1) {
    throw std::runtime_error(
        "Invalid constant: source has " + std::to_string(srcNumElements) +
        " elements but target has " + std::to_string(targetNumElements) +
        " elements (not a splat)");
  }

  // TODO(MXF-106): use Buffer.frombuffer when available.
  // Create buffer and copy data
  M::Driver::Python::Buffer buffer(dtype, shape, device);
  uint8_t *dstData = static_cast<uint8_t *>(buffer.data());

  if (srcNumElements == targetNumElements) {
    // Direct copy
    std::memcpy(dstData, rawData.data(), rawData.size());
  } else {
    // Splat: replicate single element
    for (size_t i = 0; i < targetNumElements; ++i) {
      std::memcpy(dstData + i * elementSize, rawData.data(), elementSize);
    }
  }

  return buffer;
}

} // namespace

void M::Graph::Python::bindGraphModule(nb::module_ m) {
  m.doc() = R"nbdoc(
        Provides Python bindings for the MAX Graph API.
    )nbdoc";

  nb::exception<KernelRegistrationError> registrationError(
      m, "KernelRegistrationError", PyExc_ValueError);
  registrationError.doc() =
      "Raised when a custom op package fails to load or check, or a custom "
      "op's Mojo kernel doesn't match its call. Where the error is in the "
      "Mojo code, the message points at it.";

  /// Attribute factories

  /// Register dialects needed to build MAX graphs.
  m.def("load_modular_dialects", &load_modular_dialects,
        "Registers all Modular MLIR dialects into the given registry.");

  /// Creates an array attribute from any Python object that implements the
  /// buffer protocol.
  /// Copies the buffer data into the array attribute interned into the
  /// MLIRContext.
  m.def("array_attr", &array_attr,
        "Creates an array attribute from a buffer and tensor type.");

  /// Creates a Buffer from an MLIR constant attribute (inverse of array_attr).
  /// Copies data and handles splatting.
  m.def("_buffer_from_constant_attr", &buffer_from_constant_attr,
        nb::arg("attr"), nb::arg("dtype"), nb::arg("shape"), nb::arg("device"),
        "Creates a Buffer from an MLIR constant attribute.");

  m.def("next_operation", &next_operation,
        "Returns the next operation in the parent block, or ``None``.");
  m.def("prev_operation", &prev_operation,
        "Returns the previous operation in the parent block, or ``None``.");
  m.def("last_operation", &last_operation,
        "Returns the last operation in the given block, or ``None``.");

  /// Converts a dtype to an MLIR type.
  m.def("dtype_to_type",
        infer_context(+[](mlir::MLIRContext *ctx, M::DType dtype) {
          auto mlirType = M::MO::toMLIRType(ctx, dtype);
          if (mlirType.isError())
            throw std::runtime_error("Invalid dtype");
          return *mlirType;
        }),
        "Converts a MAX :class:`DType` to the corresponding MLIR type.");

  m.def(
      "type_to_dtype",
      [](mlir::Type dtype) -> M::DType {
        ErrorOr<DType> type = M::MO::fromMLIRType(dtype);
        if (type.isError())
          throw std::runtime_error("Invalid type");
        return *type;
      },
      "Converts an MLIR type to the corresponding MAX :class:`DType`.");

  /// Creates an opaque location containing a python frame.
  m.def("frame_loc", &frame_loc,
        "Creates an opaque MLIR location containing a Python stack frame.");

  m.def("profile_scope_location", &profile_scope_location, "ctx"_a, "name"_a,
        "child"_a, "color"_a = std::nullopt,
        "Wraps `child` in a MOGG::ProfileScopeLocationAttr naming the active "
        "Graph.profile_scope(), with an optional color.");

  /// Serializes an operation to MLIR bytecode with Python source locations
  /// materialized, so a graph dumped without compiling it still shows where
  /// each op was built. Does not mutate the operation.
  m.def("source_locations_bytecode", &source_locations_bytecode, "op"_a,
        "Serializes the operation to MLIR bytecode with each op's Python "
        "source location materialized into printable form. Print it by "
        "reparsing through max.mlir. Does not mutate the operation.");

  m.def("_init_and_register_max_context", &_init_and_register_max_context,
        "mlir_ctx"_a,
        R"nbdoc(
      Initializes a process-wide M::Context when none exists, then registers it with the given MLIR context so compiler code can use loadContext.

      Internal API; does not expose M::Context to Python. New contexts are
      created via Init::getOrCreateContext (same Init path the Engine uses before
      attaching devices).
  )nbdoc");

  //===----------------------------------------------------------------------===//
  // KernelLibrary (backed by `mogg.kernel.decl` / `mogg.primitive.decl` ops)
  //===----------------------------------------------------------------------===//

  nb::class_<Python::KernelTorchInfo>(m, "KernelTorchInfo", R"nbdoc(
      The info needed to register a Mojo kernel as a PyTorch custom op.

      Built by :obj:`max.experimental.torch` from the kernel's ``execute``
      signature via ``KernelDeclAdaptor``.
  )nbdoc")
      .def_ro("num_dps_outputs", &Python::KernelTorchInfo::numDpsOutputs,
              "The number of leading destination-passing-style output "
              "arguments.")
      .def_ro("tensor_arg_names", &Python::KernelTorchInfo::tensorArgNames,
              "The source names of tensor-like arguments in declaration "
              "order (DPS outputs first, then inputs).");

  nb::class_<Python::Analysis>(m, "Analysis", R"nbdoc(
      Analyzes Mojo kernel libraries for custom operator integration.

      Loads one or more Mojo shared libraries and exposes their exported
      kernel operations for use in MAX graphs. Used internally to validate
      and register custom operations.

      Args:
          context: A Mojo library object or path.
          paths: The additional library search paths.
  )nbdoc")
      .def(nb::init<nb::object, std::vector<std::filesystem::path>>(),
           "context"_a, "paths"_a)
      .def_prop_ro("symbol_names", &Analysis::symbolNames,
                   "The list of exported kernel symbol names.")
      .def_prop_ro("library_paths", &Analysis::libraryPaths,
                   "The paths of all loaded libraries.")
      .def(
          "kernel",
          [](Analysis *self, const std::string &name) {
            return M::Graph::Python::castOp(self->kernelOp(name));
          },
          "Returns the MLIR operation for the named kernel symbol.")
      .def("verify_custom_op", &Analysis::verifyCustomOp,
           "Verifies that an operation matches the expected kernel signature.")
      .def("add_path", &Analysis::addPath,
           "Adds a library search path to this analysis.")
      .def("seed_kernel_decls", &Analysis::seedKernelDecls,
           "Copies kernel decl ops and the opaque-type mapping attribute "
           "into the given target module. Called by the compile path to "
           "seed a model module so the GC pipeline can skip "
           "``mogg-import-packages``.")
      .def("kernel_torch_info", &Analysis::kernelTorchInfo, "name"_a,
           "Returns the :class:`KernelTorchInfo` that "
           ":obj:`max.experimental.torch` needs to register the named kernel "
           "as a PyTorch custom op. Raises ``ValueError`` if the kernel is "
           "unknown or has an unsupported argument type.")
      .def("has_shape_function", &Analysis::hasShapeFunction, "name"_a,
           "Returns true if a shape function is registered for the named "
           "kernel.");

  M::Profiler::Python::addTracingToNanobindModuleClassMethods(
      m, "max._core.graph.");
}

MlirLocation profile_scope_location(MlirContext ctx, const std::string &name,
                                    MlirLocation child,
                                    std::optional<std::string> color) {
  mlir::MLIRContext *ctxPtr = unwrap(ctx);
  mlir::StringAttr colorAttr =
      color ? mlir::StringAttr::get(ctxPtr, *color) : mlir::StringAttr();
  mlir::Location loc = M::MOGG::ProfileScopeLocationAttr::get(
      ctxPtr, mlir::StringAttr::get(ctxPtr, name), unwrap(child), colorAttr);
  return wrap(loc);
}
