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

#include "_core/Graph/Bindings.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "Support/Nanobind/TypeCasters.h" // IWYU pragma: keep (type casters)
#include "max/max_license/MOF/API/c/detail/mojo_analysis.h"
#include "mlir/CAPI/IR.h"
#include "llvm/Support/FormatVariadic.h"

namespace nb = nanobind;

mlir::MLIRContext *M::Graph::Python::current_context() {
  static nb::handle context = nb::module_::import_("max.mlir").attr("Context");
  nb::handle current = context.attr("current");
  if (current.is_none())
    throw std::runtime_error("No active MLIR context");
  using Caster = nb::detail::make_caster<mlir::MLIRContext>;
  Caster caster;
  if (!caster.from_python(current, 0, {})) {
    throw std::runtime_error("Failed to infer MLIR context.");
  }
  return caster;
}

std::vector<std::string> M::Graph::Python::Analysis::symbolNames() {
  // Touch ensureRegistry so the cache is hot for later queries, even though
  // we iterate the module directly here (registry doesn't expose iteration).
  (void)ensureRegistry();
  mlir::ModuleOp module = *kernelDecls;
  std::vector<std::string> symbols;
  for (auto decl : module.getOps<M::MOGG::KernelDeclOp>())
    symbols.emplace_back(decl.getRegistration().str());
  for (auto decl : module.getOps<M::MOGG::PrimitiveDeclOp>())
    symbols.emplace_back(decl.getRegistration().str());
  return symbols;
}

std::vector<std::filesystem::path> M::Graph::Python::Analysis::libraryPaths() {
  return std::vector<std::filesystem::path>(libraryPathsSet.begin(),
                                            libraryPathsSet.end());
}

mlir::Operation *M::Graph::Python::Analysis::kernelOp(const std::string &name) {
  const M::MOGG::MojoRegistry &reg = ensureRegistry();
  auto kernels = reg.lookupKernel(name);
  if (!kernels.empty()) {
    M::MOGG::KernelDeclOp kernel = kernels.front();
    return kernel.getOperation();
  }
  if (reg.containsPrimitive(name)) {
    M::MOGG::PrimitiveDeclOp prim = reg.lookupPrimitive(name);
    return prim.getOperation();
  }
  return nullptr;
}

void M::Graph::Python::Analysis::verifyCustomOp(MlirOperation op) {
  auto err = M::MOGG::verifyCustomOp(ensureRegistry(), unwrap(op));
  if (err.isError())
    throw KernelRegistrationError(err.getError());
}

void M::Graph::Python::Analysis::seedKernelDecls(MlirOperation target) {
  auto targetModule = mlir::cast<mlir::ModuleOp>(unwrap(target));
  MAXG_seedKernelDecls(wrap(targetModule), wrap(ensureModule()));
}

namespace {

/// Reformat a `KernelDeclOpaqueAdaptor` dotted path (e.g. `"std.String"`,
/// `"std.utils.index.IndexList"`) as the legacy `"root::leaf"` presentation
/// that the torch error message expects.
std::string toRootLeafTypeName(llvm::StringRef dotted) {
  llvm::StringRef root = dotted.split('.').first;
  llvm::StringRef leaf = dotted.rsplit('.').second;
  if (leaf.empty() || leaf == root)
    return root.str();
  return (root + "::" + leaf).str();
}

/// Stringify a rejected argument's type for the unsupported-arg error.
std::string describeRejectedArg(const M::MOGG::KernelDeclOperandAdaptor &arg) {
  if (const auto *o =
          std::get_if<M::MOGG::KernelDeclOpaqueAdaptor>(&arg.underlyingType))
    return toRootLeafTypeName(o->typeName.strref());
  if (arg.isVariadicTensorType())
    return M::MOGG::kMojoVariadicTensorsName.str();
  return "<unknown>";
}

} // namespace

M::Graph::Python::KernelTorchInfo
M::Graph::Python::Analysis::kernelTorchInfo(const std::string &name) {
  const M::MOGG::MojoRegistry &reg = ensureRegistry();
  auto kernels = reg.lookupKernel(name);
  if (kernels.empty())
    throw nb::value_error(
        llvm::formatv("No kernel registered as '{0}'.", name).str().c_str());

  M::MOGG::KernelDeclAdaptor adaptor(kernels.front());
  if (!adaptor.executeFunction)
    throw nb::value_error(
        llvm::formatv("Kernel '{0}' has no execute function.", name)
            .str()
            .c_str());

  KernelTorchInfo info;
  const M::MOGG::KernelDeclFunctionAdaptor &fn = *adaptor.executeFunction;
  info.numDpsOutputs = fn.outputArguments.size();

  auto collect = [&](const M::MOGG::KernelDeclOperandAdaptor &arg) {
    // Regular tensors and scalar SIMD args become torch.Tensor parameters.
    if ((arg.isTensorType() && !arg.isVariadicTensorType()) ||
        arg.isScalarType()) {
      info.tensorArgNames.push_back(
          arg.sourceDescriptor->sourceName.strref().str());
      return;
    }
    // Device contexts are supplied implicitly by the graph compiler.
    if (arg.isContextType())
      return;
    throw nb::value_error(
        llvm::formatv("Unsupported argument type '{0}' in custom op '{1}'.",
                      describeRejectedArg(arg), name)
            .str()
            .c_str());
  };

  for (const auto &arg : fn.outputArguments)
    collect(arg);
  for (const auto &arg : fn.inputArguments)
    collect(arg);

  return info;
}

bool M::Graph::Python::Analysis::hasShapeFunction(const std::string &name) {
  const M::MOGG::MojoRegistry &reg = ensureRegistry();
  return reg.containsShapeFunction(name);
}

void M::Graph::Python::Analysis::addPath(std::filesystem::path path) {
  auto abs = std::filesystem::absolute(path);
  bool inserted = libraryPathsSet.insert(abs).second;
  if (!inserted)
    return;
  // If the module has already been built, append the new path; otherwise the
  // upcoming `ensureModule` call will pick it up.
  if (kernelDecls) {
    auto err = MAXG_addKernelPackage(wrap(*kernelDecls), abs);
    if (err.isError()) {
      // The failed package was rolled back; forget it so it isn't retried.
      libraryPathsSet.erase(abs);
      throw KernelRegistrationError(err.getError());
    }
    // Invalidate the cached registry: the new package's decls aren't in it.
    registry.reset();
  }
}

M::Graph::Python::Analysis::Analysis(nb::object context,
                                     std::vector<std::filesystem::path> paths)
    : context(context) {
  for (const auto &path : paths)
    libraryPathsSet.insert(std::filesystem::absolute(path));
}

M::Graph::Python::Analysis::~Analysis() = default;

const M::MOGG::MojoRegistry &M::Graph::Python::Analysis::ensureRegistry() {
  mlir::ModuleOp module = ensureModule();
  if (!registry)
    registry = std::make_unique<M::MOGG::MojoRegistry>(module);
  return *registry;
}

mlir::ModuleOp M::Graph::Python::Analysis::ensureModule() {
  if (kernelDecls)
    return *kernelDecls;
  MlirContext ctx = nb::cast<MlirContext>(context);
  auto moduleOr = MAXG_createKernelDeclsModule(ctx);
  if (moduleOr.isError())
    throw nb::value_error(moduleOr.getError());
  kernelDecls = moduleOr.takeValue();
  // Load all currently-registered paths into the fresh module. The first
  // call to MAXG_addKernelPackage on an empty module also loads the built-in
  // `kernels` / `primitives` packages; subsequent calls append. A package that
  // fails is rolled back and forgotten, and the others still load, so the
  // module stays usable after the error is raised.
  std::string errors;
  for (const auto &path :
       std::vector(libraryPathsSet.begin(), libraryPathsSet.end())) {
    auto err = MAXG_addKernelPackage(wrap(*kernelDecls), path);
    if (err.isError()) {
      libraryPathsSet.erase(path);
      errors += err.getError();
    }
  }
  // If no user paths were registered, still load built-ins by importing
  // with an empty mojo-pkg-paths list. Detect this case as "module is still
  // empty after the user-paths loop".
  if (libraryPathsSet.empty()) {
    auto err = MAXG_addKernelPackage(wrap(*kernelDecls), {});
    if (err.isError())
      throw nb::value_error(err.getError());
  }
  if (!errors.empty())
    throw KernelRegistrationError(errors);
  return *kernelDecls;
}

namespace {
llvm::DenseMap<mlir::TypeID, const std::type_info *> &typeIDRegistry() {
  static llvm::DenseMap<mlir::TypeID, const std::type_info *> registry;
  return registry;
}
} // namespace

void M::Graph::Python::registerTypeID(mlir::TypeID id,
                                      const std::type_info *info) {
  typeIDRegistry()[id] = info;
}

const std::type_info *M::Graph::Python::lookupTypeID(mlir::TypeID id) {
  auto found = typeIDRegistry().find(id);
  if (found == typeIDRegistry().end())
    return nullptr;
  return found->second;
}
