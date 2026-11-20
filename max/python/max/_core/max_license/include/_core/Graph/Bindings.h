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

#ifndef SDK_GRAPHAPI_PYTHON_BINDINGS_H
#define SDK_GRAPHAPI_PYTHON_BINDINGS_H

#include "mlir-c/IR.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "nanobind/nanobind.h"
#include <cstddef>
#include <filesystem>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Forward decl
namespace M::KGEN {
class GeneratorOp;
}
namespace M::MOGG {
class MojoRegistry;
}

namespace M::Graph::Python {

/// Info needed by `max.experimental.torch` to register a Mojo kernel as a
/// `torch.library.custom_op`. Built from the kernel's `execute` signature
/// via `M::MOGG::KernelDeclAdaptor`, with the torch-side type policy applied
/// (regular tensors and `std::SIMD` accepted as `torch.Tensor` parameters,
/// device contexts ignored, everything else rejected).
struct KernelTorchInfo {
  size_t numDpsOutputs;
  /// Source names of tensor-like arguments in declaration order
  /// (DPS outputs first, then inputs). Used to build the synthesised
  /// `inspect.Signature` for the registered torch op.
  std::vector<std::string> tensorArgNames;
};

/// Get the currently active MLIR context from Python
mlir::MLIRContext *current_context();

void bindBuiltinDialect(nanobind::module_);
void bindKGENDialect(nanobind::module_);
void bindMDialect(nanobind::module_);
void bindMODialect(nanobind::module_);
void bindMOSHDialect(nanobind::module_);
void bindRMODialect(nanobind::module_);

void bindDriverModule(nanobind::module_);
void bindDTypeModule(nanobind::module_);
void bindEngineModule(nanobind::module_);
void bindGraphModule(nanobind::module_);
void bindProfilerModule(nanobind::module_);
void bindNIXLModule(nanobind::module_);
void bindSafetensorsModule(nanobind::module_);
void bindXXHashModule(nanobind::module_);
void bindXGrammarModule(nanobind::module_);
void bindMetricsModule(nanobind::module_);
void bindRequestContextModule(nanobind::module_);
void bindDLPackModule(nanobind::module_);

//===----------------------------------------------------------------------===//
// KernelLibrary: Python-facing wrapper over a standalone `ModuleOp` populated
// with `mogg.kernel.decl` / `mogg.primitive.decl` ops by the
// `--mogg-import-packages` pass. Replaces the older `MojoLibraryAnalysis`
// side-table.
//===----------------------------------------------------------------------===//

/// A custom op package failed to load or check, or a kernel doesn't match its
/// call. Raised in Python as `max.graph.KernelRegistrationError`, a
/// `ValueError`.
class KernelRegistrationError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class Analysis : std::enable_shared_from_this<Analysis> {
public:
  Analysis(nanobind::object context, std::vector<std::filesystem::path> paths);
  std::vector<std::string> symbolNames();
  std::vector<std::filesystem::path> libraryPaths();
  mlir::Operation *kernelOp(const std::string &name);
  void verifyCustomOp(MlirOperation op);
  void addPath(std::filesystem::path path);
  // Copy `mogg.kernel.decl` / `mogg.primitive.decl` ops and the
  // `mogg.opaque_type_mapping` attribute from the analysis's standalone
  // module into `target`. Called by the compile path on a model's MLIR
  // module (a `ModuleOp` Operation) just before invoking the graph
  // compiler.
  void seedKernelDecls(MlirOperation target);
  // Walk the named kernel's `execute` signature via `KernelDeclAdaptor` and
  // return the torch-side info: number of DPS outputs and the source names
  // of tensor-like arguments. Throws `nb::value_error` if `name` is not a
  // registered kernel or if any argument has a type the torch path can't
  // marshal as a `torch.Tensor`.
  KernelTorchInfo kernelTorchInfo(const std::string &name);
  /// Whether the op bound to `name` has a registered `mogg.shape_func.decl`.
  bool hasShapeFunction(const std::string &name);

private:
  // Keep the Python MLIR context alive at least as long as the standalone
  // module; the module's ops reference types/attributes owned by the context.
  nanobind::object context;
  // Cumulative set of registered package paths. Maintained client-side so
  // `libraryPaths()` doesn't have to re-derive it from the module.
  std::set<std::filesystem::path> libraryPathsSet;
  // Lazily built on first access; holds the `mogg.kernel.decl` and
  // `mogg.primitive.decl` ops produced by `--mogg-import-packages` for the
  // accumulated paths.
  mlir::OwningOpRef<mlir::ModuleOp> kernelDecls;
  // Lazily built from `kernelDecls` and reused across query / verify calls.
  // Invalidated whenever `addPath` mutates the module. Heap-allocated and
  // forward-declared so this header doesn't drag the GC `MojoRegistry`
  // header (and its transitive includes) into Python-binding TUs.
  std::unique_ptr<M::MOGG::MojoRegistry> registry;

public:
  // Out-of-line dtor because `MojoRegistry` is only forward-declared above.
  ~Analysis();

private:
  // Ensures `kernelDecls` is populated for all currently-registered paths
  // and returns it.
  mlir::ModuleOp ensureModule();
  // Ensures `registry` is built from the current `kernelDecls` and returns
  // a reference. Each callsite that does multiple registry queries should
  // bind a const-ref once instead of re-calling.
  const M::MOGG::MojoRegistry &ensureRegistry();
};

// `mlir::Pass` has non-inline virtual functions. Since we compile it without
// typeinfo, and the place where C++ defines the typeinfo to be doesn't contain
// it, we need to bind a wrapper class instead.
struct Pass {
  std::unique_ptr<mlir::Pass> pass;
  Pass(std::unique_ptr<mlir::Pass> pass) : pass(std::move(pass)) {}
};

} // namespace M::Graph::Python

#endif // SDK_GRAPHAPI_PYTHON_BINDINGS_H
