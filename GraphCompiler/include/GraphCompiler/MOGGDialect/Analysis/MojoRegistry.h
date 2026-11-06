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

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOREGISTRY_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOREGISTRY_H

#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/Support/PlatformPredicate.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>
#include <utility>

namespace M::MOGG {

/// Returns the set of devices that `decl` applies to: its `cond=` platform,
/// or else the devices that its `device_info` fields select.
ErrorOr<PlatformPredicate> getRegistrationPlatformPredicate(KernelDeclOp decl);

/// Indexes `mogg.kernel.decl`, `mogg.primitive.decl`, and
/// `mogg.shape_func.decl` ops by registration name and sym_name. Not an MLIR
/// analysis: build one per pass that needs lookups.
class MojoRegistry {
public:
  explicit MojoRegistry(mlir::ModuleOp module);

  /// Returns all `mogg.kernel.decl` ops registered under `name` (possibly
  /// several, possibly none). The view is valid for the registry's lifetime.
  llvm::ArrayRef<KernelDeclOp> lookupKernel(llvm::StringRef name) const;

  bool containsKernel(llvm::StringRef name) const;

  /// Returns the registered kernel name closest to `name` by edit distance,
  /// if one is close enough to be a likely typo.
  std::optional<llvm::StringRef> suggestKernelName(llvm::StringRef name) const;

  /// Asserts that the op exists.
  KernelDeclOp lookupKernelBySymName(llvm::StringRef symName) const;

  KernelDeclOp lookupKernelBySymNameOrNull(llvm::StringRef symName) const;

  /// Asserts that the primitive exists.
  PrimitiveDeclOp lookupPrimitive(llvm::StringRef name) const;

  bool containsPrimitive(llvm::StringRef name) const;

  /// Returns a null op if none exists.
  ShapeFunctionDeclOp lookupShapeFunction(llvm::StringRef name) const;

  /// Asserts that the op exists.
  ShapeFunctionDeclOp
  lookupShapeFunctionBySymName(llvm::StringRef symName) const;

  ShapeFunctionDeclOp
  lookupShapeFunctionBySymNameOrNull(llvm::StringRef symName) const;

  bool containsShapeFunction(llvm::StringRef name) const;

  /// Among candidates whose non-empty device-info fields all match `device`,
  /// picks a user-defined kernel over a built-in one, then the one with the
  /// most non-empty fields. Errors if none matches, and asserts if two
  /// candidates tie for the pick.
  ErrorOr<KernelDeclOp> getResolvedKernel(llvm::StringRef name,
                                          DeviceRefAttr device) const;

  /// Like the overload above, but matches `deviceInfo` instead of looking the
  /// device up in this registry's `mo.device_info_mapping`.
  ErrorOr<KernelDeclOp> getResolvedKernel(llvm::StringRef name,
                                          DeviceInfoAttr deviceInfo) const;

  /// Finds each op whose selection is ambiguous on some device, over all
  /// representative devices of its registrations. With `builtinsOnly`, only
  /// built-in registrations take part, as in a registry with no user
  /// packages. Appends one message for each op and set of tied kernels to
  /// `messages`, and returns an error when a registration has an invalid
  /// platform. An error does not stop the scan, so `messages` keeps every
  /// ambiguity found before and after it.
  ErrorOrSuccess
  findAmbiguousSelections(bool builtinsOnly,
                          llvm::SmallVectorImpl<std::string> &messages) const;

  /// Returns the first execution device of `op`, or the host device if none.
  static DeviceRefAttr getDeviceRefForOp(mlir::Operation *op);

private:
  ErrorOr<const PlatformPredicate *>
  getPlatformPredicate(KernelDeclOp decl) const;
  ErrorOr<PlatformPredicateOrder> getOrder(KernelDeclOp first,
                                           KernelDeclOp second) const;

  /// Returns the registrations of `matches` that no other registration of
  /// the selected provenance group is narrower than. All of `matches` hold
  /// for one device.
  ErrorOr<llvm::SmallVector<KernelDeclOp>>
  getSurvivors(llvm::ArrayRef<KernelDeclOp> matches) const;

  /// Describes why no registration of `survivors` is narrower than all the
  /// others.
  std::string describeAmbiguity(
      llvm::StringRef name, llvm::ArrayRef<KernelDeclOp> survivors,
      llvm::function_ref<void(llvm::raw_ostream &)> printDevice) const;

  MO::DeviceInfoMapping deviceInfoMapping;
  llvm::StringMap<llvm::SmallVector<KernelDeclOp>> kernelByName;
  llvm::StringMap<KernelDeclOp> kernelBySymName;
  llvm::StringMap<PrimitiveDeclOp> primitiveByName;
  llvm::StringMap<ShapeFunctionDeclOp> shapeFuncByName;
  llvm::StringMap<ShapeFunctionDeclOp> shapeFuncBySymName;

  // Computed on first use. Import validates every platform, so an error here
  // means a decl that did not come from `--mogg-import-packages`. The
  // platforms live on the heap because callers keep pointers to them while
  // the map grows; a null entry has an error in `platformErrorByDecl`.
  mutable llvm::DenseMap<mlir::Operation *, std::unique_ptr<PlatformPredicate>>
      platformByDecl;
  mutable llvm::DenseMap<mlir::Operation *, std::string> platformErrorByDecl;
  mutable llvm::DenseMap<std::pair<mlir::Operation *, mlir::Operation *>,
                         PlatformPredicateOrder>
      orderByPair;
};

/// Clones every `mogg.kernel.decl` and `mogg.primitive.decl` op from `src`
/// into `dst` (e.g. for constant-folding sub-modules), avoiding a rerun of
/// `--mogg-import-packages`. Does not deduplicate against decls already in
/// `dst`.
void copyKernelDecls(mlir::ModuleOp src, mlir::ModuleOp dst);

/// Verifies that an MO operation matches its registered Mojo kernel's
/// signature. Prefer this overload from long-running consumers (Python
/// `Analysis`) to avoid rebuilding the registry per call.
ErrorOrSuccess verifyCustomOp(const MojoRegistry &registry,
                              mlir::Operation *op);

/// Builds a `MojoRegistry` over `kernelDecls`, the module holding the decl
/// ops. Backs the public C API `MAXG_verifyCustomOp`.
ErrorOrSuccess verifyCustomOp(mlir::ModuleOp kernelDecls, mlir::Operation *op);

/// Resolves the paths of the built-in packages (`kernels`, `primitives`) that
/// `--mogg-import-packages` loads. See also `getKernelPackagePathsForCacheKey`.
ErrorOr<llvm::SmallVector<std::string>>
getBuiltInKernelPackagePaths(mlir::MLIRContext *ctx);

/// Resolves the built-in packages plus every binary package on the import
/// path. The built-ins mostly re-export symbols implemented in import-path
/// packages (`linalg`, `nn`, ...), so the MEF cache key must cover those too.
/// Errors only if the built-in packages can't be resolved.
ErrorOr<llvm::SmallVector<std::string>>
getKernelPackagePathsForCacheKey(mlir::MLIRContext *ctx);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOREGISTRY_H
