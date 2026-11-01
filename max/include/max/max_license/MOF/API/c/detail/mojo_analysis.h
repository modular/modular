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

#ifndef MOF_API_C_DETAIL_MOJO_ANALYSIS_H
#define MOF_API_C_DETAIL_MOJO_ANALYSIS_H

#include "Support/ForwardDecls.h"
#include "max/max_license/MOF/API/c/symbol_export.h"
#include "mlir/CAPI/Wrap.h"
#include <filesystem>

class MlirContext;
class MlirModule;
class MlirOperation;

namespace mlir {
class ModuleOp;
template <typename OpTy>
class OwningOpRef;
} // namespace mlir

//===----------------------------------------------------------------------===//
// Kernel registry API. Backed by `mogg.kernel.decl` / `mogg.primitive.decl`
// ops stored in a standalone `ModuleOp`. The module is populated via
// `MAXG_addKernelPackage` and consumed by `MAXG_verifyCustomOp`.
//===----------------------------------------------------------------------===//

/// Create an empty `ModuleOp` to back a kernel registry. The caller owns the
/// returned module via `OwningOpRef`; destroying it releases the registered
/// kernels.
MODULAR_API_VISIBILITY_EXPORT
M::ErrorOr<mlir::OwningOpRef<mlir::ModuleOp>>
MAXG_createKernelDeclsModule(MlirContext context);

/// Append the kernels declared in the Mojo package at `path` to `module`.
/// Built-in `kernels` and `primitives` packages are loaded automatically
/// the first time this is called on an empty module; subsequent calls only
/// load the supplied `path`.
MODULAR_API_VISIBILITY_EXPORT
M::ErrorOrSuccess MAXG_addKernelPackage(MlirModule module,
                                        std::filesystem::path path);

/// Verify that `op` matches its registered Mojo kernel's signature. Checks
/// operand types and (for `mo.custom`) user-supplied parameter names against
/// the kernel's compile-time parameter list.
MODULAR_API_VISIBILITY_EXPORT
M::ErrorOrSuccess MAXG_verifyCustomOp(MlirModule module, MlirOperation op);

/// Copy `mogg.kernel.decl` / `mogg.primitive.decl` ops and the
/// `mogg.opaque_type_mapping` attribute from `source` into `target`. Used
/// just before invoking the graph compiler to seed a freshly-built model
/// module with the kernel decls that an `Analysis`-owned module already has
/// from earlier `MAXG_addKernelPackage` calls. After this, the GC pipeline
/// can skip `mogg-import-packages` (since `target` already has everything
/// the pipeline would otherwise build).
MODULAR_API_VISIBILITY_EXPORT
void MAXG_seedKernelDecls(MlirModule target, MlirModule source);

#endif // MOF_API_C_DETAIL_MOJO_ANALYSIS_H
