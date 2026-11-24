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

#include "max/max_license/MOF/API/c/detail/mojo_analysis.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "Support/Diagnostics/FormatScopedDiagnosticHandler.h"
#include "Support/ErrorOr.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FormatVariadic.h"

using namespace M;

M::ErrorOr<mlir::OwningOpRef<mlir::ModuleOp>>
MAXG_createKernelDeclsModule(MlirContext context) {
  mlir::MLIRContext *ctx = unwrap(context);
  mlir::OpBuilder builder(ctx);
  // Use an UnknownLoc; the module is a kernel-decls container, not associated
  // with any user source location.
  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(builder.getUnknownLoc()));
  return std::move(module);
}

M::ErrorOrSuccess MAXG_addKernelPackage(MlirModule module,
                                        std::filesystem::path path) {
  mlir::ModuleOp mod = unwrap(module);
  mlir::MLIRContext *ctx = mod.getContext();

  // First call to a fresh module loads built-ins; subsequent calls append the
  // delta. We detect the "first call" case as "module has no kernel decls
  // yet" so the caller doesn't need to track this themselves. Passing an
  // empty path is valid on the first call: it loads built-ins only.
  bool isFirstCall = mod.getOps<MOGG::KernelDeclOp>().empty() &&
                     mod.getOps<MOGG::PrimitiveDeclOp>().empty();

  MOGG::ImportPackagesOptions opts;
  if (!path.empty())
    opts.mojoPkgPaths.push_back(path.string());
  opts.append = !isFirstCall;

  mlir::PassManager importPm(ctx);
  importPm.addPass(MOGG::createImportPackages(opts));
  // Graph build skips the compile-time kernel checks, so a package is
  // checked here, where its errors can still point at its Mojo source.
  mlir::PassManager verifyPm(ctx);
  verifyPm.addPass(MOGG::createVerifyKernels());

  // A failed package must not stay half-loaded, or every later load would
  // report its errors again: remember what the module held before.
  llvm::SmallPtrSet<mlir::Operation *, 32> before;
  for (mlir::Operation &op : *mod.getBody())
    before.insert(&op);
  mlir::DictionaryAttr attrsBefore = mod->getAttrDictionary();

  // Formatted after the run: printing from inside a handler would register
  // another one while the engine walks its handler list.
  std::vector<mlir::Diagnostic> errors;
  mlir::ScopedDiagnosticHandler handler(ctx, [&](mlir::Diagnostic &diag) {
    if (diag.getSeverity() != mlir::DiagnosticSeverity::Error)
      return mlir::failure();
    errors.push_back(std::move(diag));
    return mlir::success();
  });
  bool imported = mlir::succeeded(importPm.run(mod));
  if (imported && (path.empty() || mlir::succeeded(verifyPm.run(mod))))
    return success();

  for (mlir::Operation &op :
       llvm::make_early_inc_range(llvm::reverse(*mod.getBody())))
    if (!before.contains(&op))
      op.erase();
  mod->setAttrs(attrsBefore);

  // A kernel check failure is about the user's Mojo, so it reads like
  // `verifyCustomOp`'s errors, without the import's header.
  std::string message =
      imported
          ? ""
          : llvm::formatv(
                "MAXG_addKernelPackage: failed to import kernels from '{0}'",
                path.string())
                .str();
  for (const mlir::Diagnostic &diag : errors)
    message += llvm::formatv(
        "\n{0}", FormatScopedDiagnosticHandler::formatWithSource(diag));
  return Error(message);
}

M::ErrorOrSuccess MAXG_verifyCustomOp(MlirModule module, MlirOperation op) {
  return MOGG::verifyCustomOp(unwrap(module), unwrap(op));
}

void MAXG_seedKernelDecls(MlirModule target, MlirModule source) {
  MOGG::copyKernelDecls(unwrap(source), unwrap(target));
}
