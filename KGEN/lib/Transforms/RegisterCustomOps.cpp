//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/CustomDialect/CustomUtils.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/TransformUtils/SlicingUtils.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace Custom;

namespace M::KGEN {
#define GEN_PASS_DEF_REGISTERCUSTOMOPS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct RegisterCustomOpsPass
    : KGEN::impl::RegisterCustomOpsBase<RegisterCustomOpsPass> {

  LogicalResult initialize(MLIRContext *ctx) override;

  /// Slice custom operation definitions from the module to a new module, that
  /// is then translated into bytecode and stored in a dense resource attribute.
  static DenseResourceElementsAttr
  sliceCustomModuleToAttr(ModuleOp theModule, SymbolTableCollection &tables);

  void runOnOperation() override;

private:
  /// The compilation target.
  TargetInfoAttr target;
};

} // namespace

LogicalResult RegisterCustomOpsPass::initialize(MLIRContext *ctx) {
  // Get the default target, as the JIT is going to be ran on the host.
  if (!target) {
    ErrorOr<TargetInfoAttr> targetOr =
        getTargetInfoFor(ctx, llvm::sys::getDefaultTargetTriple(),
                         llvm::sys::getHostCPUName(), getHostCPUFeatures());
    if (targetOr.isError())
      return mlir::emitError(UnknownLoc::get(ctx), targetOr.getError());
    target = targetOr.takeValue();
  }
  return success();
}

DenseResourceElementsAttr
RegisterCustomOpsPass::sliceCustomModuleToAttr(ModuleOp theModule,
                                               SymbolTableCollection &tables) {
  auto implsOp = CustomOpImplsOp::lookupOp(theModule);

  // Do not clone the module if no custom operations are defined.
  if (!implsOp)
    return {};

  ExportMap exportedSymbols;
  // This operation maps custom operation names to their implementations.
  // It should stay as exported to not get deleted.
  exportedSymbols.try_emplace(
      StringAttr::get(implsOp.getContext(), CustomOpImplsOp::kSymbolName),
      ExportKind::Exported);

  // Add all op implementation and canonicalization patterns.
  // They specifically are not marked as exported.
  for (CustomOpImplAttr impl : implsOp.getImpls()) {
    exportedSymbols.try_emplace(
        impl.getOpImplementation().getSymbol().getRootReference(),
        ExportKind::NotExported);

    SymbolConstantAttr canonSym = impl.getOpCanonicalization();
    if (!canonSym)
      continue;
    exportedSymbols.try_emplace(canonSym.getSymbol().getRootReference(),
                                ExportKind::NotExported);
  }

  // Create a new module that will contain all the symbols, and place it in a
  // dense resoruce attribute.
  OwningOpRef<ModuleOp> newModule = produceStandaloneModule(
      tables.getSymbolTable(theModule.getOperation()), exportedSymbols);
  return writeModuleToBytecodeAttr(*newModule);
}

void RegisterCustomOpsPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  auto implsOp = CustomOpImplsOp::lookupOp(theModule);

  // No canonicalization patterns here, or a failure somewhere, so we exit
  // early.
  if (!implsOp)
    return;

  // No canonicalization here, so we exit early as well.
  if (implsOp.getImpls().empty()) {
    implsOp.erase();
    return;
  }

  auto &symtabAnalysis = getAnalysis<mlir::SymbolTableAnalysis>();
  SymbolTableCollection &tables = symtabAnalysis.getSymbolTables();

  // Slice a module for the op implementations.
  DenseResourceElementsAttr customOpResource =
      sliceCustomModuleToAttr(theModule, tables);
  theModule->setAttr(kCustomOpImplModuleAttr, customOpResource);

  // Erase the custom op definitions mapping.
  // This allows custom op definitions to be DCE'd, as now all information is
  // stored in the dense resource attribute attached to the main module.
  implsOp.erase();
}
