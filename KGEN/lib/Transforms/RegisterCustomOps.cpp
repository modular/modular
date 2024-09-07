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
class RegisterCustomOpsPass
    : public impl::RegisterCustomOpsBase<RegisterCustomOpsPass> {
public:
  RegisterCustomOpsPass(CompileCanonicalizationFnsFn &&compilePatternsFn = {})
      : compilePatternsFn(std::move(compilePatternsFn)) {}

  void runOnOperation() override;

private:
  /// Given the main module, slice the definitions of functions reuiqred for the
  /// implementation of the ops. In another words, this creates a module rooted
  /// at the `impl` and `canonicalize` functions for each custom op.
  static OwningOpRef<ModuleOp> sliceCustomModule(SymbolTable &symtab,
                                                 CustomOpImplsOp impls);

  /// Given the custom op module, JIT and compile the canonicalizer patterns
  /// within the module for each custom op.
  ErrorOrSuccess compilePatterns(OwningOpRef<ModuleOp> &&opImplsModule);

  /// The function to compile the canonicalization patterns out of a
  /// pre-elaboration module.
  CompileCanonicalizationFnsFn compilePatternsFn;
};

} // namespace

OwningOpRef<ModuleOp>
RegisterCustomOpsPass::sliceCustomModule(SymbolTable &symtab,
                                         CustomOpImplsOp impls) {
  ExportMap exportedSymbols;
  // This operation maps custom operation names to their implementations.
  // It should stay as exported to not get deleted.
  exportedSymbols.try_emplace(
      StringAttr::get(impls.getContext(), CustomOpImplsOp::kSymbolName),
      ExportKind::Exported);

  // Add all op implementation and canonicalization patterns.
  // They specifically are not marked as exported.
  for (CustomOpImplAttr impl : impls.getImpls()) {
    exportedSymbols.try_emplace(
        impl.getOpImplementation().getSymbol().getRootReference(),
        ExportKind::NotExported);

    if (SymbolConstantAttr canonSym = impl.getOpCanonicalization()) {
      exportedSymbols.try_emplace(canonSym.getSymbol().getRootReference(),
                                  ExportKind::NotExported);
    }
  }

  // Create a new module that will contain all the symbols, and place it in a
  // dense resoruce attribute.
  return produceStandaloneModule(symtab, exportedSymbols);
}

ErrorOrSuccess
RegisterCustomOpsPass::compilePatterns(OwningOpRef<ModuleOp> &&opImplsModule) {
  auto customDialect = getContext().getLoadedDialect<CustomDialect>();
  // Get the op canonicalization patterns symbols from the op implementation
  // module.
  SymbolTable opImplsTable(*opImplsModule);
  auto opImplOp = CustomOpImplsOp::lookupOp(*opImplsModule);
  DenseMap<StringAttr, SymbolConstantAttr> canonicalizationSyms;
  for (auto opImplAttr : opImplOp.getImpls()) {
    auto canonicalizationSym = opImplAttr.getOpCanonicalization();
    if (!canonicalizationSym)
      continue;
    canonicalizationSyms.try_emplace(opImplAttr.getOpName(),
                                     canonicalizationSym);

    // Set the operation as exported so it doesn't get DCE'd.
    opImplsTable
        .lookup<ExportInterface>(
            canonicalizationSym.getSymbol().getLeafReference())
        .setExported();
  }

  // If there are no canonicalization patterns, then just exit.
  if (canonicalizationSyms.empty()) {
    customDialect->areCanonicalizationFnLoaded = true;
    return success();
  }

  ErrorOr<TargetInfoAttr> targetOr =
      getTargetInfoFor(&getContext(), llvm::sys::getDefaultTargetTriple(),
                       llvm::sys::getHostCPUName(), getHostCPUFeatures());
  if (targetOr.isError())
    return targetOr.takeError();
  TargetInfoAttr target = targetOr.takeValue();

  // Compile them.
  auto errorOrCanonFn =
      compilePatternsFn(*opImplsModule, canonicalizationSyms, target);
  if (errorOrCanonFn.isError())
    return errorOrCanonFn.takeError();

  // Insert jit'ed canonicalization patterns to the custom dialect.
  for (auto &[name, capiCanonFn] : errorOrCanonFn.takeValue()) {
    auto canonFunc = [func = capiCanonFn](Operation *op,
                                          PatternRewriter &rewriter) mutable {
      // Both the operation and the rewriter are passed as pointers, as the
      // mojo canonicalization pattern is marked as inout.
      MlirOperation c_op = wrap(op);
      MlirRewriterBase c_rewriter = wrap(&rewriter);
      return mlir::success(func(&c_op, &c_rewriter));
    };
    customDialect->canonicalizationFns.try_emplace(name, canonFunc);
  }

  customDialect->areCanonicalizationFnLoaded = true;

  // Return them.
  return success();
}

void RegisterCustomOpsPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto impls = CustomOpImplsOp::lookupOp(module);

  // If there are no custom ops, exit early.
  if (!impls)
    return;
  // If the impls operation exists but there are no custom ops, remove it and
  // exit early.
  if (impls.getImpls().empty()) {
    impls.erase();
    return;
  }

  auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
  SymbolTable &symtab = analysis.getTopLevelSymbolTable();

  // Slice a module containing the custom op definitions.
  OwningOpRef<ModuleOp> customOpModule = sliceCustomModule(symtab, impls);

  DenseResourceElementsAttr customOpResource =
      writeModuleToBytecodeAttr(*customOpModule);
  module->setAttr(kCustomOpImplModuleAttr, customOpResource);

  ErrorOrSuccess errOrSucc = compilePatterns(std::move(customOpModule));
  if (errOrSucc.isError()) {
    mlir::emitError(module.getLoc())
        << "Error while JIT'ing custom canonicalization patterns: "
        << errOrSucc.getError();
    signalPassFailure();
    return;
  }

  // Erase the custom op definitions mapping.
  // This allows custom op definitions to be DCE'd, as now all
  // information is stored in the dense resource attribute attached to
  // the main module.
  impls.erase();
}

std::unique_ptr<mlir::Pass>
KGEN::createRegisterCustomOps(CompileCanonicalizationFnsFn compilePatternsFn) {
  return std::make_unique<RegisterCustomOpsPass>(std::move(compilePatternsFn));
}
