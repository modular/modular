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
  // Slice out a new module rooted at the custom ops. This pulls out their
  // definitions: their implementation methods and canonicalizer methods.
  ExportMap exportedSymbols;
  exportedSymbols.try_emplace(
      StringAttr::get(impls.getContext(), CustomOpImplsOp::kSymbolName),
      ExportKind::Exported);
  return produceStandaloneModule(symtab, exportedSymbols);
}

ErrorOrSuccess
RegisterCustomOpsPass::compilePatterns(OwningOpRef<ModuleOp> &&opImplsModule) {
  auto customDialect = getContext().getLoadedDialect<CustomDialect>();
  SymbolTable opImplsTable(*opImplsModule);
  auto opImplOp = CustomOpImplsOp::lookupOp(*opImplsModule);

  // Compute a mapping from custom op name to canonicalizer pattern. Mark them
  // as exported in the sliced module.
  DenseMap<StringAttr, SymbolConstantAttr> canonicalizationSyms;
  for (auto opImplAttr : opImplOp.getImpls()) {
    auto symbol = opImplAttr.getOpCanonicalization();
    if (!symbol)
      continue;
    canonicalizationSyms.try_emplace(opImplAttr.getOpName(), symbol);

    // Export the functions to compile time.
    opImplsTable.lookup<ExportInterface>(symbol.getSymbol().getLeafReference())
        .setExported();
  }

  // If there are no canonicalization patterns, then just exit.
  if (canonicalizationSyms.empty()) {
    customDialect->areCanonicalizationFnLoaded = true;
    return success();
  }

  // Compile the patterns for the compiler host target.
  ErrorOr<TargetInfoAttr> target =
      getTargetInfoFor(&getContext(), llvm::sys::getDefaultTargetTriple(),
                       llvm::sys::getHostCPUName(), getHostCPUFeatures());
  if (target.isError())
    return target.takeError();

  auto canonFuncs =
      compilePatternsFn(*opImplsModule, canonicalizationSyms, *target);
  if (canonFuncs.isError())
    return canonFuncs.takeError();

  // Add the JIT'd canonicalizer functions into the Custom Dialect.
  for (auto &[name, capiCanonFn] : canonFuncs.takeValue()) {
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

  // Serialize and store these on the module so they can be accessed later.
  DenseResourceElementsAttr customOpResource =
      writeModuleToBytecodeAttr(*customOpModule);
  module->setAttr(kCustomOpImplModuleAttr, customOpResource);

  // Attempt to compile the canonicalization patterns.
  if (ErrorOrSuccess errOrSucc = compilePatterns(std::move(customOpModule))) {
    mlir::emitError(module.getLoc())
        << "error while JIT'ing custom canonicalization patterns: "
        << errOrSucc.getError();
    return signalPassFailure();
  }

  // Erase the custom op definitions mapping. This allows custom op definitions
  // to be DCE'd, as now all information is stored in the dense resource
  // attribute attached to the main module.
  impls.erase();
}

std::unique_ptr<mlir::Pass>
KGEN::createRegisterCustomOps(CompileCanonicalizationFnsFn compilePatternsFn) {
  return std::make_unique<RegisterCustomOpsPass>(std::move(compilePatternsFn));
}
