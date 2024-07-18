//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/CAPI/IR.h"
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

  RegisterCustomOpsPass(
      CompileCanonicalizationFnFn compileCanonicalizationFnFn = {})
      : target{},
        compileCanonicalizationFnFn(std::move(compileCanonicalizationFnFn)) {};
  LogicalResult initialize(MLIRContext *ctx) override;

  /// Collect all kgen.generators that are used as canonicalization functions,
  /// with their associated op name.
  /// In case of a failure, returns an empty DenseMap.
  LogicalResult collectCanonicalizationGenerators(
      DenseMap<StringAttr, GeneratorOp> &canonGens);

  void runOnOperation() override;

private:
  /// The compilation target.
  TargetInfoAttr target;

  /// The function to compile the canonicalization patterns out of a
  /// pre-elaboration module.
  CompileCanonicalizationFnFn compileCanonicalizationFnFn;
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

LogicalResult RegisterCustomOpsPass::collectCanonicalizationGenerators(
    DenseMap<StringAttr, GeneratorOp> &canonGens) {
  ModuleOp theModule = getOperation();
  auto implsOp = CustomOpImplsOp::lookupOp(theModule);

  // Exit early if no custom operations were defined.
  if (!implsOp)
    return success();

  auto &symtabAnalysis = getAnalysis<mlir::SymbolTableAnalysis>();
  SymbolTableCollection &tables = symtabAnalysis.getSymbolTables();

  for (CustomOpImplAttr impl : implsOp.getImpls()) {
    SymbolConstantAttr canonSym = impl.getOpCanonicalization();
    if (!canonSym)
      continue;
    auto canonicalizeOp = tables.lookupSymbolIn<GeneratorOp>(
        theModule.getOperation(), canonSym.getSymbol());
    if (!canonicalizeOp) {
      return implsOp->emitOpError()
             << "symbol does not refer to an existing operation: " << canonSym;
    }
    canonGens.try_emplace(impl.getOpName(), canonicalizeOp);
  }

  return success();
}

void RegisterCustomOpsPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  MLIRContext *ctx = theModule->getContext();

  // Collect all the canonicalization patterns.
  DenseMap<StringAttr, GeneratorOp> canonGens;
  if (failed(collectCanonicalizationGenerators(canonGens))) {
    signalPassFailure();
    return;
  }

  // No canonicalization patterns here, or a failure somewhere, so we exit
  // early.
  if (canonGens.empty())
    return;

  // Create the list of symbols we want to JIT.
  ExportMap exportMap;
  for (auto [opName, generatorOp] : canonGens)
    exportMap.try_emplace(generatorOp.getSymNameAttr(), ExportKind::Exported);

  auto &symtabAnalysis = getAnalysis<mlir::SymbolTableAnalysis>();
  SymbolTableCollection &tables = symtabAnalysis.getSymbolTables();

  ErrorOr<DenseMap<StringAttr, CAPICanonicalizationFn>> funcsOrErr =
      compileCanonicalizationFnFn(
          theModule, tables.getSymbolTable(theModule.getOperation()), exportMap,
          target);
  if (failed(funcsOrErr)) {
    mlir::emitError(theModule.getLoc())
        << "Error while compiling the custom canonicalization patterns: "
        << funcsOrErr.takeError() << "\n";
    return;
  }
  DenseMap<StringAttr, CAPICanonicalizationFn> funcs = funcsOrErr.takeValue();

  // Register the canonicalization patterns in the custom dialect.
  CustomDialect *customDialect = ctx->getLoadedDialect<CustomDialect>();
  for (auto [opName, generatorOp] : canonGens) {
    auto func = funcs.at(generatorOp.getSymNameAttr());
    auto canonFunc = [func = func](Operation *op,
                                   PatternRewriter &rewriter) mutable {
      // FIXME: Only call the canonicalization function for now without doing
      // anything. This is for testing purposes, and future commits will fix
      // this.
      func(wrap(op));
      return success();
    };
    customDialect->canonicalizationFns.try_emplace(opName,
                                                   std::move(canonFunc));
  }
}

std::unique_ptr<mlir::Pass> KGEN::createRegisterCustomOps(
    CompileCanonicalizationFnFn compileCanonicalizationFnFn) {
  return std::make_unique<RegisterCustomOpsPass>(
      std::move(compileCanonicalizationFnFn));
}
