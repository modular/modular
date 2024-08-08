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
#include "KGEN/TransformUtils/ManglingUtils.h"
#include "KGEN/TransformUtils/SlicingUtils.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace M;
using namespace KGEN;
using namespace Custom;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERCUSTOMOPS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerCustomOpsPass : impl::LowerCustomOpsBase<LowerCustomOpsPass> {
  LowerCustomOpsPass(RunKGENPipelineFn runKGENPipelineFn = {})
      : runKGENPipelineFn(std::move(runKGENPipelineFn)) {};

  void runOnOperation() override;

  /// Generate a new generator that will call a given generator with the given
  /// parameters. Return the new generator symbol.
  static SymbolConstantAttr
  specializeParametrizedGenerator(SymbolConstantAttr generatorSym,
                                  ArrayRef<TypedAttr> parameters, Location loc,
                                  SymbolTable &opImplsTable);

  /// Get a symbol to an op implementation specialized with the given
  /// parameters. For op that do not have parameters, this will simply return
  /// the symbol of the op implementation. For ops with parameters, this will
  /// create a new specialized operation in the SymbolTable using
  /// `specializeParametrizedGenerator`.
  static SymbolConstantAttr getSymbolForCustomOp(SymbolConstantAttr opImplSym,
                                                 Attribute implParamsAttr,
                                                 SymbolTable &opImplsTable,
                                                 Location loc);

  /// Convert a custom op into a call to its implementation.
  /// Implementations are cached in `paramOpsToSym`, and may create new
  /// `kgen.generators` operations for parameter specialization in the provided
  /// symbol table.
  static LogicalResult
  convertCustomOp(Operation *op, CustomOpImplArrayAttr opImpls,
                  DenseMap<std::pair<StringAttr, Attribute>, SymbolConstantAttr>
                      &paramOpsToSym,
                  SymbolTable &opImplsTable);

private:
  RunKGENPipelineFn runKGENPipelineFn;
};
} // namespace

SymbolConstantAttr LowerCustomOpsPass::specializeParametrizedGenerator(
    SymbolConstantAttr generatorSym, ArrayRef<TypedAttr> parameters,
    Location loc, SymbolTable &opImplsTable) {
  MLIRContext *ctx = generatorSym.getContext();
  // Get the specialized function symbol and signature.
  SignatureType specializedSig =
      generatorSym.getType().getSpecializedSignature(parameters, loc);
  auto specializedSymbol = SymbolConstantAttr::get(generatorSym.getSymbol(),
                                                   parameters, specializedSig);

  // Create a new name based on the original name and parameters.
  auto generator = opImplsTable.lookup<GeneratorOp>(
      generatorSym.getSymbol().getLeafReference());
  std::string newName = mangleParameterValues(generator, parameters);

  auto newNameAttr = StringAttr::get(ctx, newName);

  // Create the wrapper generator.
  OpBuilder builder(ctx);
  auto specializedFunc =
      builder.create<GeneratorOp>(loc, newNameAttr, specializedSig);
  StringAttr specializedFuncSymbol = opImplsTable.insert(specializedFunc);
  specializedFunc.setInlineLevel(InlineLevel::Always);
  specializedFunc.setExportKind(ExportKind::Exported);
  auto argLocs = std::vector<Location>(specializedSig.getNumArguments(), loc);
  auto funcBlock = builder.createBlock(&specializedFunc->getRegion(0), {},
                                       specializedSig.getArguments(), argLocs);

  // Create a call inside the specialized generator that will call the original
  // generator with the given parameters.
  builder.setInsertionPointToStart(funcBlock);
  auto specializedCall =
      builder.create<CallOp>(loc, specializedSig.getResults(),
                             specializedSymbol, specializedFunc.getArguments());
  builder.create<ReturnOp>(loc, specializedCall.getResults());

  // Finally, get the symbol of the specialized generator.
  return SymbolConstantAttr::get(SymbolRefAttr::get(specializedFuncSymbol), {},
                                 specializedSig);
}

SymbolConstantAttr LowerCustomOpsPass::getSymbolForCustomOp(
    SymbolConstantAttr opImplSym, Attribute implParamsAttr,
    SymbolTable &opImplsTable, Location loc) {
  // If the operation doesn't specify parameters, this means the generator
  // doesn't have any parameters, so we can return the generator symbol.
  if (!implParamsAttr) {
    auto implOp = opImplsTable.lookup<ExportInterface>(
        opImplSym.getSymbol().getLeafReference());
    implOp.setExported();
    return opImplSym;
  }

  // FIXME(math-fehr): Support multiple parameters
  SmallVector<TypedAttr> parameters;
  parameters.push_back(
      cast<TypedAttr>(cast<PreservedAttr>(implParamsAttr).getValue()));

  return specializeParametrizedGenerator(opImplSym, parameters, loc,
                                         opImplsTable);
}

LogicalResult LowerCustomOpsPass::convertCustomOp(
    Operation *op, CustomOpImplArrayAttr opImpls,
    DenseMap<std::pair<StringAttr, Attribute>, SymbolConstantAttr>
        &paramOpsToSym,
    SymbolTable &opImplsTable) {
  // Get the op name and the op implementation parameters.
  StringAttr opName = op->getName().getIdentifier();
  Attribute implParamsAttr = op->getAttr(kCustomOpParamsAttrName);

  // Get the symbol to the op implementation generator.
  CustomOpImplAttr opImplAttr = opImpls.getOpImpl(opName);
  if (!opImplAttr) {
    op->emitError() << "no implementation found for custom op '"
                    << opName.strref() << "'";
    return failure();
  }

  // Get a symbol to the generator specialized with the parameters
  SymbolConstantAttr specializedOpImplSym;
  auto opSymbolIt = paramOpsToSym.find({opName, implParamsAttr});
  if (opSymbolIt != paramOpsToSym.end()) {
    specializedOpImplSym = opSymbolIt->getSecond();
  } else {
    SymbolConstantAttr opImplSym =
        opImpls.getOpImpl(opName).getOpImplementation();
    specializedOpImplSym = getSymbolForCustomOp(opImplSym, implParamsAttr,
                                                opImplsTable, op->getLoc());
    paramOpsToSym.try_emplace({opName, implParamsAttr}, specializedOpImplSym);
  }

  // Replace the custom op with a call to its implementation.
  OpBuilder builder(op);
  auto callOp = builder.create<CallOp>(op->getLoc(), op->getResultTypes(),
                                       specializedOpImplSym, op->getOperands());
  op->replaceAllUsesWith(callOp->getResults());
  op->erase();
  return success();
}

void LowerCustomOpsPass::runOnOperation() {
  ModuleOp theModule = getOperation();

  // Get the op definitions attached to the module
  auto opDefsResource = cast_or_null<DenseResourceElementsAttr>(
      theModule->getAttr(kCustomOpImplModuleAttr));

  // We don't have anything to do here, there should be no custom ops
  if (!opDefsResource)
    return;

  auto opImplsModule = readOpFromBytecodeFile<ModuleOp>(opDefsResource);
  auto opImplsTable = SymbolTable(*opImplsModule);

  Dialect *customDialect = getContext().getLoadedDialect<CustomDialect>();

  // Get the op implementation map that is stored in the module.
  auto opImplsOp = CustomOpImplsOp::lookupOp(*opImplsModule);
  if (!opImplsOp) {
    theModule->emitError() << "no '" << CustomOpImplsOp::getOperationName()
                           << "' op found at the toplevel module";
    signalPassFailure();
    return;
  }
  CustomOpImplArrayAttr opImpls = opImplsOp.getImplsAttr();

  // The mapping of custom op and parameters to generator symbols.
  // These symbols are used to replace the custom ops with a function call.
  DenseMap<std::pair<StringAttr, Attribute>, SymbolConstantAttr> paramOpsToSym;

  // Replace all custom ops with a call to their (potentially specialized)
  // implementation.
  theModule->walk(
      [&opImpls, &paramOpsToSym, &opImplsTable, customDialect](Operation *op) {
        // Only convert operations from the custom dialect.
        if (op->getDialect() != customDialect)
          return WalkResult::advance();

        if (failed(convertCustomOp(op, opImpls, paramOpsToSym, opImplsTable)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });

  // Leave early if no custom operation exist in the IR.
  if (paramOpsToSym.empty()) {
    theModule->removeAttr(kCustomOpImplModuleAttr);
    return;
  }

  // Run the KGEN pipeline on the op implementations.
  ErrorOrSuccess pipelineSucceeded =
      runKGENPipelineFn(*opImplsModule, getTargetInfo(theModule));

  if (pipelineSucceeded.isError()) {
    opImplsModule->emitError() << "error while runing the KGEN pipeline when "
                                  "lowering custom operations: "
                               << pipelineSucceeded.takeError();
    signalPassFailure();
    return;
  }

  // Recompute the symbol table for the op implementations, as it was
  // invalidated by the KGEN pipeline.
  opImplsTable = SymbolTable(*opImplsModule);

  // Get the symbol table for both the current module.
  auto &symtabAnalysis = getAnalysis<mlir::SymbolTableAnalysis>();
  SymbolTable table = symtabAnalysis.getTopLevelSymbolTable();

  // Move all new operations to the original module. Operations with the same
  // names are functionally equivalent, so they are not compiled.
  auto rewriter = mlir::IRRewriter(&getContext());
  for (auto toplevelOp : llvm::make_early_inc_range(
           opImplsModule->getOps<mlir::SymbolOpInterface>())) {
    if (table.lookup(toplevelOp.getNameAttr()))
      continue;
    rewriter.moveOpAfter(toplevelOp, theModule.getBody(),
                         theModule.getBody()->begin());
  }

  // Finally, remove the op implementation module from our module, as all custom
  // ops are now lowered.
  theModule->removeAttr(kCustomOpImplModuleAttr);
}

std::unique_ptr<Pass>
KGEN::createLowerCustomOps(RunKGENPipelineFn runKGENPipelineFn) {
  return std::make_unique<LowerCustomOpsPass>(std::move(runKGENPipelineFn));
}
