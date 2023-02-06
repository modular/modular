//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_LIFTMLIROPERATIONS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LiftMLIROperationsPass
    : impl::LiftMLIROperationsBase<LiftMLIROperationsPass> {
  using LiftMLIROperationsBase::LiftMLIROperationsBase;

  void runOnOperation() override;
};
} // namespace

static SymbolConstantAttr materializeOperation(Location loc,
                                               SymbolTable &symtab,
                                               unsigned &counter,
                                               MLIROpAttr opExpr) {
  ImplicitLocOpBuilder b(loc, OpBuilder(loc.getContext()));
  StringAttr name =
      b.getStringAttr(opExpr.getName().getValue() + Twine(counter++));
  auto gen = b.create<GeneratorOp>(name, opExpr.getType());
  Block *body = b.createBlock(&gen.getBodyRegion());
  SmallVector<Value> operands;
  for (Type type : opExpr.getType().getValueInputs())
    operands.push_back(body->addArgument(type, loc));

  // "Bind" any remaining parameters by making them parameter reference
  // attributes.
  OperationState state(loc, opExpr.getName(), operands,
                       opExpr.getType().getValueResults(),
                       opExpr.getAttrs().getValue());
  for (ParamDeclAttr param : opExpr.getType().getInputParams())
    state.attributes.set(param.getName(), ParamDeclRefAttr::get(param));

  Operation *op = b.create(state);
  b.create<ReturnOp>(ArrayRef<TypedAttr>(), op->getResults());

  StringAttr symName =
      symtab.insert(gen, cast<ModuleOp>(symtab.getOp()).getBody()->begin());

  return SymbolConstantAttr::get(FlatSymbolRefAttr::get(symName),
                                 opExpr.getType());
}

void LiftMLIROperationsPass::runOnOperation() {
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  // Deduplicate identical operations.
  DenseMap<MLIROpAttr, SymbolConstantAttr> opExprMap;
  // Reduce the odds of name collision by incrementing a counter.
  unsigned counter = 0;

  // Collect all operations to materialize.
  getOperation()->walk([&](Operation *op) {
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](MLIROpAttr opExpr) -> Attribute {
      SymbolConstantAttr &symbol = opExprMap[opExpr];
      if (!symbol)
        symbol = materializeOperation(op->getLoc(), symtab, counter, opExpr);
      return symbol;
    });
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true, /*replaceLocs=*/false,
                               /*replaceTypes=*/true);
  });
}
