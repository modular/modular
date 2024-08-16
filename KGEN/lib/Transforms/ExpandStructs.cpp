//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/COOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/TransformUtils/StructUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_EXPANDSTRUCTS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class ExpandStructsPass : public impl::ExpandStructsBase<ExpandStructsPass> {
public:
  using ExpandStructsBase::ExpandStructsBase;
  void runOnOperation() override;

private:
  static void recursiveRewrite(Operation *op, mlir::AttrTypeReplacer &replacer);
  static void rewriteFn(Operation *op, mlir::AttrTypeReplacer &replacer);
};

/// Helper enum to indicate whether we need to expand in the operands or results
/// of an op, based on whether they flow inputs to a callee or outputs from one.
enum CallFlow { None, In, Out, Body };
} // namespace

/// Get the flow kind of the operation. Direct calls flow both in and out,
/// whereas operations that form closures only flow in. Operations that
/// semantically invoke closures flow out only.
static int getFlowKind(Operation *op) {
  int result = CallFlow::None;

  if (isa<HLCF::ControlFlowTerminator, HLCF::ControlFlowNode, CallOp,
          CallIndirectOp, CreateClosureOp, POP::ExternalCallOp, CO::InvokeOp,
          CO::HotInvokeOp>(op))
    result |= CallFlow::In;
  if (isa<HLCF::ControlFlowNode, CallOp, CallIndirectOp, POP::ExternalCallOp,
          CO::GetResultsOp, CO::AwaitOp>(op))
    result |= CallFlow::Out;
  if (isa<HLCF::ControlFlowNode, FuncOp>(op))
    result |= CallFlow::Body;

  return result;
}

void ExpandStructsPass::recursiveRewrite(Operation *op,
                                         mlir::AttrTypeReplacer &replacer) {
  for (Region &region : op->getRegions())
    for (Operation &op : llvm::make_early_inc_range(region.front()))
      rewriteFn(&op, replacer);
}

void ExpandStructsPass::rewriteFn(Operation *op,
                                  mlir::AttrTypeReplacer &replacer) {
  replacer.replaceElementsIn(op, /*replaceAttrs=*/true, /*replaceLocs=*/true,
                             /*replaceTypes=*/true);

  int kind = getFlowKind(op);
  if (kind == CallFlow::None)
    return recursiveRewrite(op, replacer);

  mlir::IRRewriter b{OpBuilder(op)};
  if (kind & CallFlow::In)
    flattenStructsInOperands(b, op);

  if (kind & CallFlow::Body) {
    for (Region &region : op->getRegions()) {
      Block *body = &region.front();
      b.setInsertionPointToStart(body);
      flattenStructsInArguments(b, op->getLoc(), body);
    }
  }

  if (kind & CallFlow::Out && !isa<FuncOp>(op)) {
    b.setInsertionPointAfter(op);
    Operation *newOp = flattenStructsInResults(b, op);
    // The op was deleted, so we need to skip over it.
    return recursiveRewrite(newOp, replacer);
  }

  return recursiveRewrite(op, replacer);
}

void ExpandStructsPass::runOnOperation() {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([](SignatureType sig) {
    SmallVector<Type> newArgTypes, newResTypes;
    SmallVector<ArgConvention> newConvs;

    for (Type type : sig.getResults())
      flattenTypeIfStruct(type, newResTypes);
    for (auto [type, conv] :
         llvm::zip(sig.getArguments(), sig.getArgConventions())) {
      unsigned curSize = newArgTypes.size();
      flattenTypeIfStruct(type, newArgTypes);
      newConvs.append(newArgTypes.size() - curSize, conv);
    }

    auto func = FunctionType::get(sig.getContext(), newArgTypes, newResTypes);
    return SignatureType::get(func, newConvs, sig.getFnEffects());
  });

  rewriteFn(getOperation(), replacer);
}
