//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// lowerLexicalTerminators
//===----------------------------------------------------------------------===//

/// Lower all lexical terminators in the function and remove dead code.
static LogicalResult lowerLexicalTerminators(DeclRefType errType,
                                             LIT::FuncOp func) {
  if (func.getBodyRegion().empty())
    return success();
  if (func.isThrows() && !errType)
    return func.emitError("function throws but no 'Error' type was found");

  // Collect all the terminators first to avoid iterator invalidation.
  SmallVector<Operation *> terminators;
  func.walk([&](Operation *op) {
    if (isa<LIT::ReturnOp, LIT::RaiseOp, LIT::BreakOp, LIT::ContinueOp>(op))
      terminators.push_back(op);
  });

  // Lower all the terminators as they are encountered.
  auto getErrorOr = [&] {
    return POP::VariantType::get({errType, func.getResultType()});
  };
  LIT::ReturnOp firstResultParamsReturn;
  ParameterExprArrayAttr resultParams;
  SmallVector<Block *> deadBlocks;
  for (Operation *op : terminators) {
    // Ignore dead operations.
    if (op->getBlock() != &op->getParentRegion()->front())
      continue;

    ImplicitLocOpBuilder b(op->getLoc(), OpBuilder(op));
    auto createReturn = [&](ArrayRef<TypedAttr> params, ValueRange operands) {
      if (op->getParentOp() == func)
        b.create<KGEN::ReturnOp>(params, operands);
      else
        b.create<HLCF::ReturnOp>(operands);
    };
    if (auto returnOp = dyn_cast<LIT::ReturnOp>(op)) {
      if (resultParams && returnOp.getParametersAttr() != resultParams) {
        return returnOp
                   .emitError("function return defines different result "
                              "meta-parameters than previous return statement")
                   .attachNote(firstResultParamsReturn.getLoc())
               << "see conflicting result meta-parameters here";
      }
      firstResultParamsReturn = returnOp;
      resultParams = returnOp.getParametersAttr();

      ValueRange operands = returnOp.getOperands();
      Value result;
      if (func.isThrows()) {
        result = b.create<POP::VariantCreateOp>(getErrorOr(), operands.front());
        operands = result;
      }
      createReturn(resultParams, operands);

    } else if (auto raiseOp = dyn_cast<LIT::RaiseOp>(op)) {
      auto tryOp = raiseOp->getParentOfType<LIT::TryOp>();
      if (tryOp &&
          tryOp.getTryRegion().isAncestor(raiseOp->getBlock()->getParent())) {
        b.create<LIT::TryRaiseOp>(raiseOp.getError());
      } else {
        // TODO(#6449): Can't have result parameters in a function that raises.
        Value err =
            b.create<POP::VariantCreateOp>(getErrorOr(), raiseOp.getError());
        createReturn({}, err);
      }

    } else if (auto breakOp = dyn_cast<LIT::BreakOp>(op)) {
      b.create<HLCF::BreakOp>();
    } else {
      assert(isa<LIT::ContinueOp>(op) && "unknown terminator");
      b.create<HLCF::ContinueOp>();
    }

    // Check and warn about dead code.
    if (!op->getNextNode()->hasTrait<OpTrait::IsTerminator>())
      op->getNextNode()->emitWarning("unreachable code after ")
          << op->getName().stripDialect() << " statement";

    // Mark all subsequent operations as dead.
    deadBlocks.push_back(op->getBlock()->splitBlock(op));
  }

  // Remove all dead code.
  for (Block *block : deadBlocks)
    block->erase();

  // Check if the function lacks a top-level terminator. If the function
  // nominally returns `!lit.none`, then insert one. Otherwise, emit an error.
  Operation *terminator = func.getBody()->getTerminator();
  if (!isa<LIT::EndFuncOp>(terminator))
    return success();
  if (func.getNumResults() != 1 || !isa<LIT::NoneType>(func.getResultType()) ||
      !func.getResultParamTypes().empty())
    return terminator->emitError(
        "return expected at end of function with results");

  ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(terminator));
  Value none = b.create<ParamConstantOp>(b.getAttr<LIT::NoneAttr>());
  if (func.isThrows())
    none = b.create<POP::VariantCreateOp>(getErrorOr(), none);
  b.create<KGEN::ReturnOp>(ArrayRef<TypedAttr>(), none);
  terminator->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// lowerThrows
//===----------------------------------------------------------------------===//

/// Lower `<...>(...) throws -> T` to `<...>(...) -> ErrorOr<T>`. Update all
/// callsites to signature types to reflect this change.
static LogicalResult lowerThrows(DeclRefType errType, Operation *op) {
  // Find every throwing call.
  SmallVector<KGENCallOpInterface> throwingCalls;
  op->walk([&](KGENCallOpInterface call) {
    if (isa<GeneratorInterfaceOp>(*call))
      return;
    if (cast<SignatureType>(call.getCallee().getType()).isThrows())
      throwingCalls.push_back(call);
  });

  // Replace every throwing signature type with a variant.
  OpBuilder b(op);
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](SignatureType sigType) {
    if (!sigType.isThrows())
      return sigType;
    // Erase the `throws` bit and alter the result type to be wrapped.
    return SignatureType::get(
        sigType.getInputParams(), sigType.getResultParamTypes(),
        b.getFunctionType(sigType.getValueInputs(),
                          POP::VariantType::get(
                              {errType, sigType.getValueResults().front()})),
        b.getAttr<ConventionsAttr>(
            sigType.getValueInputConventions(),
            bitEnumClear(sigType.getFnEffects(), FnEffects::Throws)));
  });
  replacer.recursivelyReplaceElementsIn(
      op, /*replaceAttrs=*/true, /*replaceLocs=*/false, /*replaceTypes=*/true);

  // Update all the callsites.
  for (KGENCallOpInterface call : throwingCalls) {
    // FIXME: `kgen.addressof` returns a `FunctionType` (function pointer),
    // which is an important feature to have, but that means we can't globally
    // see calls to a throwing function pointer.
    if (isa<AddressOfOp>(call))
      return call.emitError("FIXME: cannot take address of throwing function");

    // We need to update the result types of the function, which means
    // rebuilding the operation.
    Type resultType = call->getResultTypes().front();
    OperationState state(call.getLoc(), call->getName(), call->getOperands(),
                         POP::VariantType::get({errType, resultType}));
    // Micro-optimization: skip a re-hash by assigning the dictionary attribute.
    state.attributes = call->getAttrDictionary();
    b.setInsertionPoint(call);
    auto newCall = cast<KGENCallOpInterface>(b.create(state));
    Value value = b.create<LIT::UnwrapOrPropagateOp>(call.getLoc(), resultType,
                                                     newCall->getResult(0));
    call->replaceAllUsesWith(ArrayRef(value));
    call.erase();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERLITTERMINATORS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerTerminatorsPass
    : impl::LowerLITTerminatorsBase<LowerTerminatorsPass> {
  using LowerLITTerminatorsBase::LowerLITTerminatorsBase;

  void runOnOperation() override {
    // Look for an error type declaration.
    DeclRefType errType;
    getOperation()->walk([&](StructDeclOp decl) {
      if (decl.getName() != "Error" || !decl.getInputParamDecls().empty())
        return;
      // Reconstruct the full symbol reference.
      errType = DeclRefType::get(
          LIT::getFullyResolvedSymbolRef(cast<mlir::SymbolOpInterface>(*decl)));
    });
    // Walk all top-level functions.
    WalkResult result =
        getOperation()->walk<mlir::WalkOrder::PreOrder>([&](LIT::FuncOp func) {
          if (failed(lowerLexicalTerminators(errType, func)))
            return WalkResult::interrupt();
          return WalkResult::skip();
        });
    if (result.wasInterrupted())
      return signalPassFailure();

    // Lower all functions that throw.
    if (!errType)
      return;
    if (failed(lowerThrows(errType, getOperation())))
      return signalPassFailure();
  }
};
} // namespace
