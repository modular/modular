//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/COOps.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERCALLINGCONVENTIONS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerCallingConventionsPass
    : KGEN::impl::LowerCallingConventionsBase<LowerCallingConventionsPass> {
  void runOnOperation() override;
};
} // namespace

/// Filters out none types from the given range, and returns a flag indicating
/// if there were any in the input.
static std::pair<bool, SmallVector<Type>> removeNoneTypes(TypeRange types) {
  SmallVector<Type> newTypes;
  for (Type type : types)
    if (!isa<KGEN::NoneType>(type))
      newTypes.push_back(type);
  return {newTypes.size() != types.size(), std::move(newTypes)};
}

/// Lower the signature results by replacing all the `!kgen.none` results in a
/// signature. This will also set the signature to non-throwing, and erase
/// `byref_result` argument conventions.
static SignatureType lowerResult(SignatureType signature) {
  auto [anyNone, newResults] = removeNoneTypes(signature.getResults());
  // Micro-optimization: don't hash a new type if it won't change.
  if (!anyNone)
    return signature;
  SmallVector<ArgConvention> maybeNewConventions;
  ArrayRef<ArgConvention> conventions = signature.getArgConventions();
  if (!conventions.empty() && conventions[0] == ArgConvention::ByRefResult) {
    llvm::append_range(maybeNewConventions, conventions);
    maybeNewConventions[0] = ArgConvention::None;
    conventions = maybeNewConventions;
  }
  return SignatureType::get(
      FunctionType::get(signature.getContext(), signature.getArguments(),
                        newResults),
      conventions, signature.getFnEffects().setThrows(false));
}

/// Remove `none` results from coroutine types.
static CO::CoroutineType removeNoneCoroutine(CO::CoroutineType coro) {
  auto [anyNone, newTypes] = removeNoneTypes(coro.getTypes());
  if (!anyNone)
    return coro;
  return CO::CoroutineType::get(coro.getContext(), newTypes);
}

/// Remove none types from the results of debuginfo subroutine types as well.
static DebugInfo::DISubroutineType
removeDINoneResults(DebugInfo::DISubroutineType type) {
  // None types in the subroutine type will be wrapped in an unresolved type.
  SmallVector<DebugInfo::DIType> newTypes;
  for (DebugInfo::DIType type : type.getResultTypes()) {
    auto unresolved = dyn_cast<DebugInfo::DIUnresolvedMLIRType>(type);
    if (!unresolved || !isa<KGEN::NoneType>(unresolved.getType()))
      newTypes.push_back(type);
  }
  if (newTypes.size() == type.getResultTypes().size())
    return type;
  return DebugInfo::DISubroutineType::get(type.getContext(),
                                          type.getArgumentTypes(), newTypes);
}

void LowerCallingConventionsPass::runOnOperation() {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(lowerResult);
  replacer.addReplacement(removeNoneCoroutine);
  replacer.addReplacement(removeDINoneResults);

  auto walkFn = [&](Operation *op) {
    // Recursively replace all signatures in the operation. This will handle the
    // signatures of `kgen.func`, `kgen.stage_closure`, and `lit.async.execute`.
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true, /*replaceLocs=*/true,
                               /*replaceAttrs=*/true);

    // Handle exiting terminators.
    if (isa<ReturnOp, HLCF::YieldOp, HLCF::BreakOp>(op)) {
      // Remove none results.
      SmallVector<Value> newOperands;
      for (Value operand : op->getOperands())
        if (!isa<KGEN::NoneType>(operand.getType()))
          newOperands.push_back(operand);
      op->setOperands(newOperands);
      return;
    }

    // Handle `hlcf.if`, `hlcf.loop`, `kgen.call`, and `kgen.call_indirect`.
    if (isa<HLCF::IfOp, HLCF::LoopOp, CallOp, CallIndirectOp>(op)) {
      auto [anyNone, newResults] = removeNoneTypes(op->getResultTypes());
      // Exit early if there are no none results.
      if (!anyNone)
        return;
      OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                           newResults);
      // Micro-optimization: set the DictionaryAttr directly to avoid a re-hash.
      state.attributes = op->getAttrDictionary();
      for (Region &region : op->getRegions())
        state.addRegion()->takeBody(region);
      Operation *newOp = OpBuilder(op).create(state);

      // Lazily construct a none constant only when needed.
      Value noneImpl;
      auto getNone = [&] {
        if (!noneImpl) {
          noneImpl = OpBuilder(op).create<ParamConstantOp>(
              op->getLoc(), NoneAttr::get(op->getContext()));
        }
        return noneImpl;
      };

      unsigned newResultIdx = 0;
      for (Value result : op->getResults()) {
        if (!isa<KGEN::NoneType>(result.getType()))
          result.replaceAllUsesWith(newOp->getResult(newResultIdx++));
        else if (!result.use_empty())
          result.replaceAllUsesWith(getNone());
      }
      assert(newResultIdx == newOp->getNumResults());
      op->erase();
      return;
    }

    // Handle `co.promise`. None types take up no space, so it is
    // safe to bitcast the promise pointer.
    if (auto promise = dyn_cast<CO::CoroutinePromiseOp>(op)) {
      auto [anyNone, newResults] =
          removeNoneTypes(cast<StructType>(promise.getType().getElementType())
                              .getElementTypes());
      if (!anyNone)
        return;

      OpBuilder b(promise);
      Value newPromise = b.create<CO::CoroutinePromiseOp>(
          promise.getLoc(),
          PointerType::get(StructType::get(b.getContext(), newResults)),
          promise.getCoroutine());
      Value casted = b.create<POP::PointerBitcastOp>(
          promise.getLoc(), promise.getType(), newPromise);
      promise.replaceAllUsesWith(casted);
      promise.erase();
      return;
    }
  };
  getOperation().walk(walkFn);
}
