//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/COOps.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "mlir/IR/PatternMatch.h"
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

/// Lower a concrete pack to a struct.
static StructType lowerPackTypeToStruct(PackType pack) {
  ArrayRef<TypedAttr> typeExprs = pack.getVariadicIfResolved().getValues();
  SmallVector<Type> elementTypes;
  elementTypes.reserve(typeExprs.size());
  for (TypedAttr typeExpr : typeExprs)
    elementTypes.push_back(cast<TypeConstantAttr>(typeExpr).getMlirType());
  return StructType::get(pack.getContext(), elementTypes);
}

/// Lower a concrete pack attribute to a struct attribute.
static StructAttr lowerPackAttrToStruct(PackAttr pack) {
  StructType structType = lowerPackTypeToStruct(pack.getType());
  return StructAttr::get(pack.getValues(), structType);
}

/// We need to roll our own walk function because we are converting types and
/// operations at the same time. We need a pre-order walk to convert argument
/// types before their users, but we are also erasing ops with regions.
static void recursiveRewrite(Operation *op, mlir::AttrTypeReplacer &replacer);

/// Rewrite a single operation, recursing if it has regions.
static void rewriteFn(Operation *op, mlir::AttrTypeReplacer &replacer) {
  // Recursively replace all signatures in the operation. This will handle the
  // signatures of `kgen.func`, `kgen.stage_closure`, and `co.execute`.
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

  // Handle `hlcf.if`, `hlcf.loop`, `kgen.call`, `kgen.call_indirect`, and
  // `co.get_results`.
  if (isa<HLCF::IfOp, HLCF::LoopOp, CallOp, CallIndirectOp, CO::GetResultsOp,
          CO::AwaitOp>(op)) {
    auto [anyNone, newResults] = removeNoneTypes(op->getResultTypes());
    // Exit early if there are no none results.
    if (!anyNone)
      return recursiveRewrite(op, replacer);
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
    // Recurse on the new op.
    recursiveRewrite(newOp, replacer);
    return;
  }

  // Handle pack operations. Be mindful here of the ODS methods because the
  // operand and result types will have already been lowered to `!kgen.struct`.
  mlir::IRRewriter b{OpBuilder(op)};
  if (auto create = dyn_cast<PackCreateOp>(op)) {
    b.replaceOpWithNewOp<StructCreateOp>(create, create->getResultTypes(),
                                         create.getElements());
    return;
  }
  if (auto extract = dyn_cast<PackExtractOp>(op)) {
    b.replaceOpWithNewOp<StructExtractOp>(
        extract, extract->getOperand(0), cast<IntegerAttr>(extract.getIndex()));
    return;
  }
  if (auto gep = dyn_cast<PackGEPOp>(op)) {
    b.replaceOpWithNewOp<StructGEPOp>(gep, gep.getType(), gep->getOperand(0),
                                      cast<IntegerAttr>(gep.getIndex()));
    return;
  }
  if (auto size = dyn_cast<PackSizeOp>(op)) {
    ArrayRef<Type> types =
        cast<StructType>(size->getOperand(0).getType()).getElementTypes();
    b.replaceOpWithNewOp<ParamConstantOp>(size, b.getIndexAttr(types.size()));
    return;
  }
  if (auto load = dyn_cast<PackLoadOp>(op)) {
    SmallVector<Value> elements;
    ArrayRef<Type> types =
        cast<StructType>(load->getOperand(0).getType()).getElementTypes();
    elements.reserve(types.size());
    for (auto [i, _] : llvm::enumerate(types)) {
      auto ptr =
          b.create<StructExtractOp>(op->getLoc(), load->getOperand(0), i);
      elements.push_back(b.create<POP::LoadOp>(op->getLoc(), ptr));
    }
    b.replaceOpWithNewOp<StructCreateOp>(load, load->getResultTypes(),
                                         elements);
    return;
  }

  // In the general case, try to recurse on the op.
  recursiveRewrite(op, replacer);
}

/// Out-of-line definition.
static void recursiveRewrite(Operation *op, mlir::AttrTypeReplacer &replacer) {
  for (Region &region : op->getRegions())
    for (Operation &op : llvm::make_early_inc_range(region.front()))
      rewriteFn(&op, replacer);
}

void LowerCallingConventionsPass::runOnOperation() {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(lowerResult);
  replacer.addReplacement(removeDINoneResults);
  replacer.addReplacement(lowerPackTypeToStruct);
  replacer.addReplacement(lowerPackAttrToStruct);

  rewriteFn(getOperation(), replacer);
}
