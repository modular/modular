//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// Legalizes operations on types that LLVM and POP->LLVM don't support.
//
// Some operations (e.g., FP8 arithmetic) have no direct LLVM codegen path and
// aren't emulated during POP → LLVM lowering. This pass handles them by:
//   1. Casting operands to a wider, supported type
//   2. Performing the operation in that type
//   3. Casting the result back to the original type
//
// Example 0: Negating an f8e5m2 value on NVPTX
//
//   Before (illegal):
//     %res = pop.neg %input : !pop.scalar<f8e5m2>
//
//   After (legal):
//     %0 = pop.cast %input : !pop.scalar<f8e5m2> to !pop.scalar<f16>
//     %1 = pop.neg %0 : !pop.scalar<f16>
//     %2 = pop.cast %1 : !pop.scalar<f16> to !pop.scalar<f32>
//     %res = pop.cast %2 : !pop.scalar<f32> to !pop.scalar<f8e5m2>
//
// For this example NVPTX target has no direct conversion from f16 to
// f8e5m2, but POP->LLVM implements f32->f8e5m2 conversion by using special PTX
// instructions and f16->f32 is natively supported by fptext, therefore multiple
// conversions are required to convert result back to f8e5m2.
//
// Example 1: Converting f8e5m2 to bf16 on NVPTX
//    Before (illegal):
//      %res = pop.cast %input : !pop.scalar<f8e5m2> to !pop.scalar<bf16>
//
//    After (legal):
//      %0 = pop.cast %input : !pop.scalar<f8e5m2> to !pop.scalar<f16>
//      %1 = pop.cast %0 : !pop.scalar<f16> to !pop.scalar<f32>
//      %res = pop.cast %1 : !pop.scalar<f32> to !pop.scalar<bf16>
//
// In this example, there's no direct conversion from f8e5m2 to bf16, but it can
// be done with a sequence of conversions f8e5m2->f16->f32->bf16
//
// Scope:
//   - Unary and binary ops on FP8 or narrower types
//   - CastOps with no direct target-supported conversion
//
// TODO:
//  - Support other than NVPTX targets
//  - Query legal conversions from `ConvertPOPCast`
//  - Revisit algorithm to find shortest/most cost effective sequence of
//    conversions
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "LLVMLoweringUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Attributes.h"

using namespace M;
using namespace KGEN;
using namespace POP;

namespace M::KGEN {
#define GEN_PASS_DEF_LEGALIZEPOPOPERATIONS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

/// Return KGENDType that corresponds to the scalar type of the given \p type.
static KGENDType getScalarKGENDType(Type type) {
  if (auto simd = dyn_cast<SIMDType>(type)) {
    std::optional<KGENDType> dtype = simd.getResolvedDType();
    assert(dtype && "dtype must be resolved at this point");
    return *dtype;
  }
  return KGENDType();
}

/// Return similar type to \p type, but with scalar type of \p dtype
static Type convertKGENDTypeToType(KGENDType dtype, Type type) {
  if (auto simd = dyn_cast<SIMDType>(type))
    return SIMDType::get(simd.getContext(), *simd.getResolvedSize(), dtype);
  // TODO: add support for other types
  llvm_unreachable("unsupported type");
}

namespace {

class LegalizePOPOperations
    : public KGEN::impl::LegalizePOPOperationsBase<LegalizePOPOperations> {
public:
  void runOnOperation() override;

private:
  // A map of types for which target has implemented conversions
  //              "target-arch" -> {input-type -> {output-types...}}
  // For example, "nvptx-sm_90" -> {f8e4m3fn   -> {f16, f32}}
  // TODO: Query this information from `ConvertPOPCast`
  DenseMap<StringRef, DenseMap<KGENDType, llvm::SetVector<KGENDType>>>
      targetLegalConversion;

  /// Initialize map of known conversions for a target
  void initializeTargetLegalConversions(MLIRContext *ctx);

  /// Return true if legalization of the operation succeeded.
  LogicalResult legalizeOperation(Operation *op, TargetInfoAttr target);

  /// Return 'success' if legalization has to be done for the operation. Return
  /// false otherwise. An operation requires legalization if lowering the
  /// operation to LLVM with converted types will generate invalid IR. For now
  /// legalize all arithmetic operations that operate on unsupported by LLVM
  /// types, such as F8 types.
  bool operationRequiresLegalization(Operation *op,
                                     TargetInfoAttr target) const;

  /// Return true if lowering of the conversion is supported
  bool conversionRequiresLegalization(CastOp castOp,
                                      TargetInfoAttr target) const;

  /// Return 'success' if legalization has to be done for the conversion.
  LogicalResult legalizeConversion(CastOp castOp, TargetInfoAttr target);

  /// Return true if the type requires legalization
  bool typeRequiresLegalization(KGENDType dtype) const;
  bool typeRequiresLegalization(Type type) const;

  /// Return supported type that can be used instead of \p type
  Type getSupportedType(Type type, TargetInfoAttr target) const;

  /// Helper function to find a sequence of available conversions from \p
  /// fromType to \p toType That is, this function returns a vector {type0,
  /// type1, ..., typeN} such that fromType -> type0 -> type1 -> ... -> typeN ->
  /// toType
  SmallVector<Type> findConversionSequence(Type fromType, Type toType,
                                           TargetInfoAttr target) const;

  /// Return target key that is used to get all legal conversion for the target.
  std::string getTargetKey(TargetInfoAttr target) const {
    // TODO: Support other targets as well
    if (!isNVPTX_HopperAndAbove(target))
      return "";
    return "nvptx-sm_90";
  }
};

//===----------------------------------------------------------------------===//
// initializeTargetLegalConversions
//===----------------------------------------------------------------------===//

void LegalizePOPOperations::initializeTargetLegalConversions(MLIRContext *ctx) {
  // List of conversion for NVPTX Hopper and above
  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f8e5m2)].insert(
      KGENDType(KGENDType::f16));

  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f8e4m3fn)].insert(
      KGENDType(KGENDType::f16));

  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f32)].insert(
      KGENDType(KGENDType::f8e5m2));
  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f32)].insert(
      KGENDType(KGENDType::f8e4m3fn));
  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f32)].insert(
      KGENDType(KGENDType::f16)); // supported by LLVM
  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f32)].insert(
      KGENDType(KGENDType::bf16)); // supported by LLVM

  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f16)].insert(
      KGENDType(KGENDType::f32)); // supported by LLVM
  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f16)].insert(
      KGENDType(KGENDType::f64)); // supported by LLVM

  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::bf16)].insert(
      KGENDType(KGENDType::f32)); // supported by LLVM
  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f16)].insert(
      KGENDType(KGENDType::f64)); // supported by LLVM
  // TODO: This list has to be complete for all supported types, but ideally it
  // should be taken from lowering of the POP::CastOp.
}

//===----------------------------------------------------------------------===//
// typeRequiresLegalization
//===----------------------------------------------------------------------===//

/// Return true if the type requires legalization
bool LegalizePOPOperations::typeRequiresLegalization(KGENDType dtype) const {
  if (dtype.isInvalid())
    return false;

  // Assume that any floating point type below 8 bits are not supported by
  // LLVM
  return dtype.isFloat() && dtype.getWidthInBits() <= 8;
}

/// Return true if the type requires legalization
bool LegalizePOPOperations::typeRequiresLegalization(Type type) const {
  return typeRequiresLegalization(getScalarKGENDType(type));
}

//===----------------------------------------------------------------------===//
// operationRequiresLegalization
//===----------------------------------------------------------------------===//

bool LegalizePOPOperations::operationRequiresLegalization(
    Operation *op, TargetInfoAttr target) const {
  // If operands have supported types, do not need to do anything extra with the
  // operation.
  for (auto operand : op->getOperands())
    if (!typeRequiresLegalization(operand.getType()))
      return false;

  return isa<NegOp, AddOp, SubOp, MulOp, DivOp, RemOp, MaxOp, MinOp>(op);
}

//===----------------------------------------------------------------------===//
// conversionRequiresLegalization
//===----------------------------------------------------------------------===//

bool LegalizePOPOperations::conversionRequiresLegalization(
    CastOp castOp, TargetInfoAttr target) const {
  if (!typeRequiresLegalization(castOp.getInput().getType()) &&
      !typeRequiresLegalization(castOp.getOutput().getType()))
    return false;

  KGENDType fromType = getScalarKGENDType(castOp.getInput().getType());
  KGENDType toType = getScalarKGENDType(castOp.getOutput().getType());

  auto targetConversionsIt = targetLegalConversion.find(getTargetKey(target));
  assert(targetConversionsIt != targetLegalConversion.end() &&
         "targetLegalConversion map was not initialized");

  // If there's no direct conversion available, then legalization is required
  auto typeConversionsIt = targetConversionsIt->second.find(fromType);
  if (typeConversionsIt == targetConversionsIt->second.end())
    return true;

  return !typeConversionsIt->second.contains(toType);
}

//===----------------------------------------------------------------------===//
// findConversionSequence
//===----------------------------------------------------------------------===//

SmallVector<Type>
LegalizePOPOperations::findConversionSequence(Type fromType, Type toType,
                                              TargetInfoAttr target) const {
  KGENDType fromDType = getScalarKGENDType(fromType);
  KGENDType toDType = getScalarKGENDType(toType);
  auto targetConversionsIt = targetLegalConversion.find(getTargetKey(target));
  assert(targetConversionsIt != targetLegalConversion.end() &&
         "target not found");
  SmallVector<Type> types;
  DenseSet<KGENDType> visited;
  // Recusively find a sequence of available conversions that can be used to
  // covert fromType to toType
  // TODO: Revisit algorithm if shortest/most cost effective sequence is
  // required.
  std::function<bool(KGENDType, KGENDType)> walker =
      [&walker, &types, &targetConversionsIt, &target, toType,
       &visited](KGENDType fromDType, KGENDType toDType) -> bool {
    // To avoid possible cycles, do not visit the same type twice
    if (!visited.insert(fromDType).second)
      return false;

    auto fromTypeIt = targetConversionsIt->second.find(fromDType);
    if (fromTypeIt == targetConversionsIt->second.end())
      return false;

    size_t fromTypeSize = fromDType.getWidthInBits(target);
    size_t toTypeSize = toDType.getWidthInBits(target);
    bool isUpconversion = fromTypeSize < toTypeSize;

    for (KGENDType commonDType : fromTypeIt->second) {
      // Do not try to select commonType if original conversion is:
      // - upconversion and common type is smaller or equal than the fromType
      // - downconversion and common type is smaller or equal than the toType
      // otherwise this will lead to precision losses.
      // TODO: Revisit "equal than" condition: it might be possible to use that
      // type if it won't introduce precision losses
      size_t commonTypeSize = commonDType.getWidthInBits(target);
      if ((commonTypeSize <= fromTypeSize && isUpconversion) ||
          (commonTypeSize <= toTypeSize && !isUpconversion))
        continue;

      auto commonTypeIt = targetConversionsIt->second.find(commonDType);
      if (commonTypeIt == targetConversionsIt->second.end())
        continue;

      // If direct conversion from commonType to toType exists, we can safely
      // use it, otherwise try to find a type to convert from commonType to
      // toType
      // TODO: Return the smallest available type
      if (commonTypeIt->second.contains(toDType) ||
          walker(commonDType, toDType)) {
        types.push_back(convertKGENDTypeToType(commonDType, toType));
        return true;
      }
    }
    return false;
  };
  (void)walker(fromDType, toDType);
  return SmallVector<Type>(llvm::reverse(types));
}

//===----------------------------------------------------------------------===//
// legalizeConversion
//===----------------------------------------------------------------------===//

LogicalResult LegalizePOPOperations::legalizeConversion(CastOp castOp,
                                                        TargetInfoAttr target) {
  assert(conversionRequiresLegalization(castOp, target) &&
         "legalization not required");
  ImplicitLocOpBuilder b(castOp.getLoc(), castOp);

  Type type = castOp.getType();

  SmallVector<Type> commonTypes =
      findConversionSequence(castOp.getInput().getType(), type, target);
  if (commonTypes.empty())
    return castOp->emitError("cannot legalize conversion ");

  Value newResult = castOp.getInput();
  for (Type commonType : commonTypes)
    newResult = CastOp::create(b, commonType, newResult);

  newResult = CastOp::create(b, type, newResult);

  castOp.replaceAllUsesWith(newResult);
  return success();
}

//===----------------------------------------------------------------------===//
// legalizeOperation
//===----------------------------------------------------------------------===//
LogicalResult LegalizePOPOperations::legalizeOperation(Operation *op,
                                                       TargetInfoAttr target) {
  assert(operationRequiresLegalization(op, target) &&
         "legalization not required");
  ImplicitLocOpBuilder b(op->getLoc(), op);

  if (!llvm::all_of(op->getOperands(), [&](Value operand) {
        return operand.getType() == op->getOperand(0).getType();
      })) {
    return op->emitError(
        "Cannot legalize operation with different operand types");
  }

  if (!llvm::all_of(op->getResultTypes(), [&](Type resultType) {
        return resultType == op->getResultTypes()[0];
      })) {
    return op->emitError(
        "Cannot legalize operation with different result types");
  }

  if (op->getOperand(0).getType() != op->getResultTypes()[0])
    return op->emitError("Cannot legalize non-homogeneous operation ");

  Type inputType = op->getOperand(0).getType();
  SmallVector<Type> types =
      findConversionSequence(inputType, inputType, target);
  if (types.empty()) {
    return op->emitError(
        "cannot legalize operand to LLVM's supported type on that target");
  }
  SmallVector<Value> newOperands;

  // First type for which it's safe to perform an operation
  auto supportedTypeIt = llvm::find_if(
      types, [&](Type type) { return !typeRequiresLegalization(type); });

  for (Value operand : op->getOperands()) {
    Value newOperand = operand;
    // Do conversion of the operand until it has supported type
    for (auto typeIt = types.begin(); typeIt != supportedTypeIt; ++typeIt)
      newOperand = CastOp::create(b, *typeIt, newOperand);
    newOperand = CastOp::create(b, *supportedTypeIt, newOperand);
    newOperands.push_back(newOperand);
  }

  SmallVector<Type> newResultTypes(op->getNumResults(), *supportedTypeIt);

  OperationState state(op->getLoc(), op->getName(), newOperands, newResultTypes,
                       op->getAttrs());
  // Construct new operation that with a supported type
  Operation *newOp = b.create(state);

  // Finally perform remaining conversion of the result down to the original
  // type.
  for (auto [newResult, oldResult] :
       llvm::zip(newOp->getResults(), op->getResults())) {
    Value resultToUse = newResult;
    for (auto typeIt = std::next(supportedTypeIt, 1); typeIt != types.end();
         ++typeIt) {
      resultToUse = CastOp::create(b, *typeIt, resultToUse);
    }
    resultToUse = CastOp::create(b, oldResult.getType(), resultToUse);
    oldResult.replaceAllUsesWith(resultToUse);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// runOnOperation
//===----------------------------------------------------------------------===//

void LegalizePOPOperations::runOnOperation() {
  Operation *op = getOperation();
  TargetInfoAttr target = lookupTargetInfo(op);
  if (getTargetKey(target).empty())
    return;

  initializeTargetLegalConversions(&getContext());

  op->walk([&](Operation *op) {
    if (auto castOp = dyn_cast<CastOp>(op)) {
      if (conversionRequiresLegalization(castOp, target)) {
        if (failed(legalizeConversion(castOp, target)))
          return WalkResult::interrupt();
        castOp->erase();
      }
      return WalkResult::advance();
    }
    if (!operationRequiresLegalization(op, target))
      return WalkResult::advance();

    if (failed(legalizeOperation(op, target)))
      return WalkResult::interrupt();
    op->erase();
    return WalkResult::advance();
  });
}

} // namespace
