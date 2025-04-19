//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
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

/// Return simiar type to \p type, but with scalar type of \p dtype
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
  DenseMap<StringRef, DenseMap<KGENDType, SmallVector<KGENDType>>>
      targetLegalConversion;

  /// Initialize map of known conversions for a target
  void initializeTargetLegalConversions(MLIRContext *ctx);

  /// Return true if legalization of the operation succeeded.
  LogicalResult legalizeOperation(Operation *op, TargetInfoAttr target);

  /// Return true if legalization has to be done for the operation. Return false
  /// otherwise.
  /// An operation requires legalization if lowering the operation to LLVM with
  /// converted types will generate invalid IR.
  /// For now legalize all arithmetic operations that operate on unsupported by
  /// LLVM types, such as F8 types.
  bool operationRequiresLegalization(Operation *op, TargetInfoAttr target);

  /// Return true if the type requires legalization
  bool typeRequiresLegalization(KGENDType dtype);
  bool typeRequiresLegalization(Type type);

  /// Return supported type that can be used instead of \p type
  Type getSupportedType(Type type, TargetInfoAttr target);
};

//===----------------------------------------------------------------------===//
// initializeTargetLegalConversions
//===----------------------------------------------------------------------===//

void LegalizePOPOperations::initializeTargetLegalConversions(MLIRContext *ctx) {
  // List of conversion for NVPTX Hopper and above
  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f8e5m2)].push_back(
      KGENDType(KGENDType::f16));
  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f8e5m2)].push_back(
      KGENDType(KGENDType::f32));

  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f8e4m3fn)]
      .push_back(KGENDType(KGENDType::f16));
  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f8e4m3fn)]
      .push_back(KGENDType(KGENDType::f32));

  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f32)].push_back(
      KGENDType(KGENDType::f8e5m2));
  targetLegalConversion["nvptx-sm_90"][KGENDType(KGENDType::f32)].push_back(
      KGENDType(KGENDType::f8e4m3fn));
}

//===----------------------------------------------------------------------===//
// typeRequiresLegalization
//===----------------------------------------------------------------------===//

/// Return true if the type requires legalization
bool LegalizePOPOperations::typeRequiresLegalization(KGENDType dtype) {
  if (dtype.isInvalid())
    return false;

  // Assume that any floating point type below 8 bits are not supported by
  // LLVM
  return dtype.isFloat() && dtype.getWidthInBits() <= 8;
}

/// Return true if the type requires legalization
bool LegalizePOPOperations::typeRequiresLegalization(Type type) {
  return typeRequiresLegalization(getScalarKGENDType(type));
}

//===----------------------------------------------------------------------===//
// operationRequiresLegalization
//===----------------------------------------------------------------------===//

bool LegalizePOPOperations::operationRequiresLegalization(
    Operation *op, TargetInfoAttr target) {
  // If operands have supported types, do not need to do anything extra with the
  // operation.
  for (auto operand : op->getOperands())
    if (!typeRequiresLegalization(operand.getType()))
      return false;

  return isa<NegOp, AddOp, SubOp, MulOp, DivOp, RemOp, MaxOp, MinOp>(op);
}

//===----------------------------------------------------------------------===//
// getSupportedType
//===----------------------------------------------------------------------===//

Type LegalizePOPOperations::getSupportedType(Type type, TargetInfoAttr target) {
  // TODO: Support other targets as well
  if (!isNVPTX_HopperAndAbove(target))
    return nullptr;

  StringRef targetKey = "nvptx-sm_90";

  KGENDType dtype = getScalarKGENDType(type);
  assert(dtype.isValid() && "dtype must be valid");

  auto targetConversionsIt = targetLegalConversion.find(targetKey);
  assert(targetConversionsIt != targetLegalConversion.end() &&
         "targetLegalConversion map was not initialized");
  auto typeConversionsIt = targetConversionsIt->second.find(dtype);
  if (typeConversionsIt == targetConversionsIt->second.end())
    return nullptr;

  // So far we have found a conversion from type to other types. We cannot
  // simply take any type toType from that list as
  //  - toType can be unsupported by LLVM
  //  - there's no roundtrip conversion implemented in POP -> LLVM Dialect
  //    lowering, i.e. `toType -> type` conversion may not exist.
  // Iterate over all available conversion to make sure toType is making sense
  // to be used.
  for (KGENDType toDType : typeConversionsIt->second) {
    if (typeRequiresLegalization(toDType))
      continue;

    auto toTypeConversionsIt = targetConversionsIt->second.find(toDType);
    if (toTypeConversionsIt == targetConversionsIt->second.end()) {
      // No conversions from toType available
      continue;
    }

    // If there's a conversion from toType to type, we can use toType for
    // legalization
    // TODO: Return the smallest available type
    if (llvm::any_of(toTypeConversionsIt->second,
                     [&](KGENDType t) { return t == dtype; })) {
      return convertKGENDTypeToType(toDType, type);
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// legalizeOperation
//===----------------------------------------------------------------------===//
LogicalResult LegalizePOPOperations::legalizeOperation(Operation *op,
                                                       TargetInfoAttr target) {
  assert(operationRequiresLegalization(op, target) &&
         "legalization not required");
  ImplicitLocOpBuilder b(op->getLoc(), op);

  SmallVector<Value> newOperands;

  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    Type type = operand.getType();
    if (!typeRequiresLegalization(type)) {
      newOperands.push_back(operand);
      continue;
    }

    Type newType = getSupportedType(type, target);

    if (!newType) {
      return op->emitError("cannot legalize operand #")
             << idx << " to LLVM's supported type on that target";
    }

    newOperands.push_back(b.create<CastOp>(newType, operand));
  }

  SmallVector<Type> newResultTypes;
  for (auto [idx, resType] : llvm::enumerate(op->getResultTypes())) {
    Type newType = getSupportedType(resType, target);
    if (!newType) {
      return op->emitError("cannot legalize result ") << idx
                                                      << " to LLVM's supported "
                                                         "type on that target";
    }
    newResultTypes.push_back(newType);
  }

  OperationState state(op->getLoc(), op->getName(), newOperands, newResultTypes,
                       op->getAttrs());
  Operation *newOp = b.create(state);

  for (auto [newResult, oldResult] :
       llvm::zip(newOp->getResults(), op->getResults())) {
    Value resultToUse = newResult;
    if (newResult.getType() != oldResult.getType())
      resultToUse = b.create<CastOp>(oldResult.getType(), newResult);

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

  initializeTargetLegalConversions(&getContext());

  op->walk([&](Operation *op) {
    if (!operationRequiresLegalization(op, target))
      return WalkResult::advance();

    if (failed(legalizeOperation(op, target)))
      return WalkResult::interrupt();
    op->erase();
    return WalkResult::advance();
  });
}

} // namespace
