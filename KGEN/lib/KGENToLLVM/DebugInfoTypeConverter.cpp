//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/COTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/DebugInfoEncoding.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Support/DebugStringHelper.h"

using namespace M;
using namespace KGEN;
using namespace DebugInfo;

//===----------------------------------------------------------------------===//
// DebugInfoTypeConverter
//===----------------------------------------------------------------------===//

/// Build an integer or floating point debug type `T` with the given name and
/// width.
template <typename T>
auto buildIntFpDebugType(MLIRContext *ctx, uint8_t dtype, unsigned width,
                         unsigned conservativeAlign) {
  // TODO: This should be driven by target info in the longer term.
  uint32_t align =
      llvm::PowerOf2Ceil(llvm::divideCeil(width, CHAR_BIT)) * CHAR_BIT;
  align = std::min(align, conservativeAlign);

  uint64_t size = llvm::alignTo(width, align);
  return T::get(ctx, DebugInfoEncoding::getKGENDTypeAsString(KGENDType(dtype)),
                size, align);
}

static DIType buildDebugTypeFromDType(MLIRContext *ctx, uint8_t dtype,
                                      size_t indexWidth) {
  // Process various builtin dtypes.
  switch (dtype) {
  case DType::kBool:
    return buildIntFpDebugType<DIBasicBoolType>(ctx, dtype, 8, 8);
  case DType::si1:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, dtype, 1, 1);
  case DType::ui1:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, dtype, 1, 1);
  case DType::si2:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, dtype, 2, 8);
  case DType::ui2:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, dtype, 2, 8);
  case DType::si4:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, dtype, 4, 8);
  case DType::ui4:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, dtype, 4, 8);
  case DType::si8:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, dtype, 8, 8);
  case DType::ui8:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, dtype, 8, 8);
  case DType::si16:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, dtype, 16, 16);
  case DType::ui16:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, dtype, 16, 16);
  case DType::si32:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, dtype, 32, 32);
  case DType::ui32:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, dtype, 32, 32);
  case DType::si64:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, dtype, 64, 64);
  case DType::ui64:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, dtype, 64, 64);
  case DType::si128:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, dtype, 128, 64);
  case DType::ui128:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, dtype, 128, 64);

  case DType::f8:
    return buildIntFpDebugType<DIBasicFloatType>(ctx, dtype, 8, 8);
  case DType::f16:
    return buildIntFpDebugType<DIBasicFloatType>(ctx, dtype, 16, 16);
  case DType::f32:
    return buildIntFpDebugType<DIBasicFloatType>(ctx, dtype, 32, 32);
  case DType::f64:
    return buildIntFpDebugType<DIBasicFloatType>(ctx, dtype, 64, 64);
  case DType::f128:
    return buildIntFpDebugType<DIBasicFloatType>(ctx, dtype, 128, 64);
  case DType::bf16:
    return buildIntFpDebugType<DIBasicFloatType>(ctx, dtype, 16, 16);
  case DType::f24:
    return buildIntFpDebugType<DIBasicFloatType>(ctx, dtype, 24, 32);
  case DType::f80:
    return buildIntFpDebugType<DIBasicFloatType>(ctx, dtype, 80, 64);
  case DType::tf32:
    return buildIntFpDebugType<DIBasicFloatType>(ctx, dtype, 32, 32);

  case DType::invalid:
    return DIUnspecifiedType::get(
        ctx, DebugInfoEncoding::getKGENDTypeAsString(KGENDType(dtype)));

  case KGENDType::address:
  case KGENDType::index:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, dtype, indexWidth,
                                                indexWidth);

    // TODO: Process the remaining dtypes.
  default:
    return nullptr;
  }
}

DIType KGEN::DebugInfoTypeConverter::buildDebugStructTypeFromTypeAttrs(
    ArrayRef<Type> types, StringAttr name) {
  SmallVector<DIMemberType> elementTypes;
  for (auto [idx, type] : llvm::enumerate(types)) {
    DIType mDIType = convertDebugType(type);
    DIMemberType mMemberDIType = DIMemberType::get(
        StringAttr::get(name.getContext(), "m" + Twine(idx)), mDIType);
    elementTypes.push_back(mMemberDIType);
  }
  return DIStructType::get(name, elementTypes);
}

DIType
KGEN::DebugInfoTypeConverter::buildDebugSubroutineType(FunctionType type) {
  SmallVector<DIType> argTypes, resultTypes;
  for (Type arg : type.getInputs())
    argTypes.push_back(convertDebugType(arg));
  for (Type result : type.getResults())
    resultTypes.push_back(convertDebugType(result));
  return DISubroutineType::get(type.getContext(), argTypes, resultTypes);
}

DIType KGEN::DebugInfoTypeConverter::buildPointerType(DIType type) {
  return buildPointerType(type, /*addressSpace=*/std::nullopt);
}
DIType KGEN::DebugInfoTypeConverter::buildPointerType(
    DIType type, std::optional<unsigned> addressSpace) {
  return DIPointerType::get(type, tc.getPointerBitwidth(),
                            tc.getPointerBitwidth(), addressSpace);
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(
    DITargetIndependentPointerType type) {
  return buildPointerType(convertDebugType(type.getElementType()));
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(IndexType type) {
  // We treat index types as signed.
  return DIBasicSIntType::get(type.getContext(), "index",
                              tc.getIndexTypeBitwidth(),
                              tc.getIndexTypeBitwidth());
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(StringType type) {
  MLIRContext *ctx = type.getContext();
  Builder b(ctx);
  // This must be kept in sync with `getLLVMTYpeForKGENStringType`.
  return DIStructType::get(
      b.getStringAttr("!kgen.string"),
      {DIMemberType::get("data", buildPointerType(buildDebugTypeFromDType(
                                     ctx, KGENDType::si8, 0))),
       DIMemberType::get("size", convertDebugType(IndexType::get(ctx)))});
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(SignatureType type) {
  return buildPointerType(buildDebugSubroutineType(type.getValues()));
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(ParamRefType type) {
  if (isa<UnknownAttr>(type.getParam()))
    return DIUnspecifiedType::get(type.getContext(), "unknown");
  llvm_unreachable("unresolved type parameter in debuginfo");
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(POP::UnionType type) {
  SmallVector<DIMemberType> variantMembers;
  uint64_t maxMemberSizeInBits = 0;
  for (auto [index, member] : llvm::enumerate(type.getTypes())) {
    DIType debugType = convertDebugType(member);
    variantMembers.push_back(DIMemberType::get("v" + Twine(index), debugType));
    maxMemberSizeInBits =
        std::max(maxMemberSizeInBits, debugType.getSizeInBits());
  }

  return DIVariantType::get(
      StringAttr::get(type.getContext(), ""), maxMemberSizeInBits,
      *type.getTypeAlign(tc.getTarget()) * CHAR_BIT, variantMembers);
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(KGEN::NoneType type) {
  return DIStructType::get(StringAttr::get(type.getContext(), "!kgen.none"));
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(POP::ArrayType type) {
  int64_t size = *type.getResolvedSize();
  DIType elementType = convertDebugType(type.getElementType());
  return DIArrayType::get(elementType, size);
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(CO::CoroutineType type) {
  return buildPointerType(
      DIStructType::get(StringAttr::get(type.getContext(), "!co.routine")));
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(PointerType type) {
  return buildPointerType(convertDebugType(type.getElementType()),
                          cast<IntegerAttr>(type.getAddressSpace()).getInt());
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(POP::SIMDType type) {
  int64_t size = *type.getResolvedSize();
  DIType baseType = buildDebugTypeFromDType(type.getContext(),
                                            type.getResolvedDType()->getValue(),
                                            tc.getIndexTypeBitwidth());
  return DIVectorType::get(
      baseType, size,
      StringAttr::get(type.getContext(), mlir::debugString(type)));
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(StructType type) {
  return buildDebugStructTypeFromTypeAttrs(
      type.getElementTypes(),
      StringAttr::get(type.getContext(), mlir::debugString(type)));
}

KGEN::DebugInfoTypeConverter::DebugInfoTypeConverter(POPToLLVMTypeConverter &tc)
    : tc(tc) {
  // Let the LLVM conversion handle a majority of the debug info generation.
  addUnresolvedConverter(tc);

  // Add conversions for partially resolved debug info types.
  addConversion([&](DITargetIndependentPointerType type) {
    return buildDebugType(type);
  });

  // Add direct debug info conversions.
  addConversion([&](IndexType type) { return buildDebugType(type); });
  addConversion([&](ParamRefType type) { return buildDebugType(type); });
  addConversion([&](StringType type) { return buildDebugType(type); });
  addConversion([&](SignatureType type) { return buildDebugType(type); });
  addConversion([&](POP::UnionType type) { return buildDebugType(type); });
  addConversion([&](KGEN::NoneType type) { return buildDebugType(type); });
  addConversion([&](POP::ArrayType type) { return buildDebugType(type); });
  addConversion([&](CO::CoroutineType type) { return buildDebugType(type); });
  addConversion([&](PointerType type) { return buildDebugType(type); });
  addConversion([&](POP::SIMDType type) { return buildDebugType(type); });
  addConversion([&](StructType type) { return buildDebugType(type); });
}
