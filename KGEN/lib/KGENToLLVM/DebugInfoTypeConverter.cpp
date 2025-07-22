//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/COTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
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

/// Returns a type name for the given dtype, specialized to the target if
/// applicable.
static std::string getKGENDTypeAsString(TargetInfoAttr targetInfo,
                                        KGENDType dtype) {
  if (targetInfo.getTriple().isNVPTX())
    if (std::optional<std::string> name =
            DebugInfoEncoding::getKGENDTypeAsCppString(dtype))
      return *name;
  return DebugInfoEncoding::getKGENDTypeAsString(dtype);
}

/// Build an integer or floating point debug type `T` with the given name and
/// width.
template <typename T>
auto buildIntFpDebugType(MLIRContext *ctx, TargetInfoAttr targetInfo,
                         uint8_t dtype, unsigned width,
                         unsigned conservativeAlign) {
  // TODO: This should be driven by target info in the longer term.
  uint32_t align =
      llvm::PowerOf2Ceil(llvm::divideCeil(width, CHAR_BIT)) * CHAR_BIT;
  align = std::min(align, conservativeAlign);

  uint64_t size = llvm::alignTo(width, align);
  return T::get(ctx, getKGENDTypeAsString(targetInfo, KGENDType(dtype)), size,
                align);
}

/// Returns a target-specific conversion for the dtype if applicable. Otherwise
/// returns null.
static DIType buildTargetSpecificDebugTypeFromDType(MLIRContext *ctx,
                                                    TargetInfoAttr targetInfo,
                                                    uint8_t dtype,
                                                    size_t indexWidth) {
  if (targetInfo.getTriple().isNVPTX()) {
    // cuda-gdb expects non-standard data types to be structure types with
    // special names. e.g.
    // https://docs.nvidia.com/cuda/cuda-math-api/struct____nv__bfloat16.html
    if (llvm::is_contained(
            {DType::f8e5m2, DType::f8e4m3fn, DType::bf16, DType::f16}, dtype)) {
      StringRef name;
      if (dtype == DType::f8e5m2)
        name = "__nv_fp8_e5m2";
      else if (dtype == DType::f8e4m3fn)
        name = "__nv_fp8_e4m3";
      else if (dtype == DType::bf16)
        name = "__nv_bfloat16";
      else
        name = "__half";
      unsigned width = DType(dtype).getWidthInBits();
      DType storageDType = *DType::getInt(width, /*isSigned=*/false);
      // The structure contains a single `__x` field.
      auto baseType = buildIntFpDebugType<DIBasicUIntType>(
          ctx, targetInfo, storageDType.getValue(), width, width);
      auto memberType = DIMemberType::get("__x", baseType);
      return DIStructType::get(StringAttr::get(ctx, name), {memberType});
    }
  }

  return {};
}

static DIType buildDebugTypeFromDType(MLIRContext *ctx,
                                      TargetInfoAttr targetInfo, uint8_t dtype,
                                      size_t indexWidth) {
  // Check if the target implements specialized conversions for this dtype.
  if (DIType specializedResult = buildTargetSpecificDebugTypeFromDType(
          ctx, targetInfo, dtype, indexWidth))
    return specializedResult;

  // Process various builtin dtypes.
  switch (dtype) {
  case DType::kBool:
    return buildIntFpDebugType<DIBasicBoolType>(ctx, targetInfo, dtype, 8, 8);
  case DType::si1:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, targetInfo, dtype, 1, 1);
  case DType::ui1:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, targetInfo, dtype, 1, 1);
  case DType::si2:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, targetInfo, dtype, 2, 8);
  case DType::ui2:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, targetInfo, dtype, 2, 8);
  case DType::si4:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, targetInfo, dtype, 4, 8);
  case DType::ui4:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, targetInfo, dtype, 4, 8);
  case DType::si8:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, targetInfo, dtype, 8, 8);
  case DType::ui8:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, targetInfo, dtype, 8, 8);
  case DType::si16:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, targetInfo, dtype, 16, 16);
  case DType::ui16:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, targetInfo, dtype, 16, 16);
  case DType::si32:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, targetInfo, dtype, 32, 32);
  case DType::ui32:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, targetInfo, dtype, 32, 32);
  case DType::si64:
    return buildIntFpDebugType<DIBasicSIntType>(ctx, targetInfo, dtype, 64, 64);
  case DType::ui64:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, targetInfo, dtype, 64, 64);

  case DType::invalid:
    return DIUnspecifiedType::get(
        ctx, getKGENDTypeAsString(targetInfo, KGENDType(dtype)));

  case KGENDType::address:
  case KGENDType::index:
    return buildIntFpDebugType<DIBasicUIntType>(ctx, targetInfo, dtype,
                                                indexWidth, indexWidth);

    // TODO: Process the remaining dtypes.
  default:
    DType type(dtype);
    if (type.isSInt())
      return buildIntFpDebugType<DIBasicSIntType>(
          ctx, targetInfo, dtype, type.getIntegerWidthInBits(), 64);
    if (type.isUInt())
      return buildIntFpDebugType<DIBasicUIntType>(
          ctx, targetInfo, dtype, type.getIntegerWidthInBits(), 64);
    if (auto *semantics = type.getFloatSemantics()) {
      size_t bitwidth = APFloat::getSizeInBits(*semantics);
      return buildIntFpDebugType<DIBasicFloatType>(ctx, targetInfo, dtype,
                                                   bitwidth, bitwidth);
    }

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
                                     ctx, targetInfo, KGENDType::si8, 0))),
       DIMemberType::get("size", convertDebugType(IndexType::get(ctx)))});
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(GeneratorType type) {
  // TODO(MOCO-1513): Remove generator type case from type lowerer.
  return convertDebugType(type.getBody());
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(FuncType type) {
  return buildPointerType(buildDebugSubroutineType(type.getValues()));
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(ParamType type) {
  if (isa<UnknownAttr, UninitMemAttr>(type.getParam()))
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

  DIType baseType;
  if (std::optional<KGENDType> dtype = type.getResolvedDType()) {
    baseType =
        buildDebugTypeFromDType(type.getContext(), targetInfo,
                                dtype->getValue(), tc.getIndexTypeBitwidth());
  } else {
    baseType = DIUnspecifiedType::get(type.getContext(), "unknown");
  }
  return DIVectorType::get(
      baseType, size,
      StringAttr::get(type.getContext(), mlir::debugString(type)));
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(StructType type) {
  return buildDebugStructTypeFromTypeAttrs(
      type.getElementTypes(),
      StringAttr::get(type.getContext(), mlir::debugString(type)));
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(StructInstanceType type) {
  // Recursively translate member types.
  SmallVector<DebugInfo::DIMemberType> memberTypes;
  for (StructDefFieldAttr field : type.getFields()) {
    DebugInfo::DIType fieldDIType = convertDebugType(field.getType());
    auto memberDIType =
        DebugInfo::DIMemberType::get(field.getName(), fieldDIType);
    memberTypes.push_back(memberDIType);
  }
  return DebugInfo::DIStructType::get(type.getName(), memberTypes);
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(TypeValueType type) {
  TypedAttr typeValue = type.getTypeValue();
  if (auto cst = dyn_cast<TypeParamAttr>(typeValue))
    return buildDebugType(cst);
  if (auto ref = dyn_cast<TypeInstanceRefAttr>(typeValue))
    return buildDebugType(ref);
  llvm_unreachable("illegal non-concrete type value");
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(TypeParamAttr attr) {
  return convertDebugType(attr.getTypeValue());
}

DIType KGEN::DebugInfoTypeConverter::buildDebugType(TypeInstanceRefAttr attr) {
  assert(attr.isConstant() && "illegal non-concrete type-constant reference");
  auto structInst =
      symtab.lookup<StructInstanceOp>(attr.getSymbol().getLeafReference());
  return convertDebugType(structInst.getValueDomainType());
}

KGEN::DebugInfoTypeConverter::DebugInfoTypeConverter(POPToLLVMTypeConverter &tc,
                                                     TargetInfoAttr targetInfo,
                                                     SymbolTable &symtab)
    : tc(tc), symtab(symtab), targetInfo(targetInfo) {
  // Let the LLVM conversion handle a majority of the debug info generation.
  addUnresolvedConverter(tc);

  // Add conversions for partially resolved debug info types.
  addConversion([&](DITargetIndependentPointerType type) {
    return buildDebugType(type);
  });

  // Add direct debug info conversions.
  addConversion([&](GeneratorType type) { return buildDebugType(type); });
  addConversion([&](IndexType type) { return buildDebugType(type); });
  addConversion([&](ParamType type) { return buildDebugType(type); });
  addConversion([&](StringType type) { return buildDebugType(type); });
  addConversion([&](FuncType type) { return buildDebugType(type); });
  addConversion([&](POP::UnionType type) { return buildDebugType(type); });
  addConversion([&](KGEN::NoneType type) { return buildDebugType(type); });
  addConversion([&](POP::ArrayType type) { return buildDebugType(type); });
  addConversion([&](CO::CoroutineType type) { return buildDebugType(type); });
  addConversion([&](PointerType type) { return buildDebugType(type); });
  addConversion([&](POP::SIMDType type) { return buildDebugType(type); });
  addConversion([&](StructType type) { return buildDebugType(type); });
  addConversion([&](StructInstanceType type) { return buildDebugType(type); });
  addConversion([&](TypeValueType type) { return buildDebugType(type); });

  // Break cyclic StructInstance types.
  addCycleBreaker([&](StructInstanceType type) {
    // TODO(MOCO-720): Encode recursive struct types.
    return DebugInfo::DIStructType::get(type.getName(), /*members=*/{});
  });
}
