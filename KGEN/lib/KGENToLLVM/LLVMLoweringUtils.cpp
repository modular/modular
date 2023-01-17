//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLVMLoweringUtils.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

/// Since !kgen.string type is not parameterized on the size of the string,
/// we lower it as struct with a pointer field holding the data and a index
/// field holding the string size.
static Type getLLVMTypeForKGENStringType(MLIRContext *ctx, Type strSizeType) {
  SmallVector<Type> elementTypes{
      LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8)), strSizeType};
  return LLVM::LLVMStructType::getLiteral(ctx, elementTypes);
}

//===----------------------------------------------------------------------===//
// POPToLLVMTypeConverter
//===----------------------------------------------------------------------===//

std::optional<Type> M::KGEN::getMLIRTypeForDType(MLIRContext *ctx,
                                                 KGENDType dtype,
                                                 size_t indexBitwidth) {
  if (dtype.isBool())
    return IntegerType::get(ctx, 1);

  if (dtype.isAddress())
    return LLVM::LLVMPointerType::get(ctx);

  if (dtype.isIndex())
    return IntegerType::get(ctx, indexBitwidth);

  // This intentionally discards signed-ness because LLVM is signless.
  if (dtype.isInt())
    return IntegerType::get(ctx, dtype.getIntegerWidthInBits());

  if (dtype.isFloat()) {
    if (FloatType fpType = getEquivalentFloatType(ctx, dtype))
      return fpType;
    return {};
  }

  return {};
}

Type M::KGEN::getLLVMPointerTo(MLIRContext *ctx, KGENDType dtype,
                               size_t indexBitwidth) {
  if (std::optional<Type> type = getMLIRTypeForDType(ctx, dtype, indexBitwidth))
    return LLVM::LLVMPointerType::get(*type);
  return LLVM::LLVMPointerType::get(ctx);
}

POPToLLVMTypeConverter::POPToLLVMTypeConverter(
    mlir::Location loc, const mlir::LowerToLLVMOptions &options)
    : LLVMTypeConverter(loc.getContext(), options), loc(loc) {

  // Convert a DType expression to an MLIR type.
  auto convertDType = [&](auto type) -> std::optional<Type> {
    if (std::optional<KGENDType> dtype = type.getResolvedDType())
      return getMLIRTypeForDType(type.getContext(), *dtype,
                                 options.getIndexBitwidth());
    return {};
  };

  // Convert a size expression to a C++ unsigned integer.
  auto convertSize = [&](auto type) -> std::optional<uint64_t> {
    auto size = dyn_cast_if_present<IntegerAttr>(type.getSize());
    if (!size)
      return {};
    const APInt &value = size.getValue();
    assert(APInt(value.getBitWidth(), value.getLimitedValue()) == value &&
           "couldn't narrow vector size");
    return value.getLimitedValue();
  };

  // Convert pointer types to LLVM pointer types. If the element type is
  // unspecified, return an opaque pointer.
  addConversion([=](POP::PointerType pointer) -> std::optional<Type> {
    if (Type type = pointer.getResolvedElementType())
      if (Type elementType = convertType(type))
        return LLVM::LLVMPointerType::get(elementType);
    return LLVM::LLVMPointerType::get(pointer.getContext());
  });

  // Convert array types to LLVM array types.
  addConversion([=](POP::ArrayType array) -> std::optional<Type> {
    std::optional<int64_t> size = array.getResolvedSize();
    Type elementType = array.getResolvedElementType();
    if (!size || !elementType)
      return {};
    elementType = convertType(elementType);
    if (!elementType)
      return {};
    return LLVM::LLVMArrayType::get(elementType, *size);
  });

  // Convert string types to LLVM literal structs: struct{ptr, size} of type
  // !llvm.struct<(ptr<i8>, index).
  addConversion([=](KGEN::StringType stringType) -> std::optional<Type> {
    return getLLVMTypeForKGENStringType(stringType.getContext(),
                                        getIndexType());
  });

  // Convert struct types to LLVM literal structs.
  addConversion([=](POP::StructType structType) -> std::optional<Type> {
    SmallVector<Type> elementTypes;
    elementTypes.reserve(structType.getNumElements());
    for (TypedAttr elementType : structType.getElementTypes()) {
      auto typeCst = dyn_cast<ConcreteTypeConstantAttr>(elementType);
      if (!typeCst)
        return {};
      Type converted = convertType(typeCst.getValue());
      if (!converted)
        return {};
      elementTypes.push_back(converted);
    }
    return LLVM::LLVMStructType::getLiteral(&getContext(), elementTypes);
  });

  // Convert SIMD types to vector types.
  addConversion([=](POP::SIMDType simd) -> std::optional<Type> {
    std::optional<Type> dtype = convertDType(simd);
    std::optional<uint64_t> size = convertSize(simd);
    if (!dtype)
      return {};
    if (!size) {
      emitError("SIMD size not fully specified: ") << simd;
      return {};
    }

    // Scalar case, size = 1
    if (*size == 1)
      return *dtype;

    // Vector case, size != 1
    return LLVM::getFixedVectorType(*dtype, *size);
  });

  // Convert data type types to `i8`.
  addConversion([=](DTypeType dtype) -> std::optional<Type> {
    return Builder(&getContext()).getI8Type();
  });

  // Convert variant types to a struct with enough space to contain the largest
  // variant type plus a discriminator.
  addConversion([=](POP::VariantType variant) -> std::optional<Type> {
    // TODO: The generated assembly is sensitive to the content type of the
    // variant type. This needs to be optimized. For now, use an array of
    // word-size integers.
    uint64_t maxSize = 0;
    for (TypedAttr typeExpr : variant.getTypes()) {
      Type variantType = typeExpr.cast<ConcreteTypeConstantAttr>().getValue();
      Type type = convertType(variantType);
      if (!type)
        return {};
      maxSize = std::max(maxSize, llvm::alignTo(dl.getTypeSize(type),
                                                dl.getTypeABIAlignment(type)));
    }
    // FIXME: The alignment of the generated type must equal or exceed the
    // greatest alignment requirement of any subtype. Right now it's just the
    // pointer width.
    auto contentType = LLVM::LLVMArrayType::get(
        getIndexType(),
        llvm::divideCeil(maxSize * CHAR_BIT, getIndexTypeBitwidth()));
    // Compute the smallest integer to contain the discriminator. If there is
    // only one variant type, use i1 since LLVM does not accept i0.
    auto discrType = IntegerType::get(
        &getContext(),
        std::max(1u, llvm::Log2_32_Ceil(variant.getTypes().size())));
    return LLVM::LLVMStructType::getLiteral(&getContext(),
                                            {contentType, discrType});
  });

  // Coroutine handles are always lowered to opaque pointers.
  addConversion([](POP::CoroutineType coro) {
    return LLVM::LLVMPointerType::get(Builder(coro.getContext()).getI8Type());
  });
}

//===----------------------------------------------------------------------===//
// VariantHelper
//===----------------------------------------------------------------------===//

/// Advance the variant storage pointer by a set amount no more than the current
/// amount of remaining space in the current storage element.
template <typename ItType>
static unsigned advanceStoragePtr(ItType &valueIt, unsigned &storageOffset,
                                  unsigned amt) {
  auto curStorageSize = cast<IntegerType>(valueIt->getType()).getWidth();
  unsigned advanceBy = std::min(amt, curStorageSize - storageOffset);
  storageOffset += advanceBy;
  if (storageOffset == curStorageSize) {
    ++valueIt;
    storageOffset = 0;
  }
  return advanceBy;
}

/// Pad the variant storage by a bit amount. This is used to add padding to the
/// variant layout.
template <typename ItType>
static void addStoragePadding(ItType &valueIt, unsigned &storageOffset,
                              unsigned &offset, unsigned alignment) {
  unsigned padding = llvm::alignTo(offset, alignment * CHAR_BIT) - offset;
  for (unsigned added = 0; added != padding;)
    added += advanceStoragePtr(valueIt, storageOffset, padding - added);
  offset += padding;
}

void VariantHelper::walkAndCreateVariant(
    MutableArrayRef<Value>::iterator &valueIt, unsigned &storageOffset,
    unsigned &offset, Value value) {
  // Align the storage pointer to the current value being stored.
  addStoragePadding(valueIt, storageOffset, offset,
                    dl.getTypeABIAlignment(value.getType()));

  // Aggregate types like structs and arrays are flattened to their leaf types.
  // Leaf types are integers, floats, and pointers.
  if (isa<IntegerType, FloatType, LLVM::LLVMPointerType>(value.getType())) {
    Value normalizedValue;
    // Normalize the value to store to an integer.
    if (isa<IntegerType>(value.getType()))
      normalizedValue = value;
    else if (auto fpType = dyn_cast<FloatType>(value.getType()))
      normalizedValue =
          b.create<LLVM::BitcastOp>(b.getIntegerType(fpType.getWidth()), value);
    else
      normalizedValue = b.create<LLVM::PtrToIntOp>(
          b.getIntegerType(dl.getTypeSizeInBits(value.getType())), value);

    unsigned curValueSize =
        cast<IntegerType>(normalizedValue.getType()).getWidth();
    offset += curValueSize;
    unsigned curValueOffset = 0;
    while (curValueOffset != curValueSize) {
      // Compute the remaining space.
      auto curStorageType = cast<IntegerType>(valueIt->getType());

      // Ignore the bits of the value that has already been stored.
      Value valueToStore = b.create<LLVM::LShrOp>(
          normalizedValue, b.create<LLVM::ConstantOp>(normalizedValue.getType(),
                                                      curValueOffset));
      // Match the type with the storage type.
      if (curValueSize < curStorageType.getWidth())
        valueToStore = b.create<LLVM::ZExtOp>(curStorageType, valueToStore);
      else
        valueToStore = b.create<LLVM::TruncOp>(curStorageType, valueToStore);
      // Shift the current value to store to the current storage offset.
      valueToStore = b.create<LLVM::ShlOp>(
          valueToStore,
          b.create<LLVM::ConstantOp>(curStorageType, storageOffset));
      // Set the bits of the current value to store.
      *valueIt = b.create<LLVM::OrOp>(*valueIt, valueToStore);

      curValueOffset += advanceStoragePtr(valueIt, storageOffset,
                                          curValueSize - curValueOffset);
    }

    // The value has been stored.
    return;
  }

  // This is an aggregate type. Extract the next elements and recurse.
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(value.getType())) {
    for (unsigned i = 0, e = arrayType.getNumElements(); i < e; ++i) {
      Value nestedValue = b.create<LLVM::ExtractValueOp>(value, i);
      walkAndCreateVariant(valueIt, storageOffset, offset, nestedValue);
    }
    return;
  }
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(value.getType())) {
    for (unsigned i = 0, e = structType.getBody().size(); i < e; ++i) {
      Value nestedValue = b.create<LLVM::ExtractValueOp>(value, i);
      walkAndCreateVariant(valueIt, storageOffset, offset, nestedValue);
    }
    return;
  }
  if (auto vecType = dyn_cast<LLVM::LLVMFixedVectorType>(value.getType())) {
    for (unsigned i = 0, e = vecType.getNumElements(); i < e; ++i) {
      Value nestedValue = b.create<LLVM::ExtractElementOp>(
          value, b.create<LLVM::ConstantOp>(b.getI32Type(), i));
      walkAndCreateVariant(valueIt, storageOffset, offset, nestedValue);
    }
    return;
  }
  auto vectorType = cast<VectorType>(value.getType());
  for (unsigned i = 0, e = vectorType.getNumElements(); i < e; ++i) {
    Value nestedValue = b.create<LLVM::ExtractElementOp>(
        value, b.create<LLVM::ConstantOp>(b.getI32Type(), i));
    walkAndCreateVariant(valueIt, storageOffset, offset, nestedValue);
  }
}

Value VariantHelper::walkAndExtractVariant(ArrayRef<Value>::iterator &valueIt,
                                           unsigned &storageOffset,
                                           unsigned &offset, Type type) {
  // Align the storage pointer to the current value being stored.
  addStoragePadding(valueIt, storageOffset, offset,
                    dl.getTypeABIAlignment(type));

  // Given a leaf type, extract a value of that type from the current storage.
  if (isa<IntegerType, FloatType, LLVM::LLVMPointerType>(type)) {
    IntegerType normalizedType;
    if (auto intType = dyn_cast<IntegerType>(type))
      normalizedType = intType;
    else if (auto fpType = dyn_cast<FloatType>(type))
      normalizedType = b.getIntegerType(fpType.getWidth());
    else
      normalizedType = b.getIntegerType(dl.getTypeSize(type));

    unsigned curValueSize = normalizedType.getWidth();
    offset += curValueSize;
    unsigned curValueOffset = 0;
    Value curValue = b.create<LLVM::ConstantOp>(normalizedType, 0);
    while (curValueOffset != curValueSize) {
      auto storageType = cast<IntegerType>(valueIt->getType());

      // Drop the parts of the storage that have already been read.
      Value valueToLoad = b.create<LLVM::LShrOp>(
          *valueIt, b.create<LLVM::ConstantOp>(storageType, storageOffset));
      // Shift the data to load into position.
      valueToLoad = b.create<LLVM::ShlOp>(
          valueToLoad, b.create<LLVM::ConstantOp>(storageType, curValueOffset));
      // Match the type to the value type.
      if (normalizedType.getWidth() <= storageType.getWidth())
        valueToLoad = b.create<LLVM::TruncOp>(normalizedType, valueToLoad);
      else
        valueToLoad = b.create<LLVM::ZExtOp>(normalizedType, valueToLoad);

      curValue = b.create<LLVM::OrOp>(curValue, valueToLoad);

      curValueOffset += advanceStoragePtr(valueIt, storageOffset,
                                          curValueSize - curValueOffset);
    }

    if (isa<FloatType>(type))
      return b.create<LLVM::BitcastOp>(type, curValue);
    else if (isa<LLVM::LLVMPointerType>(type))
      return b.create<LLVM::IntToPtrOp>(type, curValue);
    return curValue;
  }

  // This is an aggregate type. Read the required elements.
  Value result = b.create<LLVM::UndefOp>(type);
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(type)) {
    for (unsigned i = 0, e = arrayType.getNumElements(); i < e; ++i) {
      Value element = walkAndExtractVariant(valueIt, storageOffset, offset,
                                            arrayType.getElementType());
      result = b.create<LLVM::InsertValueOp>(result, element, i);
    }
    return result;
  }
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
    for (auto [idx, elementType] : llvm::enumerate(structType.getBody())) {
      Value element =
          walkAndExtractVariant(valueIt, storageOffset, offset, elementType);
      result = b.create<LLVM::InsertValueOp>(result, element, idx);
    }
    return result;
  }
  if (auto vecType = dyn_cast<LLVM::LLVMFixedVectorType>(type)) {
    for (unsigned i = 0, e = vecType.getNumElements(); i < e; ++i) {
      Value element = walkAndExtractVariant(valueIt, storageOffset, offset,
                                            vecType.getElementType());
      result = b.create<LLVM::InsertElementOp>(
          result, element, b.create<LLVM::ConstantOp>(b.getI32Type(), i));
    }
    return result;
  }
  auto vectorType = cast<VectorType>(type);
  for (unsigned i = 0, e = vectorType.getNumElements(); i < e; ++i) {
    Value element = walkAndExtractVariant(valueIt, storageOffset, offset,
                                          vectorType.getElementType());
    result = b.create<LLVM::InsertElementOp>(
        result, element, b.create<LLVM::ConstantOp>(b.getI32Type(), i));
  }
  return result;
}

Value VariantHelper::materializeLLVMVariant(Type type, Value value,
                                            int64_t index) {
  auto variantType = cast<LLVM::LLVMStructType>(type);
  auto contentType = cast<LLVM::LLVMArrayType>(variantType.getBody().front());
  SmallVector<Value> storageValues;
  for (unsigned i = 0, e = contentType.getNumElements(); i < e; ++i)
    storageValues.push_back(
        b.create<LLVM::ConstantOp>(contentType.getElementType(), 0));

  MutableArrayRef<Value>::iterator valueIt = storageValues.begin();
  unsigned storageOffset = 0;
  unsigned offset = 0;
  walkAndCreateVariant(valueIt, storageOffset, offset, value);

  Value content = b.create<LLVM::UndefOp>(contentType);
  for (auto [idx, value] : llvm::enumerate(storageValues))
    content = b.create<LLVM::InsertValueOp>(content, value, idx);

  // Build the result struct.
  Value variant = b.create<LLVM::UndefOp>(variantType);
  variant = b.create<LLVM::InsertValueOp>(variant, content, 0);
  Value discrVal = b.create<LLVM::ConstantOp>(
      variantType.getBody().back().cast<IntegerType>(), index);
  return b.create<LLVM::InsertValueOp>(variant, discrVal, 1);
}

//===----------------------------------------------------------------------===//
// Attribute Conversion
//===----------------------------------------------------------------------===//

/// Convert a SIMD vector constant.
static Value convertSIMDAttr(ImplicitLocOpBuilder &b, TypeConverter &tc,
                             POP::SIMDAttr simd) {
  KGENDType dtype = *simd.getType().getResolvedDType();
  auto asConst = [&](TypedAttr value) {
    return b.create<LLVM::ConstantOp>(value);
  };

  // Handle scalar constants.
  if (simd.getValues().size() == 1) {
    const POP::DTypeValue &value = simd.getValues().front();
    if (dtype.isBool())
      return asConst(b.getBoolAttr(value.getBoolVal()));
    if (dtype.isInt())
      return asConst(b.getIntegerAttr(
          b.getIntegerType(dtype.getIntegerWidthInBits()), value.getIntVal()));
    if (dtype.isIndex() || dtype.isAddress()) {
      Value addr = asConst(b.getIntegerAttr(tc.convertType(b.getIndexType()),
                                            value.getIndexVal()));
      if (dtype.isIndex())
        return addr;
      return b.create<LLVM::IntToPtrOp>(
          LLVM::LLVMPointerType::get(b.getContext()), addr);
    }
    return asConst(b.getFloatAttr(getEquivalentFloatType(b.getContext(), dtype),
                                  value.getFloatVal()));
  }

  // Handle vector constants.
  if (dtype.isBool()) {
    SmallVector<APInt> values;
    for (const POP::DTypeValue &value : simd.getValues())
      values.emplace_back(1, value.getBoolVal());
    return asConst(IntArrayElementsAttr::get(
        VectorType::get(values.size(), b.getI1Type()), values));
  }
  if (dtype.isInt()) {
    SmallVector<APInt> values;
    for (const POP::DTypeValue &value : simd.getValues())
      values.push_back(value.getIntVal());
    return asConst(IntArrayElementsAttr::get(
        VectorType::get(values.size(),
                        b.getIntegerType(dtype.getIntegerWidthInBits())),
        values));
  }
  if (dtype.isIndex() || dtype.isAddress()) {
    SmallVector<APInt> values;
    auto indexType = cast<IntegerType>(tc.convertType(b.getIndexType()));
    for (const POP::DTypeValue &value : simd.getValues())
      values.push_back(APInt(indexType.getWidth(), value.getIndexVal()));
    Value addr = asConst(IntArrayElementsAttr::get(
        VectorType::get(values.size(), indexType), values));
    if (dtype.isIndex())
      return addr;
    return b.create<LLVM::IntToPtrOp>(
        LLVM::LLVMFixedVectorType::get(
            LLVM::LLVMPointerType::get(b.getContext()), values.size()),
        addr);
  }
  SmallVector<APFloat> values;
  for (const POP::DTypeValue &value : simd.getValues())
    values.push_back(value.getFloatVal());
  return asConst(FloatArrayElementsAttr::get(
      VectorType::get(values.size(),
                      getEquivalentFloatType(b.getContext(), dtype)),
      values));
}

/// Lower the string to a pop.global_constant and create a llvm struct of type
/// !llvm.struct<(ptr<i8>, i64)> holding the pointer to the global string and
/// its size.
static Value lowerStringToGlobalConstant(StringAttr strAttr,
                                         ImplicitLocOpBuilder &b,
                                         Type strSizeType) {
  SmallVector<TypedAttr> values;
  StringRef str = strAttr.getValue();
  auto charType = b.getType<POP::SIMDType>(1, DType::si8);
  for (char c : str)
    values.push_back(POP::SIMDAttr::get(
        {APSInt(APInt(CHAR_BIT, c), /*isUnsigned=*/false), DType::si8},
        charType));
  auto arrayType = POP::ArrayType::get(str.size(), charType);
  Value globalConst = b.create<POP::GlobalConstantOp>(
      POP::PointerType::get(arrayType), POP::ArrayAttr::get(values, arrayType));
  IntegerType byteType = b.getI8Type();
  Value ptrBitcast = b.create<POP::PointerBitcastOp>(
      b.getLoc(), POP::PointerType::get(byteType), globalConst);
  Value unrealizedCast =
      b.create<mlir::UnrealizedConversionCastOp>(
           b.getLoc(), LLVM::LLVMPointerType::get(byteType), ptrBitcast)
          .getResult(0);
  Value sizeVal =
      b.create<LLVM::ConstantOp>(b.getLoc(), b.getI64IntegerAttr(str.size()));
  Value undefOp = b.create<LLVM::UndefOp>(
      b.getLoc(), getLLVMTypeForKGENStringType(b.getContext(), strSizeType));
  Value structVal0 =
      b.create<LLVM::InsertValueOp>(b.getLoc(), undefOp, unrealizedCast, 0);
  return b.create<LLVM::InsertValueOp>(b.getLoc(), structVal0, sizeVal, 1);
}

Value KGEN::convertParameterToLLVM(ImplicitLocOpBuilder &b,
                                   mlir::LLVMTypeConverter &tc,
                                   TypedAttr attr) {
  //===--------------------------------------------------------------------===//
  // Builtin

  // Drop the sign on integer attributes; LLVM is signless.
  if (auto intCst = dyn_cast<IntegerAttr>(attr)) {
    return b.create<LLVM::ConstantOp>(
        b.getIntegerAttr(cast<IntegerType>(tc.convertType(intCst.getType())),
                         intCst.getValue()));
  }

  // Float attributes are fine as-is.
  if (isa<FloatAttr>(attr))
    return b.create<LLVM::ConstantOp>(attr);

  // Convert DType constants to `i8` constants of the DType's enum value.
  if (auto dtypeCst = dyn_cast<DTypeConstantAttr>(attr))
    return b.create<LLVM::ConstantOp>(
        b.getI8IntegerAttr(dtypeCst.getDType().getValue()));

  // Convert string constant to a struct{ptr, size} of type
  // !llvm.struct<(ptr<i8>, index).
  if (auto stringAttr = dyn_cast<StringAttr>(attr))
    return lowerStringToGlobalConstant(stringAttr, b, tc.getIndexType());

  //===--------------------------------------------------------------------===//
  // POP

  // Convert SIMD constants to an array of integer or float constants.
  if (auto simd = dyn_cast<POP::SIMDAttr>(attr))
    return convertSIMDAttr(b, tc, simd);

  // Convert array or struct constants to LLVM array or struct constants.
  if (isa<POP::ArrayAttr, POP::StructAttr>(attr)) {
    Type type = tc.convertType(attr.getType());
    if (!type)
      return {};
    Value aggregate = b.create<LLVM::UndefOp>(type);
    ArrayRef<TypedAttr> values = isa<POP::ArrayAttr>(attr)
                                     ? cast<POP::ArrayAttr>(attr).getValues()
                                     : cast<POP::StructAttr>(attr).getValues();
    for (auto [idx, value] : llvm::enumerate(values)) {
      Value element = convertParameterToLLVM(b, tc, value);
      if (!element)
        return {};
      aggregate = b.create<LLVM::InsertValueOp>(aggregate, element, idx);
    }
    return aggregate;
  }

  // Bitpack variant constants.
  if (auto variant = dyn_cast<POP::VariantAttr>(attr)) {
    auto variantType = llvm::cast_if_present<LLVM::LLVMStructType>(
        tc.convertType(variant.getType()));
    if (!variantType)
      return {};
    Value value = convertParameterToLLVM(b, tc, variant.getValue());
    if (!value)
      return {};

    VariantHelper helper(b, b.getLoc());
    return helper.materializeLLVMVariant(
        variantType, value,
        *variant.getType().getTypeIndex(variant.getValue().getType()));
  }

  // Unknown attribute to convert.
  return {};
}

//===----------------------------------------------------------------------===//
// POPToLLVMDebugInfoTypeConverter
//===----------------------------------------------------------------------===//

/// Build an integer or floating point debug type `T` with the given name and
/// width.
template <typename T>
auto buildIntFpDebugType(MLIRContext *ctx, StringRef name, unsigned width,
                         unsigned conservativeAlign) {
  // TODO: This should be driven by target info in the longer term.
  uint32_t align =
      llvm::PowerOf2Ceil(llvm::divideCeil(width, CHAR_BIT)) * CHAR_BIT;
  align = std::min(align, conservativeAlign);

  uint64_t size = llvm::alignTo(width, align);
  return T::get(ctx, name, size, align);
}

static DebugInfo::DIType
buildDebugTypeFromDType(MLIRContext *ctx, uint8_t dtype, unsigned indexWidth) {
  // Process various builtin dtypes.
  switch (dtype) {
  case DType::kBool:
    return buildIntFpDebugType<DebugInfo::DIBasicBoolType>(ctx, "bool", 8, 8);
  case DType::si1:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si1", 1, 8);
  case DType::ui1:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui1", 1, 8);
  case DType::si2:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si2", 2, 8);
  case DType::ui2:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui2", 2, 8);
  case DType::si4:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si4", 4, 8);
  case DType::ui4:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui4", 4, 8);
  case DType::si8:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si8", 8, 8);
  case DType::ui8:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui8", 8, 8);
  case DType::si16:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si16", 16, 16);
  case DType::ui16:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui16", 16, 16);
  case DType::si32:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si32", 32, 32);
  case DType::ui32:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui32", 32, 32);
  case DType::si64:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si64", 64, 64);
  case DType::ui64:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui64", 64, 64);
  case DType::si128:
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(ctx, "si128", 128,
                                                           64);
  case DType::ui128:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(ctx, "ui128", 128,
                                                           64);

  case DType::f8:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f8", 8, 8);
  case DType::f16:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f16", 16, 16);
  case DType::f32:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f32", 32, 32);
  case DType::f64:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f64", 64, 64);
  case DType::f128:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f128", 128,
                                                            64);
  case DType::bf16:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "bf16", 16,
                                                            16);
  case DType::f24:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f24", 24, 32);
  case DType::f80:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "f80", 80, 64);
  case DType::tf32:
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "tf32", 32,
                                                            32);

  case KGENDType::index:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(
        ctx, "index", indexWidth, indexWidth);

    // TODO: Process the remaining dtypes.
  default:
    return nullptr;
  }
}

POPToLLVMDebugInfoTypeConverter::POPToLLVMDebugInfoTypeConverter(
    POPToLLVMTypeConverter &converter) {
  // Let the LLVM conversion handle a majority of the debug info generation.
  addUnresolvedConverter(converter);

  // Add direct debug info conversions.
  addConversion([&](POP::SIMDType type) -> std::optional<Type> {
    // We can only build debug info if the dtype and size have been resolved.
    std::optional<KGENDType> dtype = type.getResolvedDType();
    std::optional<int64_t> size = type.getResolvedSize();
    if (!dtype || !size)
      return std::nullopt;

    // Get the base debug type from the dtype.
    DebugInfo::DIType baseType = buildDebugTypeFromDType(
        type.getContext(), dtype->getValue(), converter.getIndexTypeBitwidth());
    if (!baseType)
      return std::nullopt;

    // Single element SIMD becomes a scalar, multi-element become vectors.
    if (*size == 1)
      return baseType;
    return DebugInfo::DIVectorType::get(baseType, *size);
  });

  // TODO: Add debug generation for variant and dtype.
}
