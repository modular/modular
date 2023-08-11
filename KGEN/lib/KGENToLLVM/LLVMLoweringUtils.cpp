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
#include "mlir/Dialect/Index/IR/IndexOps.h"
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
// LLVMDataLayout
//===----------------------------------------------------------------------===//

int64_t LLVMDataLayout::getTypeSizeInBits(Type type) const {
  assert(LLVM::isCompatibleType(type) && "expected an LLVM type");
  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();
  if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(type)) {
    assert(ptrType.getAddressSpace() == 0 &&
           "only default address space supported");
    return target.getDataLayout().getPointerBitWidth();
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    return target.getDataLayout().getVectorBitWidth(
        vecType.getNumElements(), vecType.getElementTypeBitWidth());
  }
  if (auto vecType = dyn_cast<LLVM::LLVMFixedVectorType>(type)) {
    return target.getDataLayout().getVectorBitWidth(
        vecType.getNumElements(), getTypeSizeInBits(vecType.getElementType()));
  }
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(type)) {
    return arrayType.getNumElements() *
           getTypeStoreSize(arrayType.getElementType()) * CHAR_BIT;
  }
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
    int64_t size = 0;
    int64_t strictest = 1;
    for (Type type : structType.getBody()) {
      int64_t eltABIAlign = getTypeABIAlign(type);
      size = llvm::alignTo(size, eltABIAlign) + getTypeAllocSize(type);
      strictest = std::max(strictest, eltABIAlign);
    }
    return llvm::alignTo(size, strictest) * CHAR_BIT;
  }
  llvm::report_fatal_error("unsupported LLVM dialect type");
}

int64_t LLVMDataLayout::getTypeABIAlign(Type type) const {
  assert(LLVM::isCompatibleType(type) && "expected an LLVM type");
  if (auto intType = dyn_cast<IntegerType>(type))
    return target.getDataLayout().getIntegerABIAlign(intType.getWidth());
  if (auto fpType = dyn_cast<FloatType>(type))
    return target.getDataLayout().getFloatABIAlign(fpType.getWidth());
  if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(type)) {
    assert(ptrType.getAddressSpace() == 0 &&
           "only default address space supported");
    return target.getDataLayout().getPointerABIAlign();
  }
  if (auto vecType = dyn_cast<VectorType>(type)) {
    return target.getDataLayout().getVectorABIAlign(
        vecType.getNumElements(), vecType.getElementTypeBitWidth());
  }
  if (auto vecType = dyn_cast<LLVM::LLVMFixedVectorType>(type)) {
    return target.getDataLayout().getVectorABIAlign(
        vecType.getNumElements(), getTypeSizeInBits(vecType.getElementType()));
  }
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(type))
    return getTypeABIAlign(arrayType.getElementType());
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
    int64_t strictest = 1;
    for (Type type : structType.getBody())
      strictest = std::max(strictest, getTypeABIAlign(type));
    return strictest;
  }
  llvm::report_fatal_error("unsupported LLVM dialect type");
}

//===----------------------------------------------------------------------===//
// TargetInfoAttr
//===----------------------------------------------------------------------===//

ArrayAttr KGEN::attachTargetPassthroughAttrs(OpBuilder &b,
                                             TargetInfoAttr target,
                                             ArrayAttr passthrough) {
  SmallVector<Attribute> attrs;
  if (passthrough)
    llvm::append_range(attrs, passthrough);
  // Attach the target info attributes.
  attrs.push_back(b.getArrayAttr(
      {b.getStringAttr("target-cpu"), b.getStringAttr(target.getCpu())}));
  attrs.push_back(b.getArrayAttr({b.getStringAttr("target-features"),
                                  b.getStringAttr(target.getFeatures())}));
  if (!target.getTuneCpu().empty())
    attrs.push_back(b.getArrayAttr(
        {b.getStringAttr("tune-cpu"), b.getStringAttr(target.getTuneCpu())}));
  return b.getArrayAttr(attrs);
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

/// Build LLVM lowering options for a target.
static mlir::LowerToLLVMOptions buildLLVMLoweringOpts(TargetInfoAttr target) {
  mlir::LowerToLLVMOptions opts(target.getContext());
  opts.overrideIndexBitwidth(target.getDataLayout().getPointerBitWidth());
  opts.dataLayout.reset(target.getDataLayout().toString());
  opts.useOpaquePointers = false;
  return opts;
}

POPToLLVMTypeConverter::POPToLLVMTypeConverter(TargetInfoAttr target)
    : LLVMTypeConverter(target.getContext(), buildLLVMLoweringOpts(target)),
      LLVMDataLayout(target) {

  // Convert pointer types to LLVM pointer types. If the element type is
  // unspecified, return an opaque pointer.
  addConversion([=](POP::PointerType pointer) -> std::optional<Type> {
    unsigned addressSpace =
        cast<IntegerAttr>(pointer.getAddressSpace()).getInt();
    if (Type elementType = convertType(pointer.getElementAsType()))
      return LLVM::LLVMPointerType::get(elementType, addressSpace);
    return LLVM::LLVMPointerType::get(pointer.getContext(), addressSpace);
  });

  // Convert array types to LLVM array types.
  addConversion([=](POP::ArrayType array) -> std::optional<Type> {
    std::optional<int64_t> size = array.getResolvedSize();
    if (!size)
      return {};
    Type elementType = convertType(array.getElementAsType());
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
  auto convertElementTypesToStruct =
      [=](ArrayRef<TypedAttr> elements) -> std::optional<Type> {
    SmallVector<Type> types;
    types.reserve(elements.size());
    for (TypedAttr elementType : elements) {
      auto constant = dyn_cast<ConcreteTypeConstantAttr>(elementType);
      if (!constant)
        return {};
      Type converted = convertType(constant.getValue());
      if (!converted)
        return {};
      types.push_back(converted);
    }
    return LLVM::LLVMStructType::getLiteral(&getContext(), types);
  };

  addConversion([=](POP::StructType structType) -> std::optional<Type> {
    return convertElementTypesToStruct(structType.getElementTypes());
  });

  // Packs are essentially identical to structs.
  addConversion([=](POP::PackType type) -> std::optional<Type> {
    auto variadic = type.getVariadicAttr();
    if (!variadic)
      return {};
    return convertElementTypesToStruct(variadic.getValues());
  });

  // Convert closure type to a struct of two pointers
  addConversion([=](POP::ClosureType closureType) -> std::optional<Type> {
    MLIRContext *ctx = closureType.getContext();
    auto pointerTy = LLVM::LLVMPointerType::get(ctx);
    return LLVM::LLVMStructType::getLiteral(ctx, {pointerTy, pointerTy});
  });

  // Convert SIMD types to vector types.
  addConversion([=](POP::SIMDType simd) -> std::optional<Type> {
    std::optional<KGENDType> dtype = simd.getResolvedDType();
    std::optional<uint64_t> size = simd.getResolvedSize();
    if (!dtype || !size)
      return {};
    std::optional<Type> type = getMLIRTypeForDType(
        simd.getContext(), *dtype, getOptions().getIndexBitwidth());
    if (!type)
      return {};

    // Scalar case, size = 1
    if (*size == 1)
      return *type;

    // Vector case, size != 1
    return LLVM::getFixedVectorType(*type, *size);
  });

  // Convert data type types to `i8`.
  addConversion([=](DTypeType dtype) -> std::optional<Type> {
    return Builder(&getContext()).getI8Type();
  });

  addConversion([=](SignatureType signatureType) -> std::optional<Type> {
    MLIRContext *ctx = signatureType.getContext();
    if (signatureType.isCapturing()) {
      auto pointerTy = LLVM::LLVMPointerType::get(ctx);
      return LLVM::LLVMStructType::getLiteral(ctx, {pointerTy, pointerTy});
    } else {
      return convertType(signatureType.getValues());
    }
  });

  // Convert variant types to a struct with enough space to contain the largest
  // variant type plus a discriminator.
  addConversion([=](POP::VariantType variant) -> std::optional<Type> {
    // TODO: The generated assembly is sensitive to the content type of the
    // variant type. This needs to be optimized. For now, use an array of
    // word-size integers.
    int64_t maxSize = 0;
    for (TypedAttr typeExpr : variant.getTypes()) {
      Type variantType = typeExpr.cast<ConcreteTypeConstantAttr>().getValue();
      Type type = convertType(variantType);
      if (!type)
        return {};
      maxSize = std::max(maxSize, getTypeAllocSize(type));
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

  // Variadic types are converted to a struct representing a pointer to the
  // elements of the sequence, and the sequence size.
  addConversion([=](KGEN::VariadicType variadic) -> std::optional<Type> {
    Type convertedType = convertType(variadic.getElementAsType());
    if (!convertedType)
      return {};

    return LLVM::LLVMStructType::getLiteral(
        &getContext(),
        {LLVM::LLVMPointerType::get(convertedType), getIndexType()});
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
                    dl.getTypeABIAlign(value.getType()));

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
  addStoragePadding(valueIt, storageOffset, offset, dl.getTypeABIAlign(type));

  // Given a leaf type, extract a value of that type from the current storage.
  if (isa<IntegerType, FloatType, LLVM::LLVMPointerType>(type)) {
    IntegerType normalizedType;
    if (auto intType = dyn_cast<IntegerType>(type))
      normalizedType = intType;
    else if (auto fpType = dyn_cast<FloatType>(type))
      normalizedType = b.getIntegerType(fpType.getWidth());
    else
      normalizedType = b.getIntegerType(dl.getTypeSizeInBits(type));

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
// Interpreter Memory Conversion
//===----------------------------------------------------------------------===//

Value InterpreterMemoryConverter::convertMemRef(ImplicitLocOpBuilder &b,
                                                MemRefAttr ref) {
  MaterializedBlobs &materialized = getOrMaterialize(b, ref.getMemory());
  Value ptr = getBlobPointer(b, LLVM::LLVMPointerType::get(b.getContext()),
                             materialized, ref.getIndex(), ref.getOffset());
  return b.create<LLVM::BitcastOp>(tc.convertType(ref.getType()), ptr);
}

Value InterpreterMemoryConverter::getBlobPointer(
    ImplicitLocOpBuilder &b, Type ptrType, MaterializedBlobs &materialized,
    int64_t index, int64_t offset) {
  PointerUnion<Operation *, Value> value = materialized[index];
  Value ptr = dyn_cast<Value>(value);
  if (!ptr) {
    ptr = b.create<LLVM::BitcastOp>(
        ptrType, b.create<LLVM::AddressOfOp>(
                     cast<LLVM::GlobalOp>(cast<Operation *>(value))));
  }
  return b.create<LLVM::GEPOp>(ptrType, b.getI8Type(), ptr,
                               LLVM::GEPArg(offset), /*inbounds=*/true);
}

Operation *InterpreterMemoryConverter::getOrCreateGlobal(Location loc,
                                                         MemoryHandle hdl) {
  // Lookup an existing global for this handle.
  if (Operation *global = globals.lookup(hdl.getKey()))
    return global;

  // If not, create it.
  OpBuilder b(loc.getContext());
  Attribute value;
  mlir::AsmResourceBlob *mem = hdl.getBlob();
  if (hdl.getResource()->getKind() ==
      DialectResourceManager::ResourceKind::String) {
    // Create a string attribute for readability.
    value = b.getStringAttr(
        StringRef(mem->getData().data(), mem->getData().size()));
  } else {
    // Store the raw bytes into an elements attribute.
    value = IntArrayElementsAttr::get(b.getContext(), mem->getData(),
                                      IntegerType::Signless);
  }

  auto global = b.create<LLVM::GlobalOp>(
      loc, LLVM::LLVMArrayType::get(b.getI8Type(), mem->getData().size()),
      /*isConstant=*/true, LLVM::Linkage::Internal, hdl.getKey(), value,
      mem->getDataAlignment());
  symtab.insert(global);
  globals.try_emplace(hdl.getKey(), global);
  return global;
}

InterpreterMemoryConverter::MaterializedBlobs &
InterpreterMemoryConverter::getOrMaterialize(ImplicitLocOpBuilder &b,
                                             MemorySpaceAttr space) {
  if (auto it = blobs.find(space); it != blobs.end())
    return it->second;

  MaterializedBlobs materialized;
  auto i8PtrType = LLVM::LLVMPointerType::get(b.getI8Type());
  auto ptrType = LLVM::LLVMPointerType::get(b.getContext());

  // First emit the allocations and the memcpy's.
  for (const MemoryBlob &blob : space) {
    if (blob.getKind() == MemoryKind::ConstGlobal) {
      materialized.emplace_back(
          getOrCreateGlobal(b.getLoc(), blob.getHandle()));
      continue;
    }
    // Create the relevant allocation.
    Value popAlloc;
    mlir::AsmResourceBlob *mem = blob.getHandle().getBlob();
    if (blob.getKind() == MemoryKind::Stack) {
      popAlloc = b.create<POP::StackAllocationOp>(
          POP::PointerType::get(b.getI8Type()), mem->getData().size(),
          b.getIndexAttr(mem->getDataAlignment()));
    } else {
      popAlloc = b.create<POP::AlignedAllocOp>(
          POP::PointerType::get(b.getI8Type()),
          b.create<mlir::index::ConstantOp>(mem->getDataAlignment()),
          b.create<mlir::index::ConstantOp>(mem->getData().size()));
    }
    Value ptr = b.create<mlir::UnrealizedConversionCastOp>(i8PtrType, popAlloc)
                    .getResult(0);
    materialized.emplace_back(Value(b.create<LLVM::BitcastOp>(ptrType, ptr)));
  }

  // Perform memcpy of non-global blobs while remapping pointer regions.
  int64_t pointerSize = tc.getTarget().getDataLayout().getPointerSize();
  for (auto [blob, value] : llvm::zip(space, materialized)) {
    // Constant globals don't have pointer regions.
    if (blob.getKind() == MemoryKind::ConstGlobal)
      continue;
    auto ptr = cast<Value>(value);
    mlir::AsmResourceBlob *mem = blob.getHandle().getBlob();
    ArrayRef<char> data = mem->getData();
    auto ptrIt = blob.getPointerRegions().begin();
    auto ptrEnd = blob.getPointerRegions().end();
    for (int64_t i = 0, e = data.size(); i != e;) {
      // GEP to the current offset.
      Value gep = b.create<LLVM::GEPOp>(ptrType, b.getI8Type(), ptr,
                                        LLVM::GEPArg(i), /*inbounds=*/true);
      // Check if the current offset is the beginning of a pointer region.
      if (ptrIt != ptrEnd && ptrIt->offset == i) {
        // Store the pointer value to the current offset.
        auto [_, index, offset] = *ptrIt++;
        b.create<LLVM::StoreOp>(
            getBlobPointer(b, ptrType, materialized, index, offset), gep,
            mem->getDataAlignment());
        i += pointerSize;
      } else {
        // Store the byte at this offset.
        // FIXME: Vectorize the stores to reduce IR bloat.
        b.create<LLVM::StoreOp>(
            b.create<LLVM::ConstantOp>(b.getI8Type(), data[i]), gep,
            mem->getDataAlignment());
        ++i;
      }
    }
  }

  return blobs.try_emplace(space, std::move(materialized)).first->second;
}

//===----------------------------------------------------------------------===//
// Attribute Conversion
//===----------------------------------------------------------------------===//

/// Convert a SIMD vector constant.
static Value convertSIMDAttr(ImplicitLocOpBuilder &b,
                             mlir::LLVMTypeConverter &tc, POP::SIMDAttr simd) {
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
      Value addr =
          asConst(b.getIntegerAttr(tc.getIndexType(), value.getIndexVal()));
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
    return asConst(cast<TypedAttr>(IntArrayElementsAttr::get(
        VectorType::get(values.size(), b.getI1Type()), values)));
  }
  if (dtype.isInt()) {
    SmallVector<APInt> values;
    for (const POP::DTypeValue &value : simd.getValues())
      values.push_back(value.getIntVal());
    return asConst(cast<TypedAttr>(IntArrayElementsAttr::get(
        VectorType::get(values.size(),
                        b.getIntegerType(dtype.getIntegerWidthInBits())),
        values)));
  }
  if (dtype.isIndex() || dtype.isAddress()) {
    SmallVector<APInt> values;
    auto indexType = cast<IntegerType>(tc.getIndexType());
    for (const POP::DTypeValue &value : simd.getValues())
      values.push_back(APInt(indexType.getWidth(), value.getIndexVal()));
    Value addr = asConst(cast<TypedAttr>(IntArrayElementsAttr::get(
        VectorType::get(values.size(), indexType), values)));
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
  return asConst(cast<TypedAttr>(FloatArrayElementsAttr::get(
      VectorType::get(values.size(),
                      getEquivalentFloatType(b.getContext(), dtype)),
      values)));
}

/// Lower the string to a pop.global_constant and create a llvm struct of type
/// !llvm.struct<(ptr<i8>, i64)> holding the pointer to the global string and
/// its size.
static Value lowerStringToGlobalConstant(StringAttr strAttr,
                                         ImplicitLocOpBuilder &b,
                                         POPToLLVMTypeConverter &tc,
                                         InterpreterMemoryConverter &imc) {
  StringRef strAttrRef = strAttr.getValue();
  // This is safe because StringAttr always stores a null terminator. If the
  // string is empty, we won't use this anyway.
  StringRef str(strAttrRef.data(), strAttrRef.size() + 1);
  if (strAttrRef.empty())
    str = "\0";

  // Add the string to the global string table.
  DialectResourceManager &mgr =
      MemoryHandle::getManagerInterface(strAttr.getContext());
  MemoryHandle hdl = mgr.getOrAddStringResource(str);
  auto global = cast<LLVM::GlobalOp>(imc.getOrCreateGlobal(b.getLoc(), hdl));

  // The actual string size does not include \0.
  auto sizeType = cast<IntegerType>(tc.getIndexType());
  Value sizeVal = b.create<LLVM::ConstantOp>(
      b.getLoc(), IntegerAttr::get(sizeType, strAttr.size()));
  Value undefOp = b.create<LLVM::UndefOp>(
      b.getLoc(), getLLVMTypeForKGENStringType(b.getContext(), sizeType));
  Value llvmString =
      b.create<LLVM::BitcastOp>(LLVM::LLVMPointerType::get(b.getI8Type()),
                                b.create<LLVM::AddressOfOp>(global));
  Value structVal0 =
      b.create<LLVM::InsertValueOp>(b.getLoc(), undefOp, llvmString, 0);
  return b.create<LLVM::InsertValueOp>(b.getLoc(), structVal0, sizeVal, 1);
}

Value KGEN::materializeLLVMStruct(ImplicitLocOpBuilder &b, Type structType,
                                  ValueRange elements) {
  Value container = b.create<LLVM::UndefOp>(structType);
  for (auto [index, element] : llvm::enumerate(elements))
    container = b.create<LLVM::InsertValueOp>(container, element, index);
  return container;
}

Value KGEN::convertParameterToLLVM(ImplicitLocOpBuilder &b,
                                   POPToLLVMTypeConverter &tc,
                                   SymbolTable &symtab,
                                   InterpreterMemoryConverter *imc,
                                   TypedAttr attr) {
  //===--------------------------------------------------------------------===//
  // Builtin

  // Truncate index constants if required.
  if (isa<IndexType>(attr.getType())) {
    return b.create<LLVM::ConstantOp>(b.getIntegerAttr(
        cast<IntegerType>(tc.getIndexType()),
        cast<IntegerAttr>(attr).getValue().trunc(tc.getIndexTypeBitwidth())));
  }

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

  // Convert pointer attributes (usually null pointers).
  if (auto ptr = dyn_cast<PointerAttr>(attr)) {
    return b.create<LLVM::IntToPtrOp>(
        tc.convertType(ptr.getType()),
        b.create<LLVM::ConstantOp>(
            b.getIntegerAttr(tc.getIndexType(), ptr.getAddr())));
  }

  // Materialize memrefs from the interpreter.
  if (imc)
    if (auto ref = dyn_cast<MemRefAttr>(attr))
      return imc->convertMemRef(b, ref);

  // Convert string constant to a struct{ptr, size} of type
  // !llvm.struct<(ptr<i8>, index).
  if (auto strAttr = dyn_cast<StringAttr>(attr))
    return lowerStringToGlobalConstant(strAttr, b, tc, *imc);

  //===--------------------------------------------------------------------===//
  // POP

  // Convert SIMD constants to an array of integer or float constants.
  if (auto simd = dyn_cast<POP::SIMDAttr>(attr))
    return convertSIMDAttr(b, tc, simd);

  // Convert array, struct, or pack constants to LLVM array or struct constants.
  if (isa<POP::ArrayAttr, POP::StructAttr, POP::PackAttr>(attr)) {
    Type type = tc.convertType(attr.getType());
    if (!type)
      return {};
    Value aggregate = b.create<LLVM::UndefOp>(type);
    ArrayRef<TypedAttr> values =
        TypeSwitch<Attribute, ArrayRef<TypedAttr>>(attr)
            .Case<POP::ArrayAttr, POP::StructAttr, POP::PackAttr>(
                [](auto attr) { return attr.getValues(); });

    for (auto [idx, value] : llvm::enumerate(values)) {
      Value element = convertParameterToLLVM(b, tc, symtab, imc, value);
      if (!element)
        return {};
      // If this is a struct with one element, return it directly.
      if (auto structAttr = dyn_cast<POP::StructAttr>(attr);
          structAttr && structAttr.getValues().size() == 1)
        return element;
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
    Value value =
        convertParameterToLLVM(b, tc, symtab, imc, variant.getValue());
    if (!value)
      return {};

    VariantHelper helper(b, b.getLoc(), tc);
    return helper.materializeLLVMVariant(
        variantType, value,
        *variant.getType().getTypeIndex(variant.getValue().getType()));
  }

  // Convert variadic sequence constants to an LLVM struct constant.
  if (auto variadic = dyn_cast<KGEN::VariadicAttr>(attr)) {
    // 1. Allocate space for an array of elements.
    Type elementType = tc.convertType(variadic.getType().getElementAsType());
    if (!elementType)
      return {};

    Value size = b.create<LLVM::ConstantOp>(
        b.getI64IntegerAttr(variadic.getValues().size()));
    Value ptr = b.create<LLVM::AllocaOp>(
        LLVM::LLVMPointerType::get(elementType), elementType, size);

    // 2. Store elements of the sequence into the allocated space.
    for (auto [idx, value] : llvm::enumerate(variadic.getValues())) {
      Value element = convertParameterToLLVM(b, tc, symtab, imc, value);
      if (!element)
        return {};

      Value destination =
          b.create<LLVM::GEPOp>(LLVM::LLVMPointerType::get(elementType), ptr,
                                ArrayRef<LLVM::GEPArg>{idx});
      b.create<LLVM::StoreOp>(element, destination);
    }

    // 3. Create a struct with a pointer to the allocation & the sequence size.
    auto variadicType = llvm::cast_if_present<LLVM::LLVMStructType>(
        tc.convertType(variadic.getType()));
    if (!variadicType)
      return {};

    return materializeLLVMStruct(b, variadicType, ValueRange{ptr, size});
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
buildDebugTypeFromIntOrIndexOrFloatType(MLIRContext *ctx, Type type,
                                        POPToLLVMTypeConverter &converter,
                                        TargetInfoAttr target) {
  if (type.isIndex())
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(
        ctx, "index", converter.getIndexTypeBitwidth(),
        converter.getIndexTypeBitwidth());
  if (!type.isIntOrIndexOrFloat())
    return DebugInfo::DIUnresolvedMLIRType::get(ctx, type);

  uint64_t sizeInBits = type.getIntOrFloatBitWidth();
  int64_t alignInBits = *DataLayoutInterface::getTypeABIAlign(target, type);

  if (type.isUnsignedInteger())
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(
        ctx, "ui" + std::to_string(sizeInBits), sizeInBits, alignInBits);

  if (type.isSignedInteger())
    return buildIntFpDebugType<DebugInfo::DIBasicSIntType>(
        ctx, "si" + std::to_string(sizeInBits), sizeInBits, alignInBits);

  if (type.isSignlessInteger())
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(
        ctx, "i" + std::to_string(sizeInBits), sizeInBits, alignInBits);

  if (type.isBF16())
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(ctx, "bf16", 16,
                                                            16);

  if (isa<FloatType>(type))
    return buildIntFpDebugType<DebugInfo::DIBasicFloatType>(
        ctx, "f" + std::to_string(sizeInBits), sizeInBits, alignInBits);

  llvm_unreachable(
      "Can't build DebugType from types that are not IntOrIndexOrFloat.");
}

static DebugInfo::DIType
buildDebugTypeFromDType(MLIRContext *ctx, uint8_t dtype, size_t indexWidth) {
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

  case DType::invalid:
    return DebugInfo::DIUnspecifiedType::get(ctx, "void");

  case KGENDType::index:
    return buildIntFpDebugType<DebugInfo::DIBasicUIntType>(
        ctx, "index", indexWidth, indexWidth);

    // TODO: Process the remaining dtypes.
  default:
    return nullptr;
  }
}

static DebugInfo::DIType
buildDebugTypeFromPOPType(MLIRContext *ctx, Type type,
                          POPToLLVMTypeConverter &converter,
                          TargetInfoAttr target);

static DebugInfo::DIType
buildDebugTypeFromFunctionType(MLIRContext *ctx, FunctionType type,
                               POPToLLVMTypeConverter &converter,
                               TargetInfoAttr target) {
  SmallVector<DebugInfo::DIType> argTypes;
  for (Type arg : type.getInputs()) {
    DebugInfo::DIType argDIType =
        buildDebugTypeFromPOPType(ctx, arg, converter, target);
    argTypes.push_back(argDIType);
  }

  SmallVector<DebugInfo::DIType> resultTypes;
  for (Type result : type.getResults()) {
    DebugInfo::DIType resultDIType =
        buildDebugTypeFromPOPType(ctx, result, converter, target);
    resultTypes.push_back(resultDIType);
  }
  return DebugInfo::DISubroutineType::get(ctx, argTypes, resultTypes);
}

static DebugInfo::DIType
buildDebugStructTypeFromTypeAttrs(MLIRContext *ctx, ArrayRef<TypedAttr> attrs,
                                  POPToLLVMTypeConverter &converter,
                                  StringAttr name, TargetInfoAttr target) {
  SmallVector<DebugInfo::DIMemberType> elementTypes;
  for (auto [idx, attr] : llvm::enumerate(attrs)) {
    DebugInfo::DIType mDIType = buildDebugTypeFromPOPType(
        ctx, cast<TypeConstantAttr>(attr).getValue(), converter, target);
    DebugInfo::DIMemberType mMemberDIType = DebugInfo::DIMemberType::get(
        StringAttr::get(ctx, "m" + std::to_string(idx)), mDIType);

    elementTypes.push_back(mMemberDIType);
  }
  return DebugInfo::DIStructType::get(name, elementTypes);
}

static DebugInfo::DIType
buildDebugTypeFromPOPType(MLIRContext *ctx, Type type,
                          POPToLLVMTypeConverter &converter,
                          TargetInfoAttr target) {
  if (auto arrayType = dyn_cast<POP::ArrayType>(type)) {
    int64_t size = *arrayType.getResolvedSize();
    DebugInfo::DIType elementType = buildDebugTypeFromPOPType(
        ctx, arrayType.getElementAsType(), converter, target);
    return DebugInfo::DIArrayType::get(elementType, size);
  }

  if (auto closureType = dyn_cast<POP::ClosureType>(type)) {
    return buildDebugTypeFromFunctionType(ctx, closureType.getFunc(), converter,
                                          target);
  }

  if (auto coroutineType = dyn_cast<POP::CoroutineType>(type)) {
    // We map coroutine types to pointers to subroutine types.
    DebugInfo::DIType srType = buildDebugTypeFromFunctionType(
        ctx, coroutineType.getSignature().getValues(), converter, target);
    return DebugInfo::DIPointerType::get(srType, converter.getPointerBitwidth(),
                                         converter.getPointerBitwidth());
  }

  if (auto packType = dyn_cast<POP::PackType>(type)) {
    return buildDebugStructTypeFromTypeAttrs(
        ctx, packType.getVariadicAttr().getValues(), converter,
        StringAttr::get(ctx, "pack"), target);
  }

  if (auto pointerType = dyn_cast<POP::PointerType>(type)) {
    DebugInfo::DIType elementDIType = buildDebugTypeFromPOPType(
        ctx, pointerType.getElementAsType(), converter, target);
    return DebugInfo::DIPointerType::get(elementDIType,
                                         converter.getPointerBitwidth(),
                                         converter.getPointerBitwidth());
  }

  if (auto simdType = dyn_cast<POP::SIMDType>(type)) {
    int64_t size = *simdType.getResolvedSize();
    DebugInfo::DIType baseType =
        buildDebugTypeFromDType(ctx, simdType.getResolvedDType()->getValue(),
                                converter.getIndexTypeBitwidth());

    if (size == 1)
      return baseType;
    return DebugInfo::DIVectorType::get(baseType, size);
  }

  if (auto structType = dyn_cast<POP::StructType>(type)) {
    return buildDebugStructTypeFromTypeAttrs(
        ctx, structType.getElementTypes(), converter,
        StringAttr::get(ctx, "struct"), target);
  }

  // TODO: Add POP::VariantType DebugInfo conversion with union like
  // DebugInfoType
  if (type.isIntOrIndexOrFloat()) {
    return buildDebugTypeFromIntOrIndexOrFloatType(ctx, type, converter,
                                                   target);
  }

  return DebugInfo::DIUnresolvedMLIRType::get(type);
}

POPToLLVMDebugInfoTypeConverter::POPToLLVMDebugInfoTypeConverter(
    POPToLLVMTypeConverter &converter, TargetInfoAttr target) {
  // Let the LLVM conversion handle a majority of the debug info generation.
  addUnresolvedConverter(converter);

  // Add direct debug info conversions.
  addConversion([&converter, target](POP::ArrayType type) {
    return buildDebugTypeFromPOPType(type.getContext(), type, converter,
                                     target);
  });

  addConversion([&converter, target](POP::ClosureType type) {
    return buildDebugTypeFromPOPType(type.getContext(), type, converter,
                                     target);
  });

  addConversion([&converter, target](POP::CoroutineType type) {
    return buildDebugTypeFromPOPType(type.getContext(), type, converter,
                                     target);
  });

  addConversion([&converter, target](POP::PackType type) {
    return buildDebugTypeFromPOPType(type.getContext(), type, converter,
                                     target);
  });

  addConversion([&converter, target](POP::PointerType type) {
    return buildDebugTypeFromPOPType(type.getContext(), type, converter,
                                     target);
  });

  addConversion([&converter, target](POP::SIMDType type) {
    return buildDebugTypeFromPOPType(type.getContext(), type, converter,
                                     target);
  });

  addConversion([&converter, target](POP::StructType type) {
    return buildDebugTypeFromPOPType(type.getContext(), type, converter,
                                     target);
  });

  addConversion([&converter, target](POP::VariantType type) {
    return buildDebugTypeFromPOPType(type.getContext(), type, converter,
                                     target);
  });
}
