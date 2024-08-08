//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLVMLoweringUtils.h"
#include "KGEN/CODialect/COOps.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
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
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/BLAKE3.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

/// Since !kgen.string type is not parameterized on the size of the string,
/// we lower it as struct with a pointer field holding the data and a index
/// field holding the string size.
static Type getLLVMTypeForKGENStringType(MLIRContext *ctx, Type strSizeType) {
  SmallVector<Type> elementTypes{LLVM::LLVMPointerType::get(ctx), strSizeType};
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
  if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(type))
    return target.getDataLayout().getPointerABIAlign();

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
      {b.getStringAttr("target-cpu"), b.getStringAttr(target.getArch())}));
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

/// Build LLVM lowering options for a target.
static mlir::LowerToLLVMOptions buildLLVMLoweringOpts(TargetInfoAttr target) {
  mlir::LowerToLLVMOptions opts(target.getContext());
  opts.overrideIndexBitwidth(target.getDataLayout().getPointerBitWidth());
  opts.dataLayout.reset(target.getDataLayout().toString());
  return opts;
}

POPToLLVMTypeConverter::POPToLLVMTypeConverter(TargetInfoAttr target)
    : LLVMTypeConverter(target.getContext(), buildLLVMLoweringOpts(target)),
      LLVMDataLayout(target) {

  //===--------------------------------------------------------------------===//
  // KGEN

  // Convert `!kgen.none` to an empty struct.
  addConversion([=](KGEN::NoneType) {
    return LLVM::LLVMStructType::getLiteral(&getContext(), {});
  });

  // Convert string types to LLVM literal structs: struct{ptr, size} of type
  // !llvm.struct<(ptr<i8>, index).
  addConversion([=](KGEN::StringType stringType) -> std::optional<Type> {
    return getLLVMTypeForKGENStringType(stringType.getContext(),
                                        getIndexType());
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

  // Variadic types are converted to a struct representing a pointer to the
  // elements of the sequence, and the sequence size.
  addConversion([=](VariadicType variadic) -> std::optional<Type> {
    Type convertedType = convertType(variadic.getElementType());
    if (!convertedType)
      return {};

    return LLVM::LLVMStructType::getLiteral(
        &getContext(),
        {LLVM::LLVMPointerType::get(variadic.getContext()), getIndexType()});
  });

  // Convert pointer types to LLVM pointer types.
  addConversion([=](PointerType pointer) -> std::optional<Type> {
    unsigned addressSpace =
        cast<IntegerAttr>(pointer.getAddressSpace()).getInt();
    return LLVM::LLVMPointerType::get(pointer.getContext(), addressSpace);
  });

  //===--------------------------------------------------------------------===//
  // POP

  // Convert array types to LLVM array types.
  addConversion([=](POP::ArrayType array) -> std::optional<Type> {
    std::optional<int64_t> size = array.getResolvedSize();
    if (!size)
      return {};
    Type elementType = convertType(array.getElementType());
    if (!elementType)
      return {};
    return LLVM::LLVMArrayType::get(elementType, *size);
  });

  // Convert struct types to LLVM literal structs.
  addConversion([=](StructType structType) -> std::optional<Type> {
    SmallVector<Type> types;
    for (Type type : structType.getElementTypes()) {
      types.push_back(convertType(type));
      if (!types.back())
        return {};
    }
    return LLVM::LLVMStructType::getLiteral(&getContext(), types);
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

  // Convert union types to an array with enough space to contain the largest
  // union element type.
  addConversion([=](POP::UnionType unionType) -> std::optional<Type> {
    // TODO: The generated assembly is sensitive to the content type of the
    // union type. This needs to be optimized. For now, use an array of
    // word-size integers.
    int64_t maxSize = 0;
    for (Type unionType : unionType.getTypes()) {
      Type type = convertType(unionType);
      if (!type)
        return {};
      maxSize = std::max(maxSize, getTypeAllocSize(type));
    }
    // FIXME: The alignment of the generated type must equal or exceed the
    // greatest alignment requirement of any subtype. Right now it's just the
    // pointer width.
    return LLVM::LLVMArrayType::get(
        getIndexType(),
        llvm::divideCeil(maxSize * CHAR_BIT, getIndexTypeBitwidth()));
  });

  // Coroutine handles are always lowered to opaque pointers.
  addConversion([](CO::CoroutineType coro) {
    return LLVM::LLVMPointerType::get(coro.getContext());
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

Value VariantHelper::materializeLLVMUnion(mlir::LLVM::LLVMArrayType type,
                                          Value value) {
  SmallVector<Value> storageValues;
  for (unsigned i = 0, e = type.getNumElements(); i < e; ++i)
    storageValues.push_back(
        b.create<LLVM::ConstantOp>(type.getElementType(), 0));

  MutableArrayRef<Value>::iterator valueIt = storageValues.begin();
  unsigned storageOffset = 0;
  unsigned offset = 0;
  walkAndCreateVariant(valueIt, storageOffset, offset, value);

  Value content = b.create<LLVM::UndefOp>(type);
  for (auto [idx, value] : llvm::enumerate(storageValues))
    content = b.create<LLVM::InsertValueOp>(content, value, idx);
  return content;
}

//===----------------------------------------------------------------------===//
// Interpreter Memory Conversion
//===----------------------------------------------------------------------===//

Value InterpreterMemoryConverter::MaterializationScope::convertMemRef(
    ImplicitLocOpBuilder &b, MemRefAttr ref) {
  MaterializedBlobs &materialized = getOrMaterialize(b, ref.getMemory());
  Value ptr = getBlobPointer(b, LLVM::LLVMPointerType::get(b.getContext()),
                             materialized, ref.getIndex(), ref.getOffset());
  return b.create<LLVM::BitcastOp>(imc.tc.convertType(ref.getType()), ptr);
}

Value InterpreterMemoryConverter::MaterializationScope::getBlobPointer(
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
                                                         MemoryHandleAttr hdl) {
  // Lookup an existing global for this handle.
  if (Operation *global = globals.lookup(hdl))
    return global;

  // If not, create it.
  OpBuilder b(loc.getContext());
  Attribute value;
  if (hdl.isString()) {
    // Create a string attribute for readability.
    value = b.getStringAttr(StringRef(hdl.getData(), hdl.getSize()));
  } else {
    // Store the raw bytes into an elements attribute.
    value = IntArrayElementsAttr::get(b.getContext(), hdl.getMemory().data,
                                      IntegerType::Signless);
  }

  auto hash =
      llvm::BLAKE3::hash({(const uint8_t *)hdl.getData(), hdl.getSize()});
  std::string key = (hdl.isString() ? "static_string_" : "memory_blob_") +
                    llvm::toHex(hash, /*LowerCase=*/true);

  auto global = b.create<LLVM::GlobalOp>(
      loc, LLVM::LLVMArrayType::get(b.getI8Type(), hdl.getSize()),
      /*isConstant=*/true, LLVM::Linkage::Internal, key, value, hdl.getAlign());
  symtab.insert(global);
  globals.try_emplace(hdl, global);
  return global;
}

/// Store `size` worth of data to offset `idx` into `ptr`, reading from `data`.
/// Use vector stores, and progressively smaller ones if `size` is not a
/// multiple of 2.
static void materializeVectorStores(int64_t idx, int64_t size, Value ptr,
                                    const char *data, ImplicitLocOpBuilder &b,
                                    Type ptrType, size_t align) {
  // Nothing to do.
  if (size == 0)
    return;

  // GEP to the current offset.
  Value gep = b.create<LLVM::GEPOp>(ptrType, b.getI8Type(), ptr,
                                    LLVM::GEPArg(idx), /*inbounds=*/true);
  // Emit a scalar store.
  if (size == 1) {
    b.create<LLVM::StoreOp>(
        b.create<LLVM::ConstantOp>(b.getI8Type(), data[idx]), gep, align);
    return;
  }

  // Round down to the nearest power of 2, inclusive.
  int64_t curSize = llvm::NextPowerOf2(size) / 2;
  int64_t remaining = size - curSize;

  ArrayRef<uint8_t> slice((const uint8_t *)data + idx, curSize);
  auto value =
      ArrayElementsAttr::get(slice, VectorType::get(curSize, b.getI8Type()));
  b.create<LLVM::StoreOp>(b.create<LLVM::ConstantOp>(value), gep, align);
  materializeVectorStores(idx + curSize, remaining, ptr, data, b, ptrType,
                          align);
}

InterpreterMemoryConverter::MaterializedBlobs &
InterpreterMemoryConverter::MaterializationScope::getOrMaterialize(
    ImplicitLocOpBuilder &b, MemorySpaceAttr space) {
  if (auto it = blobs.find(space); it != blobs.end())
    return it->second;

  MaterializedBlobs materialized;
  auto ptrType = LLVM::LLVMPointerType::get(b.getContext());

  // First emit the allocations and the memcpy's.
  for (MemoryBlobAttr blob : space) {
    if (blob.getKind() == MemoryKind::ConstGlobal) {
      materialized.emplace_back(
          imc.getOrCreateGlobal(b.getLoc(), blob.getHandle()));
      continue;
    }
    // Create the relevant allocation.
    Value popAlloc;
    MemoryHandleAttr hdl = blob.getHandle();
    if (blob.getKind() == MemoryKind::Stack ||
        // FIXME(#32052): Persistent memory requires planning, but downcast to a
        // stack allocation for now.
        blob.getKind() == MemoryKind::Persistent) {
      popAlloc = b.create<POP::StackAllocationOp>(
          PointerType::get(b.getI8Type()), hdl.getSize(),
          b.getIndexAttr(hdl.getAlign()));
    } else {
      popAlloc = b.create<POP::AlignedAllocOp>(
          PointerType::get(b.getI8Type()),
          b.create<mlir::index::ConstantOp>(hdl.getAlign()),
          b.create<mlir::index::ConstantOp>(hdl.getSize()));
    }
    Value ptr = b.create<mlir::UnrealizedConversionCastOp>(ptrType, popAlloc)
                    .getResult(0);
    materialized.emplace_back(Value(b.create<LLVM::BitcastOp>(ptrType, ptr)));
  }

  // Perform memcpy of non-global blobs while remapping pointer regions.
  int64_t pointerSize = imc.tc.getTarget().getDataLayout().getPointerSize();
  int64_t simdWidth = imc.tc.getTarget().getSimdBitWidth() / 8;
  for (auto [blob, value] : llvm::zip(space, materialized)) {
    // Constant globals don't have pointer regions.
    if (blob.getKind() == MemoryKind::ConstGlobal)
      continue;

    auto ptr = cast<Value>(value);
    MemoryHandleAttr hdl = blob.getHandle();
    ArrayRef<char> data = hdl.getMemory().data;
    auto materializeStoreImpl = [&, align = hdl.getAlign()](int64_t idx,
                                                            int64_t size) {
      materializeVectorStores(idx, size, ptr, data.data(), b, ptrType, align);
    };

    // For large, contiguous chunks of memory with the same byte value,
    // "compress" the generated IR by emitting a memset instead of a huge number
    // of SIMD stores. This struct tracks the current compression state, which
    // has to be "comitted". This will prevent large materialized blobs from
    // destroying compile time. However, if the user fills a large blob with
    // "random" data, not much can be done.
    struct CompressionState {
      int64_t startIdx;
      char value;
      int64_t numReps;
    };
    std::optional<CompressionState> compressionState;
    auto commitCompressedStores = [&] {
      if (!compressionState)
        return;
      auto [startIdx, value, numReps] = std::move(*compressionState);
      compressionState.reset();
      // Simple heuristic: if the compressed size is more than 8 times the
      // preferred SIMD width, then use a memset instead.
      if (numReps <= 8 * simdWidth) {
        for (int64_t i = 0; i < numReps; i += simdWidth)
          materializeStoreImpl(startIdx + i, std::min(simdWidth, numReps - i));
        return;
      }
      // Emit a memset.
      Value gep =
          b.create<LLVM::GEPOp>(ptrType, b.getI8Type(), ptr,
                                LLVM::GEPArg(startIdx), /*inbounds=*/true);
      b.create<LLVM::MemsetOp>(
          gep, b.create<LLVM::ConstantOp>(b.getI8Type(), value),
          b.create<LLVM::ConstantOp>(b.getI64Type(), numReps),
          /*isVolatile=*/false);
    };

    auto materializeStore = [&](int64_t idx, int64_t size) {
      if (size == 0)
        return;
      if (!llvm::all_equal(data.slice(idx, size))) {
        // No compression possible. Commit the current state and materialize the
        // next chunk.
        commitCompressedStores();
        materializeStoreImpl(idx, size);
      } else if (!compressionState) {
        // Start tracking a new compression state.
        compressionState = CompressionState{idx, data[idx], size};
      } else if (compressionState->value == data[idx]) {
        // Increase the size of the compressed chunk.
        compressionState->numReps += size;
      } else {
        // Commit the previous state.
        commitCompressedStores();
        compressionState = CompressionState{idx, data[idx], size};
      }
    };

    // Store the memory blob in chunks of the preferred SIMD width.
    auto ptrIt = blob.getPointerRegions().begin();
    auto ptrEnd = blob.getPointerRegions().end();
    for (int64_t i = 0, e = data.size(); i < e;) {
      // Check if the current chunk contains a pointer.
      if (ptrIt != ptrEnd && i <= ptrIt->offset &&
          ptrIt->offset < (i + simdWidth)) {
        // Store up to the pointer region.
        int64_t partSize = ptrIt->offset - i;
        materializeStore(i, partSize);
        i += partSize;

        // Store the pointer value to the current offset.
        commitCompressedStores();
        Value gep = b.create<LLVM::GEPOp>(ptrType, b.getI8Type(), ptr,
                                          LLVM::GEPArg(i), /*inbounds=*/true);
        auto [_, index, offset] = *ptrIt++;
        b.create<LLVM::StoreOp>(
            getBlobPointer(b, ptrType, materialized, index, offset), gep,
            hdl.getAlign());
        i += pointerSize;
        continue;
      }
      materializeStore(i, std::min(simdWidth, e - i));
      i += simdWidth;
    }
    commitCompressedStores();
  }

  return blobs.try_emplace(space, std::move(materialized)).first->second;
}

//===----------------------------------------------------------------------===//
// Attribute Conversion
//===----------------------------------------------------------------------===//

/// Convert a SIMD vector constant.
static Value convertSIMDAttr(ImplicitLocOpBuilder &b,
                             const mlir::LLVMTypeConverter &tc,
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
                                         const POPToLLVMTypeConverter &tc,
                                         InterpreterMemoryConverter &imc) {
  StringRef strAttrRef = strAttr.getValue();
  // This is safe because StringAttr always stores a null terminator. If the
  // string is empty, we won't use this anyway.
  StringRef str(strAttrRef.data(), strAttrRef.size() + 1);
  if (strAttrRef.empty())
    str = "\0";

  // Add the string to the global string table.
  MemoryHandleAttr hdl = MemoryHandleAttr::get(strAttr.getContext(), str);
  auto global = cast<LLVM::GlobalOp>(imc.getOrCreateGlobal(b.getLoc(), hdl));

  // The actual string size does not include \0.
  auto sizeType = cast<IntegerType>(tc.getIndexType());
  Value sizeVal = b.create<LLVM::ConstantOp>(
      b.getLoc(), IntegerAttr::get(sizeType, strAttr.size()));
  Value undefOp = b.create<LLVM::UndefOp>(
      b.getLoc(), getLLVMTypeForKGENStringType(b.getContext(), sizeType));
  Value llvmString =
      b.create<LLVM::BitcastOp>(LLVM::LLVMPointerType::get(b.getContext()),
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

Value KGEN::convertParameterToLLVM(
    ImplicitLocOpBuilder &b, const POPToLLVMTypeConverter &tc,
    InterpreterMemoryConverter *imc,
    InterpreterMemoryConverter::MaterializationScope *scope, TypedAttr attr) {
  //===--------------------------------------------------------------------===//
  // builtin

  // Convert unknown values to undef.
  if (isa<UnknownAttr>(attr)) {
    Type type = tc.convertType(attr.getType());
    if (!type)
      return {};
    return b.create<LLVM::UndefOp>(type);
  }

  if (auto intCst = dyn_cast<IntegerAttr>(attr)) {
    // Check for index types a truncate index constants if required.
    if (isa<IndexType>(attr.getType())) {
      return b.create<LLVM::ConstantOp>(
          b.getIntegerAttr(cast<IntegerType>(tc.getIndexType()),
                           intCst.getValue().trunc(tc.getIndexTypeBitwidth())));
    }

    // Drop the sign on integer attributes; LLVM is signless.
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

  // Convert `#kgen.none` to an empty struct.
  if (isa<NoneAttr>(attr))
    return b.create<LLVM::UndefOp>(
        LLVM::LLVMStructType::getLiteral(b.getContext(), {}));

  // Convert pointer attributes (usually null pointers).
  if (auto ptr = dyn_cast<PointerAttr>(attr)) {
    return b.create<LLVM::IntToPtrOp>(
        tc.convertType(ptr.getType()),
        b.create<LLVM::ConstantOp>(
            b.getIntegerAttr(tc.getIndexType(), ptr.getAddr())));
  }

  // We can lower `StoreToMemAttr` by writing the underlying value into a
  // stack allocation.
  if (auto store = dyn_cast<StoreToMemAttr>(attr)) {
    Value value = convertParameterToLLVM(b, tc, imc, scope, store.getValue());
    if (!value)
      return {};
    unsigned align = tc.getTypeABIAlign(value.getType());
    Value ptr = b.create<LLVM::AllocaOp>(
        tc.convertType(attr.getType()), value.getType(),
        b.create<LLVM::ConstantOp>(b.getI64IntegerAttr(1)), align);
    b.create<LLVM::StoreOp>(value, ptr, align);
    return ptr;
  }

  // Materialize memrefs from the interpreter.
  if (scope)
    if (auto ref = dyn_cast<MemRefAttr>(attr))
      return scope->convertMemRef(b, ref);

  // Convert string constant to a struct{ptr, size} of type
  // !llvm.struct<(ptr<i8>, index).
  if (auto strAttr = dyn_cast<StringAttr>(attr)) {
    if (!imc)
      return {};
    return lowerStringToGlobalConstant(strAttr, b, tc, *imc);
  }

  if (auto cst = dyn_cast<SymbolConstantAttr>(attr)) {
    if (cst.getType().isCapturing()) {
      b.emitError("TODO: capturing closures cannot be materialized as runtime "
                  "values");
      return {};
    }
    return b.create<LLVM::AddressOfOp>(
        tc.convertType(cst.getType()),
        cast<FlatSymbolRefAttr>(cst.getSymbol()));
  }

  //===--------------------------------------------------------------------===//
  // POP

  // Convert SIMD constants to an array of integer or float constants.
  if (auto simd = dyn_cast<POP::SIMDAttr>(attr))
    return convertSIMDAttr(b, tc, simd);

  // Convert array, struct, or pack constants to LLVM array or struct constants.
  if (isa<POP::ArrayAttr, StructAttr>(attr)) {
    Type type = tc.convertType(attr.getType());
    if (!type)
      return {};
    Value aggregate = b.create<LLVM::UndefOp>(type);
    ArrayRef<TypedAttr> values =
        TypeSwitch<Attribute, ArrayRef<TypedAttr>>(attr)
            .Case<POP::ArrayAttr, StructAttr>(
                [](auto attr) { return attr.getValues(); });

    for (auto [idx, value] : llvm::enumerate(values)) {
      Value element = convertParameterToLLVM(b, tc, imc, scope, value);
      if (!element)
        return {};
      aggregate = b.create<LLVM::InsertValueOp>(aggregate, element, idx);
    }
    return aggregate;
  }

  // Bitpack union constants.
  if (auto unionAttr = dyn_cast<POP::UnionAttr>(attr)) {
    Value value =
        convertParameterToLLVM(b, tc, imc, scope, unionAttr.getValue());
    if (!value)
      return {};

    auto contentType =
        cast_or_null<LLVM::LLVMArrayType>(tc.convertType(unionAttr.getType()));
    if (!contentType)
      return {};
    VariantHelper helper(b, b.getLoc(), tc);
    return helper.materializeLLVMUnion(contentType, value);
  }

  // Convert variadic sequence constants to an LLVM struct constant.
  if (auto variadic = dyn_cast<KGEN::VariadicAttr>(attr)) {
    // 1. Allocate space for an array of elements.
    Type elementType = tc.convertType(variadic.getType().getElementType());
    if (!elementType)
      return {};

    Value size = b.create<LLVM::ConstantOp>(
        b.getI64IntegerAttr(variadic.getValues().size()));
    Value ptr = b.create<LLVM::AllocaOp>(
        LLVM::LLVMPointerType::get(b.getContext()), elementType, size);

    // 2. Store elements of the sequence into the allocated space.
    for (auto [idx, value] : llvm::enumerate(variadic.getValues())) {
      Value element = convertParameterToLLVM(b, tc, imc, scope, value);
      if (!element)
        return {};

      auto destination = b.create<LLVM::GEPOp>(
          LLVM::LLVMPointerType::get(b.getContext()), elementType, ptr,
          ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(idx)});
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
