//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LLVM_LOWERING_UTILS_H
#define KGEN_LLVM_LOWERING_UTILS_H

#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"

namespace mlir::LLVM {
class CallOp;
class LLVMFuncOp;
} // namespace mlir::LLVM

namespace M::KGEN {
class KGENDType;

//===----------------------------------------------------------------------===//
// LLVMDataLayout
//===----------------------------------------------------------------------===//

/// This class is a helper to compute size and alignment of LLVM-compatible MLIR
/// types using a data layout specification.
class LLVMDataLayout {
public:
  explicit LLVMDataLayout(TargetInfoAttr target) : target(target) {}

  /// Get the size of the LLVM type in bits.
  int64_t getTypeSizeInBits(Type type) const;
  /// Get the maximum number of bytes that can be overwritten by storing the
  /// type. This is the type size in bits rounded up to the nearest byte.
  int64_t getTypeStoreSize(Type type) const {
    return llvm::divideCeil(getTypeSizeInBits(type), CHAR_BIT);
  }
  /// Get the alloc size of the type. This is the size of the type plus the
  /// required alignment padding.
  int64_t getTypeAllocSize(Type type) const {
    return llvm::alignTo(getTypeStoreSize(type), getTypeABIAlign(type));
  }
  /// Get the ABI alignment of the LLVM type.
  int64_t getTypeABIAlign(Type type) const;

  /// Get the target info.
  TargetInfoAttr getTarget() const { return target; }

private:
  /// The target info with the data layout to use.
  TargetInfoAttr target;
};

//===----------------------------------------------------------------------===//
// POPToLLVMTypeConverter
//===----------------------------------------------------------------------===//

/// Get the MLIR type for a data type.
std::optional<Type> getMLIRTypeForDType(mlir::MLIRContext *ctx, KGENDType dtype,
                                        size_t indexBitwidth);

/// Get an LLVM pointer to the given dtype. If the dtype is unknown, return an
/// untyped pointer.
Type getLLVMPointerTo(mlir::MLIRContext *ctx, KGENDType dtype,
                      size_t indexBitwidth);

/// This type converter maps fully-specified pop dialect parametric types and
/// built-in MLIR types to LLVM types.
struct POPToLLVMTypeConverter : public mlir::LLVMTypeConverter,
                                public LLVMDataLayout {
  POPToLLVMTypeConverter(TargetInfoAttr target);
};

//===----------------------------------------------------------------------===//
// LLVMBuilder
//===----------------------------------------------------------------------===//

/// These are the default LLVM fastmath flags that are always set.
static constexpr mlir::LLVM::FastmathFlags LLVM_FASTMATH_FLAGS =
    mlir::LLVM::FastmathFlags::contract;

/// Create an `LLVM::CallOp` with the default fastmath flags.
template <typename... Args>
auto createLLVMCall(OpBuilder &b, Location loc, Args &&...args) {
  auto call = b.create<mlir::LLVM::CallOp>(loc, std::forward<Args>(args)...);
  // Attach the default fastmath flags.
  call.setFastmathFlags(LLVM_FASTMATH_FLAGS);
  return call;
}

/// Attach `target-cpu` and `target-features` to the LLVM function attributes,
/// even if null. These attributes are attached to `LLVMFuncOp` and passed on
/// to LLVM IR function attributes.
ArrayAttr attachTargetPassthroughAttrs(OpBuilder &b, TargetInfoAttr target,
                                       ArrayAttr passthrough);

/// Create an `LLVMFuncOp` with the target info attributes.
template <typename... Args>
auto createLLVMFunc(OpBuilder &b, TargetInfoAttr target, Location loc,
                    Args &&...args) {
  auto func =
      b.create<mlir::LLVM::LLVMFuncOp>(loc, std::forward<Args>(args)...);
  func.setPassthroughAttr(
      attachTargetPassthroughAttrs(b, target, func.getPassthroughAttr()));
  return func;
}

/// This class is a builder, type converter, and data layout bundled together.
struct LLVMBuilder : public ImplicitLocOpBuilder,
                     public POPToLLVMTypeConverter {
  LLVMBuilder(ImplicitLocOpBuilder &b, POPToLLVMTypeConverter &tc)
      : ImplicitLocOpBuilder(b), POPToLLVMTypeConverter(tc) {}

  using ImplicitLocOpBuilder::getContext;
  using POPToLLVMTypeConverter::getIndexType;

  /// Create an `LLVM::CallOp` with the default fastmath flags.
  template <typename... Args>
  auto createCall(Args &&...args) {
    return createLLVMCall(*this, getLoc(), std::forward<Args>(args)...);
  }

  /// Create an `LLVMFuncOp` with the target info attributes.
  template <typename... Args>
  auto createFunc(Args &&...args) {
    return createLLVMFunc(*this, getTarget(), getLoc(),
                          std::forward<Args>(args)...);
  }

  /// Create an `unrealized_conversion_cast` operation.
  Value createConversion(Type type, Value src) {
    return create<mlir::UnrealizedConversionCastOp>(type, src).getResult(0);
  }
};

//===----------------------------------------------------------------------===//
// VariantHelper
//===----------------------------------------------------------------------===//

/// A helper for creating variants and extracting from them.
class VariantHelper {
public:
  VariantHelper(OpBuilder &b, Location loc, const LLVMDataLayout &dl)
      : b(loc, b), dl(dl) {}

  /// Generate the code required to materialize the provided value as a variant
  /// of the given LLVM type.
  Value materializeLLVMVariant(Type type, Value value, int64_t index);

  /// Walk a simple or aggregate LLVM type and generate the code to insert its
  /// elements into a variant's content type. This tightly packs the element
  /// types within the content type. The first argument is an iterator to the
  /// current content element values. It is initialized with zeroes. The second
  /// is an iterator to the content element types.
  void walkAndCreateVariant(MutableArrayRef<Value>::iterator &valueIt,
                            unsigned &storageOffset, unsigned &offset,
                            Value value);

  /// Walk a simple or aggregate LLVM type and generate the code to extract its
  /// elements from a variant's content type.
  Value walkAndExtractVariant(ArrayRef<Value>::iterator &valueIt,
                              unsigned &storageOffset, unsigned &offset,
                              Type type);

private:
  /// The builder to use.
  ImplicitLocOpBuilder b;
  /// The data layout to use.
  LLVMDataLayout dl;
};

//===----------------------------------------------------------------------===//
// Interpreter Memory Conversion
//===----------------------------------------------------------------------===//

/// This is a utility class for deduplicating memory instantiations from the
/// interpreter.
class InterpreterMemoryConverter {
public:
  /// Create a converter instance. A single instance is held for an entire
  /// module to ensure globals are deduplicated.
  InterpreterMemoryConverter(SymbolTable &symtab, POPToLLVMTypeConverter &tc)
      : symtab(symtab), tc(tc) {}

  /// Convert a single memory reference.
  Value convertMemRef(ImplicitLocOpBuilder &b, MemRefAttr ref);

private:
  /// Each materialized blob will have a corresponding SSA value representing
  /// the pointer to the beginning of the blob or an LLVM global for
  /// `const_global` blobs.
  using MaterializedBlobs = SmallVector<PointerUnion<Operation *, Value>>;

  /// Ensure the blobs within the memory space have been materialized and
  /// then return them.
  MaterializedBlobs &getOrMaterialize(ImplicitLocOpBuilder &b,
                                      MemorySpaceAttr space);
  /// Get a pointer into the blob at the given offset.
  static Value getBlobPointer(ImplicitLocOpBuilder &b, Type ptrType,
                              MaterializedBlobs &materialized, int64_t index,
                              int64_t offset);

  /// The symbol table to use for globals.
  SymbolTable &symtab;
  /// The type converter to use.
  POPToLLVMTypeConverter &tc;

  /// Lazily materialized memory spaces.
  DenseMap<MemorySpaceAttr, MaterializedBlobs> blobs;
  /// Lazily materialized globals.
  llvm::StringMap<Operation *> globals;
};

//===----------------------------------------------------------------------===//
// Struct Conversion
//===----------------------------------------------------------------------===//

/// Generate the LLVM IR to materialize a struct of the given LLVM struct type,
/// and insert the given element values into the struct.
Value materializeLLVMStruct(ImplicitLocOpBuilder &b, Type structType,
                            ValueRange elements);

//===----------------------------------------------------------------------===//
// Attribute Conversion
//===----------------------------------------------------------------------===//

/// Generate the LLVM IR to materialize a constant of the given value. This is
/// used to convert attribute values in `kgen.param.constant`.
Value convertParameterToLLVM(ImplicitLocOpBuilder &b,
                             POPToLLVMTypeConverter &tc, SymbolTable &symtab,
                             InterpreterMemoryConverter *imc, TypedAttr attr);

//===----------------------------------------------------------------------===//
// POPToLLVMDebugInfoTypeConverter
//===----------------------------------------------------------------------===//

/// A specialized debug info type converter for converting from POP types to
/// LLVM.
struct POPToLLVMDebugInfoTypeConverter
    : public DebugInfo::DebugInfoTypeConverter {
  POPToLLVMDebugInfoTypeConverter(POPToLLVMTypeConverter &converter,
                                  TargetInfoAttr target);
};

//===----------------------------------------------------------------------===//
// ConvertPOPToLLVMPattern
//===----------------------------------------------------------------------===//

/// This is a templated instance of the wrapper class to rewrite a specific op.
template <typename OpT>
struct ConvertPOPToLLVMPattern : public mlir::ConvertOpToLLVMPattern<OpT> {
  using mlir::ConvertOpToLLVMPattern<OpT>::ConvertOpToLLVMPattern;

  /// Get the type converter.
  POPToLLVMTypeConverter *getTypeConverter() const {
    return static_cast<POPToLLVMTypeConverter *>(
        mlir::ConvertOpToLLVMPattern<OpT>::getTypeConverter());
  }

  /// Convert a type. Return null if the type conversion failed.
  Type convertType(Type type) const {
    return getTypeConverter()->convertType(type);
  }
};

} // namespace M::KGEN

#endif // KGEN_LLVM_LOWERING_UTILS_H
