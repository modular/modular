//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LLVM_LOWERING_UTILS_H
#define KGEN_LLVM_LOWERING_UTILS_H

#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/LLVMCompilerForwardDecls.h"
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
class ParamRefType;
class PointerType;
class VariantType;
class NoneType;
class SignatureType;
class StringType;
class StructType;

namespace POP {
class ArrayType;
class SIMDType;
} // namespace POP

namespace CO {
class CoroutineType;
} // namespace CO

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
  LLVMBuilder(ImplicitLocOpBuilder &b, TargetInfoAttr target)
      : ImplicitLocOpBuilder(b), POPToLLVMTypeConverter(target) {}

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
    if (src.getType() == type)
      return src;
    return create<mlir::UnrealizedConversionCastOp>(type, src).getResult(0);
  }
  /// Lower a value's type and cast it if necessary.
  Value createConversion(Value src) {
    if (Type type = convertType(src.getType()))
      return createConversion(type, src);
    return {};
  }

  /// Get the pointer width in bytes.
  size_t getPointerByteWidth() {
    return llvm::divideCeil(getIndexTypeBitwidth(), CHAR_BIT);
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

  /// Generate the code required to materialize the provided value as a union
  /// of the given LLVM type.
  Value materializeLLVMUnion(mlir::LLVM::LLVMArrayType type, Value value);

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
  /// Each materialized blob will have a corresponding SSA value representing
  /// the pointer to the beginning of the blob or an LLVM global for
  /// `const_global` blobs.
  using MaterializedBlobs = SmallVector<PointerUnion<Operation *, Value>>;

  /// Create a converter instance. A single instance is held for an entire
  /// module to ensure globals are deduplicated.
  InterpreterMemoryConverter(SymbolTable &symtab, POPToLLVMTypeConverter &tc)
      : symtab(symtab), tc(tc) {}

  /// A conversion scope represents the range of IR in which identity is
  /// uniquely bestowed upon memory space attributes with the same value.
  /// Outside a scope, the same memory space attribute, where equality is
  /// determined by the contents, resolve to different actual addresses.
  ///
  /// FIXME: Once the parser has a correct model for lifetimes and identity in
  /// parameter expressions, the scope struct can be removed and materialization
  /// can be globally-scoped again.
  class MaterializationScope {
  public:
    /// Convert a single memory reference.
    Value convertMemRef(ImplicitLocOpBuilder &b, MemRefAttr ref);

    /// Get the parent converter.
    InterpreterMemoryConverter &getParent() { return imc; }

  private:
    explicit MaterializationScope(InterpreterMemoryConverter &imc) : imc(imc) {}

    /// Ensure the blobs within the memory space have been materialized and
    /// then return them.
    MaterializedBlobs &getOrMaterialize(ImplicitLocOpBuilder &b,
                                        MemorySpaceAttr space);
    /// Get a pointer into the blob at the given offset.
    static Value getBlobPointer(ImplicitLocOpBuilder &b, Type ptrType,
                                MaterializedBlobs &materialized, int64_t index,
                                int64_t offset);

    /// The interpreter memory converter.
    InterpreterMemoryConverter &imc;

    /// Lazily materialized memory spaces.
    DenseMap<MemorySpaceAttr, MaterializedBlobs> blobs;

    friend class InterpreterMemoryConverter;
  };

  friend class MaterializationScope;

  /// Create a new identity scope for converting memory values.
  MaterializationScope createScope() { return MaterializationScope(*this); }

  /// Get or add a global for the handle. It must be a `const_global` region.
  Operation *getOrCreateGlobal(Location loc, MemoryHandle hdl);

private:
  /// The symbol table to use for globals.
  SymbolTable &symtab;
  /// The type converter to use.
  POPToLLVMTypeConverter &tc;

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
Value convertParameterToLLVM(
    ImplicitLocOpBuilder &b, const POPToLLVMTypeConverter &tc,
    InterpreterMemoryConverter *imc,
    InterpreterMemoryConverter::MaterializationScope *scope, TypedAttr attr);

//===----------------------------------------------------------------------===//
// DebugInfoTypeConverter
//===----------------------------------------------------------------------===//

/// A specialized debug info type converter for converting from POP types to
/// LLVM.
class DebugInfoTypeConverter : public DebugInfo::DebugInfoTypeConverter {
public:
  DebugInfoTypeConverter(POPToLLVMTypeConverter &tc);

private:
  POPToLLVMTypeConverter &tc;

  /// Build the debug type for a struct-like type.
  DebugInfo::DIType buildDebugStructTypeFromTypeAttrs(ArrayRef<Type> attrs,
                                                      StringAttr name);
  /// Build the debug type for a function type.
  DebugInfo::DIType buildDebugSubroutineType(FunctionType type);
  /// Build a pointer type.
  DebugInfo::DIType buildPointerType(DebugInfo::DIType type);
  DebugInfo::DIType buildPointerType(DebugInfo::DIType type,
                                     std::optional<unsigned> addressSpace);

  /// Build fully resolved debug type from partially resolved ones.
  DebugInfo::DIType
  buildDebugType(DebugInfo::DITargetIndependentPointerType type);

  /// Build fully resolved debug type from kgen/pop types.
  DebugInfo::DIType buildDebugType(IndexType type);
  DebugInfo::DIType buildDebugType(ParamRefType type);
  DebugInfo::DIType buildDebugType(StringType type);
  DebugInfo::DIType buildDebugType(SignatureType type);
  DebugInfo::DIType buildDebugType(KGEN::VariantType type);
  DebugInfo::DIType buildDebugType(KGEN::NoneType type);
  DebugInfo::DIType buildDebugType(POP::ArrayType type);
  DebugInfo::DIType buildDebugType(CO::CoroutineType type);
  DebugInfo::DIType buildDebugType(PointerType type);
  DebugInfo::DIType buildDebugType(POP::SIMDType type);
  DebugInfo::DIType buildDebugType(StructType type);
};

//===----------------------------------------------------------------------===//
// ConvertPOPToLLVMPattern
//===----------------------------------------------------------------------===//

/// This is a templated instance of the wrapper class to rewrite a specific op.
template <typename OpT>
struct ConvertPOPToLLVMPattern : public mlir::ConvertOpToLLVMPattern<OpT> {
  using mlir::ConvertOpToLLVMPattern<OpT>::ConvertOpToLLVMPattern;

  /// Get the type converter.
  const POPToLLVMTypeConverter *getTypeConverter() const {
    return static_cast<const POPToLLVMTypeConverter *>(
        mlir::ConvertOpToLLVMPattern<OpT>::getTypeConverter());
  }

  /// Convert a type. Return null if the type conversion failed.
  Type convertType(Type type) const {
    return getTypeConverter()->convertType(type);
  }
};

//===----------------------------------------------------------------------===//
// ConvertSymbolOpToLLVM
//===----------------------------------------------------------------------===//

/// This pattern is used to rewrite symbol operations while keeping the symbol
/// table up-to-date.
template <typename OpT>
class ConvertSymbolOpToLLVM : public ConvertPOPToLLVMPattern<OpT> {
public:
  ConvertSymbolOpToLLVM(mlir::LLVMTypeConverter &typeConverter,
                        SymbolTable &symtab)
      : ConvertPOPToLLVMPattern<OpT>(typeConverter), symtab(symtab) {}

protected:
  /// The symbol table.
  SymbolTable &symtab;
};

} // namespace M::KGEN

#endif // KGEN_LLVM_LOWERING_UTILS_H
