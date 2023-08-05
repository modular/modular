//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MDIALECT_MATTRS_H
#define SUPPORT_MDIALECT_MATTRS_H

#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrInterfaces.h"
#include "Support/MDialect/MDialect.h"
#include "Support/MDialect/MTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/TargetParser/Triple.h"

namespace M {

// Forward declarations.
struct HostMachineInfo;

//===----------------------------------------------------------------------===//
// DataLayout
//===----------------------------------------------------------------------===//

/// This class defines a data layout specification for "basic" data types:
/// integers, floats, vectors, and pointers. It defines the bitwidth and ABI
/// alignment of these types. All other types should implement
/// `DataLayoutInterface`.
///
/// Bitwidth is determined as follows:
///
/// - Integers: Bitwidth is the integer bitwidth
/// - Floats:   Bitwidth is the float bitwidth
/// - Vectors:  Bitwidth is the number of elements times the element bitwidth
/// - Pointers: Bitwidth is the width of integers in the default address space.
///
/// ABI alignment is determined as follows:
///
/// - Integers: The alignment specification array is searched for an entry that
///             matches the bitwidth of the integer type. If one is not found,
///             the alignment of the next largest integer type is used. This
///             requires at least one integer type entry in the data layout
///             specification.
/// - Floats:   The alignment specification array is searched for an entry that
///             matches the bitwidth of the float type. If one is not found, the
///             alignment is taken as the bitwidth rounded up to the next byte
///             and then the first power of two at or after that.
/// - Vectors:  The alignment specification array is searched for an entry that
///             matches the bitwidth of the vector type. If one is not found,
///             the alignment is taken as the bitwidth rounded up to the next
///             byte and then the first power of two at or after that.
/// - Pointers: The alignment for pointers in the default address space is
///             returned.
///
/// This class covers the minimum surface required to interoperate with LLVM's
/// data layout. It should be expanded as required. The textual format is
/// identical to LLVM's data layout specification.
class DataLayout {
public:
  /// Get the default address space pointer bitwidth.
  int32_t getPointerBitWidth() const { return ptrWidth; }
  /// Get the default address space pointer size in bytes.
  int32_t getPointerSize() const {
    return llvm::divideCeil(getPointerBitWidth(), 8);
  }
  /// Get the bitwidth of a fixed vector type.
  int32_t getVectorBitWidth(int32_t numElts, int32_t eltBitWidth) const {
    return numElts * eltBitWidth;
  }

  /// Get the ABI alignment of an integer type.
  int32_t getIntegerABIAlign(int32_t bitwidth) const;
  /// Get the ABI alignment of float type.
  int32_t getFloatABIAlign(int32_t bitwidth) const;
  /// Get the ABI alignment of a vector type.
  int32_t getVectorABIAlign(int32_t numElts, int32_t eltBitWidth) const;
  /// Get the default address space pointer ABI alignment.
  int32_t getPointerABIAlign() const { return ptrAbiAlign; }

  /// Attempt to parse a data layout from the specification string. Returns an
  /// error if parsing failed.
  static ErrorOr<DataLayout> parse(StringRef desc);
  /// Convert the data layout to its specification string.
  StringRef toString() const { return dlSpecStr; }

private:
  DataLayout(StringRef dlSpecStr);

  /// Parse the data layout from its string specification.
  ErrorOrSuccess parse();

  /// The list of alignment entries for integers.
  SmallVector<std::pair<int32_t, int32_t>> intAbiAlign;
  /// The list of alignment entries for floats.
  SmallVector<std::pair<int32_t, int32_t>> fpAbiAlign;
  /// The list of alignment entries for vectors.
  SmallVector<std::pair<int32_t, int32_t>> vecAbiAlign;
  /// The pointer width.
  int32_t ptrWidth;
  /// The pointer ABI alignment.
  int32_t ptrAbiAlign;

  /// The underlying string representation.
  std::string dlSpecStr;
};

//===----------------------------------------------------------------------===//
// ArrayElementsAttr
//===----------------------------------------------------------------------===//

namespace detail {
class AttrIterator
    : public llvm::indexed_accessor_iterator<AttrIterator, const uint8_t *,
                                             Attribute, Attribute, Attribute> {
public:
  AttrIterator(const uint8_t *data, size_t index, Type elementType)
      : indexed_accessor_iterator(data, index), elementType(elementType) {}

  Attribute operator*() const;

private:
  /// The element type.
  Type elementType;
};
} // namespace detail

//===----------------------------------------------------------------------===//
// MemRefAttr
//===----------------------------------------------------------------------===//

using MemoryHandle = mlir::DialectResourceBlobHandle<MDialect>;
using MBlobManagerInterface =
    mlir::ResourceBlobManagerDialectInterfaceBase<MemoryHandle>;

enum class MemoryKind { Stack, Heap };

class MemoryBlob {
public:
  /// A pointer region is a chunk of memory in the reference blob that
  /// represents a pointer.
  struct PointerRegion {
    /// The location of the region within the current blob.
    int64_t offset;
    /// The index of the referenced blob.
    int64_t blobIndex;
    /// The offset into the reference blob.
    int64_t blobOffset;
  };

  /// Create a memory blob.
  MemoryBlob(MemoryHandle hdl, MemoryKind kind,
             SmallVector<PointerRegion> pointerRegions)
      : hdl(hdl), kind(kind), pointerRegions(std::move(pointerRegions)) {}

  /// Get the memory handle.
  MemoryHandle getHandle() const { return hdl; }
  /// Get the memory kind.
  MemoryKind getKind() const { return kind; }
  /// Get the pointer offsets.
  ArrayRef<PointerRegion> getPointerRegions() const { return pointerRegions; }

private:
  /// The handle to the dialect resource that contains the blob data.
  MemoryHandle hdl;
  /// The kind of memory.
  MemoryKind kind;
  /// The offsets into the data that represent pointers.
  SmallVector<PointerRegion> pointerRegions;
};

} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Support/MDialect/MAttrs.h.inc"

//===----------------------------------------------------------------------===//
// IntArrayElementsAttr
//===----------------------------------------------------------------------===//

namespace M {

/// This class represents a dense array of integers. Integer elements that do
/// not fit evenly into bytes are rounded up to the nearest byte.
class IntArrayElementsAttr : public ArrayElementsAttr {
public:
  using ArrayElementsAttr::ArrayElementsAttr;

  /// Create an integer array. All `APInt`s must have the same width.
  static IntArrayElementsAttr get(ShapedType type, ArrayRef<APInt> values);
  static IntArrayElementsAttr get(ShapedType type, ArrayRef<APSInt> values);

  /// Create an integer from an array of C++ values.
  template <typename IntT>
  static IntArrayElementsAttr get(MLIRContext *ctx, ArrayRef<IntT> values,
                                  IntegerType::SignednessSemantics signedness) {
    auto type = IntegerType::get(ctx, sizeof(IntT) * CHAR_BIT, signedness);
    return ArrayElementsAttr::get(
               {reinterpret_cast<const uint8_t *>(values.data()),
                values.size() * sizeof(IntT)},
               ArrayType::get(values.size(), type))
        .template cast<IntArrayElementsAttr>();
  }

  /// Iterate over the integer elements as `APInt`s.
  class Iterator
      : public llvm::indexed_accessor_iterator<Iterator, const uint8_t *, APInt,
                                               APInt, APInt> {
  public:
    APInt operator*() const;

  private:
    Iterator(IntegerType type, const uint8_t *base, size_t index)
        : indexed_accessor_iterator(base, index), type(type) {}

    friend class IntArrayElementsAttr;

    /// The element type.
    IntegerType type;
  };

  Iterator begin() const;
  Iterator end() const;
  auto getValues() { return llvm::make_range(begin(), end()); }

  template <typename IntT>
  ArrayRef<IntT> asArrayRef() {
    assert(sizeof(IntT) * CHAR_BIT == getElementType().getIntOrFloatBitWidth());
    return {reinterpret_cast<const IntT *>(getRawData().data()),
            static_cast<size_t>(size())};
  }

  /// Support type inquiry.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// custom<DenseIntArray>
//===----------------------------------------------------------------------===//

/// Parse or print an array of dense integers without the surrounding braces.
ParseResult parseDenseIntArray(
    AsmParser &p, IntArrayElementsAttr &result, unsigned width,
    IntegerType::SignednessSemantics signedness = IntegerType::Signed);
void printDenseIntArray(
    AsmPrinter &p, Operation *op, IntArrayElementsAttr result, unsigned width,
    IntegerType::SignednessSemantics signedness = IntegerType::Signed);

//===----------------------------------------------------------------------===//
// FloatArrayElementsAttr
//===----------------------------------------------------------------------===//

/// This class represents a dense array of floats. Float elements that do not
/// fit evenly into bytes are rounded up to the nearest byte.
class FloatArrayElementsAttr : public ArrayElementsAttr {
public:
  using ArrayElementsAttr::ArrayElementsAttr;

  /// Create a float array. All `APFloat`s must have the same width.
  static FloatArrayElementsAttr get(ShapedType type, ArrayRef<APFloat> values);

  /// Create a float array. All `APFloat`s must have the same width.
  static FloatArrayElementsAttr get(ArrayRef<APFloat> values, Type elementType);

  /// Iterate over the float elements as `APFloat`s.
  class Iterator
      : public llvm::indexed_accessor_iterator<Iterator, const uint8_t *,
                                               APFloat, APFloat, APFloat> {
  public:
    APFloat operator*() const;

  private:
    Iterator(FloatType type, const uint8_t *base, size_t index)
        : indexed_accessor_iterator(base, index), type(type) {}

    friend class FloatArrayElementsAttr;

    /// The element type.
    FloatType type;
  };

  Iterator begin() const;
  Iterator end() const;
  auto getValues() { return llvm::make_range(begin(), end()); }

  template <typename FloatT>
  ArrayRef<FloatT> asArrayRef() {
    assert(sizeof(FloatT) * CHAR_BIT ==
           getElementType().getIntOrFloatBitWidth());
    return {reinterpret_cast<const FloatT *>(getRawData().data()),
            static_cast<size_t>(size())};
  }

  /// Support type inquiry.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// IndexArrayElementsAttr
//===----------------------------------------------------------------------===//

/// This class represents a dense array of indices. Index type elements are
/// stored according to the index type's internal storage bitwidth.
class IndexArrayElementsAttr : public ArrayElementsAttr {
public:
  using ArrayElementsAttr::ArrayElementsAttr;

  /// Create an index array.
  static IndexArrayElementsAttr get(ShapedType type, ArrayRef<int64_t> values);

  using iterator = ArrayRef<int64_t>::iterator;

  iterator begin() const { return asArrayRef().begin(); }
  iterator end() const { return asArrayRef().end(); }

  ArrayRef<int64_t> asArrayRef() const {
    return {reinterpret_cast<const int64_t *>(getRawData().data()),
            static_cast<size_t>(size())};
  }

  /// Support type inquiry.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// Attribute Conversion
//===----------------------------------------------------------------------===//

/// Convert a `DenseElementsAttr` to an `ArrayElementsAttr`. Pass through any
/// other kind of attribute. This should be the only place where the splatness
/// and bitpacked-ness of the attribute are handled.
Attribute convertDenseElements(Attribute attr);

/// Returns an ArrayElementsAttr representing data. The given data is always
/// copied into the MLIR context.
ElementsAttr getInlineAttrForTensorDataCopy(ShapedType type,
                                            ArrayRef<char> data);

/// Returns an attribute to store the given tensor data. If the type's number
/// of elements is small, returns an ArrayElementsAttr. Otherwise creates
/// a blob and returns a DenseResourceElementsAttr. The given data is always
/// copied into the MLIR context.
ElementsAttr
getAttrForTensorDataCopy(ShapedType type, StringRef bufferName,
                         ArrayRef<char> data,
                         DenseResourceElementsHandleManager &resourceManager);

/// Return the contents of the IntArrayElementsAttr as a vector in int64_t.
SmallVector<int64_t> getIntBlob(IntArrayElementsAttr intElemsAttr);

/// Return the contents of the FloatArrayElementsAttr as a vector in float.
SmallVector<float> getFloatBlob(FloatArrayElementsAttr floatElemsAttr);

//===----------------------------------------------------------------------===//
// TargetInfoAttr
//===----------------------------------------------------------------------===//

/// Look for a target info specification inside the provided module. Returns
/// null if there is not one.
TargetInfoAttr getTargetInfo(ModuleOp module);
/// Set the target info specification on the provided module. The module cannot
/// already have a target specification.
void setTargetInfo(ModuleOp module, TargetInfoAttr target);
/// Look for a target info specification in the nearest surrounding module from
/// the provided operation. Returns null if one cannot be found.
TargetInfoAttr lookupTargetInfo(Operation *from);
/// Get the target info for the specified target.
ErrorOr<TargetInfoAttr> getTargetInfoFor(MLIRContext *ctx,
                                         StringRef targetTriple, StringRef cpu,
                                         StringRef features,
                                         StringRef tuneCpu = "");
/// Returns the target info partially describing the given HostMachineInfo.
/// Only some fields are captured:
///  - triple (captured as triple)
///  - cpuArch (captured as cpu)
///  - cpuFeatures (captured as features, with each feature prefixed by '+'
///    and features separated by ',')
/// Note that the data_layout and simd_bit_width fields of the result are
/// left empty/zero.
///
/// Unlike the above getTargetInfoFor/4, this method does not depend on any
/// LLVM target management infrastructure and can be used outside of a
/// jit context.
ErrorOr<TargetInfoAttr> getTargetInfoFor(MLIRContext *ctx,
                                         HostMachineInfo &hostMachineInfo);
/// Return a serialized representation of targetInfoAttr which can be
/// deserialized by M::recoverHostMachineInfo in Support/Host.h. This can
/// be used to capture assumptions about the runtime host machine architecture
/// with generated artifacts such as MEF files.
ErrorOr<std::string>
serializeTargetInfoAttrToJSON(TargetInfoAttr targetInfoAttr);

//===----------------------------------------------------------------------===//
// BuildInfoAttr
//===----------------------------------------------------------------------===//

/// Look for a build info specification inside the provided module. Returns
/// null if there is not one.
BuildInfoAttr getBuildInfo(ModuleOp module);
/// Set the build info specification on the provided module. The module cannot
/// already have a build info specification.
void setBuildInfo(ModuleOp module, BuildInfoAttr info);
/// Look for a build info specification in the nearest surrounding module from
/// the provided operation. Returns null if one cannot be found.
BuildInfoAttr lookupBuildInfo(Operation *from);

} // namespace M

#endif // SUPPORT_MDIALECT_MATTRS_H
