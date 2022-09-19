//===- MetaTypes.cpp ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MetaDialect/MetaTypes.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/MetaDialect/MetaDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include <random>

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Casting Between Meta and Builtin
//===----------------------------------------------------------------------===//

LogicalResult M::KGEN::checkMetaCastedTypes(
    function_ref<InFlightDiagnostic(StringRef)> emitError, Type metaTy,
    Type standardTy,
    function_ref<LogicalResult(Type, DTypeConstantAttr)> checkDType) {
  if (metaTy.isa<ParamRefType>())
    return success();

  if (auto scalarTy = metaTy.dyn_cast<ScalarType>()) {
    // Check that the data types match.
    if (auto dtype = scalarTy.getDType().dyn_cast<DTypeConstantAttr>();
        dtype && failed(checkDType(standardTy, dtype)))
      return emitError("incompatible scalar data type");
    return success();
  }

  // Check that the standard type is a rank 1 vector with matching dimensions.
  auto simdTy = metaTy.cast<SIMDType>();
  auto vectorTy = standardTy.dyn_cast<VectorType>();
  if (!vectorTy)
    return emitError("expected a vector type");
  if (vectorTy.getNumScalableDims() != 0)
    return emitError("vector type should not be scalable");
  if (vectorTy.getRank() != 1)
    return emitError("expected a rank 1 vector");
  if (auto size = simdTy.getSize().dyn_cast<IntegerAttr>();
      size && size.getInt() != vectorTy.getShape().front())
    return emitError("dimensions do not match");
  if (auto dtype = simdTy.getDType().dyn_cast<DTypeConstantAttr>();
      dtype && failed(checkDType(vectorTy.getElementType(), dtype)))
    return emitError("element types do not match");
  return success();
}

LogicalResult M::KGEN::checkMetaCastedTypes(
    function_ref<InFlightDiagnostic(StringRef)> emitError, Type metaTy,
    Type standardTy) {
  return checkMetaCastedTypes(emitError, metaTy, standardTy,
                              [](Type type, DTypeConstantAttr dtype) {
                                return success(dtype.isConvertibleTo(type));
                              });
}

//===----------------------------------------------------------------------===//
// Meta Type Constraints
//===----------------------------------------------------------------------===//

ScalarType M::KGEN::getScalarOfSameDType(Type type) {
  return ScalarType::get(type.getContext(),
                         type.cast<DTypeInterface>().getDType());
}

PointerType M::KGEN::getPointerOfSameDType(Type type) {
  if (TypedAttr dtype = type.cast<DTypeInterface>().getDType())
    return PointerType::get(ScalarType::get(dtype));
  return PointerType::get(type.getContext(), nullptr);
}

//===----------------------------------------------------------------------===//
// custom<ParamDTypeValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamDTypeValue(AsmParser &p,
                                        FailureOr<TypedAttr> &result) {
  TypedAttr retValue;
  if (failed(parseParamValue(p, retValue, p.getBuilder().getType<DTypeType>())))
    return failure();
  result = retValue;
  return success();
}

static void printParamDTypeValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, value);
}

//===----------------------------------------------------------------------===//
// custom<OptionalParamDTypeValue>
//===----------------------------------------------------------------------===//

static ParseResult parseOptionalParamDTypeValue(AsmParser &p,
                                                FailureOr<TypedAttr> &result) {
  if (succeeded(p.parseOptionalQuestion())) {
    result = TypedAttr();
    return success();
  }
  return parseParamDTypeValue(p, result);
}

static void printOptionalParamDTypeValue(AsmPrinter &p, Attribute value) {
  if (!value) {
    p << '?';
    return;
  }
  printParamDTypeValue(p, value);
}

//===----------------------------------------------------------------------===//
// custom<OptionalTypeParamValue>
//===----------------------------------------------------------------------===//

static ParseResult parseOptionalTypeParamValue(AsmParser &p,
                                               FailureOr<TypedAttr> &result) {
  if (succeeded(p.parseOptionalQuestion())) {
    result = TypedAttr();
    return success();
  }
  return parseTypeParamValue(p, result);
}

static void printOptionalTypeParamValue(AsmPrinter &p, TypedAttr value) {
  if (!value) {
    p << '?';
    return;
  }
  return printTypeParamValue(p, value);
}

//===----------------------------------------------------------------------===//
// ScalarType
//===----------------------------------------------------------------------===//

LogicalResult
ScalarType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   TypedAttr dtype) {
  if (!dtype.getType().isa<DTypeType>())
    return emitError() << "parameter for scalar type must be a !kgen.dtype";
  return success();
}

void ScalarType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getDType());
}

Type ScalarType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                             ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 1 && replTypes.empty());
  return ScalarType::get(replAttrs[0]);
}

/// Resolve the dtype of a DTypeInterface Type. If the interface has `invalid`
/// DType, then given a `tag` attribute, if it's a DTypeConstantAttr then pull
/// out the DType and return it. Otherwise, return failure.
static FailureOr<DType> resolveDTypeWithTag(DTypeInterface itf, Location loc,
                                            Attribute tag) {
  DType dtype = itf.resolveDType();
  if (dtype != DType::invalid)
    return dtype;

  if (auto dt = tag.dyn_cast<DTypeConstantAttr>())
    return dt.getDType();

  return emitError(loc) << "could not resolve dtype";
}

/// Fill `obj` according to `kind`, `dtype`, and `numElements`. Despite `obj`
/// being suggestively named, `obj` can be any pointer - it does not have to be
/// the pointer passed to ::populate. It must have a space allocated for
/// `numElements` objects of type `dtype`, however.
static LogicalResult doFill(Location loc, InputGenKind kind, DType dtype,
                            size_t numElements, void *obj) {
  switch (kind) {
  case InputGenKind::Zeros:
    memset(obj, 0, dtype.getSizeInBytes(numElements));
    return success();
  case InputGenKind::Ones: {
    if (dtype.isComplex()) {
      unsigned widthInBytes = dtype.getWidthInBits() / 8;
      // Set the imaginary component to zero.
      memset((char *)obj + widthInBytes, 0, widthInBytes);
      dtype = dtype.stripComplex();
    }

    // Dispatch the dtype, and just fill directly with ones.
    return dtype.dispatch<LogicalResult>(obj)
        .when([&](bool *ptr) {
          std::generate(ptr, ptr + numElements, []() { return true; });
          return success();
        })
        .whenCXXInt([&](auto *ptr) { // Standard C++ integers.
          std::generate(ptr, ptr + numElements, []() { return 1; });
          return success();
        })
        .whenCXXFP([&](auto *ptr) { // float and double.
          std::generate(ptr, ptr + numElements, []() { return 1.0; });
          return success();
        })
        .otherwise([&]() { return failure(); });
  }
  case InputGenKind::Random: {
    // Fill the given buffer with random elements from the provided
    // distribution.
    auto fillWithDistribution = [&](auto *ptr, auto distribution) {
      std::default_random_engine randEngine(/*seed=*/0);
      std::generate(ptr, ptr + numElements,
                    [&]() { return distribution(randEngine); });
    };

    return dtype.dispatch<LogicalResult>(obj)
        .when([&](bool *destPtr) {
          fillWithDistribution(destPtr, std::bernoulli_distribution());
          return success();
        })
        .whenCXXInt([&](auto *destPtr) {
          fillWithDistribution(destPtr, std::uniform_int_distribution<>(
                                            dtype.isSInt() ? -10 : 0, 10));
          return success();
        })
        .whenCXXFP([&](auto *destPtr) {
          fillWithDistribution(destPtr,
                               std::uniform_real_distribution<>(-1.0, 1.0));
          return success();
        })
        .otherwise([&]() { return failure(); });
  }
  }

  return emitError(loc) << "could not fill with gen kind: "
                        << stringifyInputGenKind(kind);
}

/// Compares raw buffers `lhs` and `rhs` of type `dtype` with `numElements`
/// elements. Returns true if they are equal, false if they are not, and failure
/// if they cannot be compared.
static FailureOr<bool> dataEquals(Location loc, DType dtype, size_t numElements,
                                  void *lhs, void *rhs) {
  return dtype.dispatch<FailureOr<bool>>(lhs, rhs)
      .whenCXXInt([&](auto *lhs, auto *rhs) {
        for (size_t i = 0; i < numElements; ++i)
          if (lhs[i] != rhs[i])
            return false;
        return true;
      })
      .whenCXXFP([&](auto *lhs, auto *rhs) {
        // This comparison could be improved if we wanted to, currently it more
        // or less just compares with relative tolerance.
        double epsilon = 1.0e-8;
        for (size_t i = 0; i < numElements; ++i)
          if (std::abs(lhs[i] - rhs[i]) >
              (std::min(std::abs(lhs[i]), std::abs(rhs[i])) * epsilon))
            return false;
        return true;
      })
      .otherwise([&]() {
        return mlir::emitError(loc) << "unknown dtype: " << dtype.getAsString();
      });
}

/// This implements `OpaqueObjectInterface::populate`. It generates a single
/// scalar according to the method prescribed by `kind`. If the dtype is
/// unknown, then this expects the tag attribute to be a type attr. Otherwise,
/// it expects UnitAttr.
LogicalResult ScalarType::populate(Location loc, InputGenKind kind,
                                   Attribute tag, void *obj) const {
  // Resolve the dtype.
  auto dtypeOr = resolveDTypeWithTag(*this, loc, tag);
  if (failed(dtypeOr))
    return failure();
  DType dtype = *dtypeOr;

  return doFill(loc, kind, dtype, 1, obj);
}

/// This implements `OpaqueObjectInterface::destroy`. Nothing to be done for
/// ScalarType, there are no additional allocations.
void ScalarType::destroy(Attribute tag, void *obj) const { return; }

/// This implements `OpaqueObjectInterface::getSizeInBytes`.
FailureOr<size_t> ScalarType::getSizeInBytes(Location loc,
                                             Attribute tag) const {
  auto dtypeOr = resolveDTypeWithTag(*this, loc, tag);
  if (failed(dtypeOr))
    return failure();

  return dtypeOr->getSizeInBytes(1);
}

FailureOr<bool> ScalarType::equals(Location loc, Attribute tag, void *lhsData,
                                   void *rhsData) const {
  // Check that the dtypes are equal. This only does something if the two dtypes
  // are actually different (i.e. unknown statically, dynamically carried by the
  // evaluation configuration). If the dtype is statically known by the
  // ScalarType then lhsDtype and rhsDtype will be equal.
  FailureOr<DType> dtypeOr = resolveDTypeWithTag(*this, loc, tag);
  if (failed(dtypeOr))
    return failure();

  // Compare the outputs if we can.
  return dataEquals(loc, *dtypeOr, 1, lhsData, rhsData);
}

//===----------------------------------------------------------------------===//
// SIMDType
//===----------------------------------------------------------------------===//

LogicalResult
SIMDType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                 TypedAttr size, TypedAttr dtype) {
  if (!size || !dtype)
    return emitError() << "simd type requires size and dtype";
  if (!size.getType().isIndex())
    return emitError() << "size parameter for simd must have type `index`";
  if (!dtype.getType().isa<DTypeType>())
    return emitError() << "type parameter for simd must be a !kgen.dtype";
  return success();
}

void SIMDType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getSize());
  walkAttrsFn(getDType());
}

Type SIMDType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                           ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 2 && replTypes.empty());
  return SIMDType::get(replAttrs[0], replAttrs[1]);
}

Optional<int64_t> SIMDType::resolveSize() const {
  if (auto intAttr = getSize().dyn_cast<IntegerAttr>())
    return intAttr.getInt();
  return {};
}

/// This implements `OpaqueObjectInterface::populate`. It generates a SIMD
/// vector (really an array of elements) according to `kind` and stores it in
/// `obj`.
LogicalResult SIMDType::populate(Location loc, InputGenKind kind, Attribute tag,
                                 void *obj) const {
  DType dtype = resolveDType();
  // If the dtype is invalid, we can't do anything. Note that we aren't trying
  // to get anything from the tag here!
  assert(dtype != DType::invalid && "SIMDType must have a valid dtype");

  auto sizeOr = resolveSize();
  if (!sizeOr.has_value())
    return failure();
  size_t numElements = *sizeOr;

  return doFill(loc, kind, dtype, numElements, obj);
}

/// This implements `OpaqueObjectInterface::destroy`. Nothing to be done for
/// SIMDType, there are no additional allocations.
void SIMDType::destroy(Attribute tag, void *obj) const { return; }

/// This implements `OpaqueObjectInterface::getSizeInBytes`. Since a SIMD vector
/// has all its elements inline, compute the size of the array needed to hold
/// tightly-packed elements for this type.
FailureOr<size_t> SIMDType::getSizeInBytes(Location loc, Attribute tag) const {
  DType dtype = resolveDType();
  // If the dtype is invalid, we can't do anything. Note that we aren't trying
  // to get anything from the tag here!
  assert(dtype != DType::invalid && "SIMDType must have a valid dtype");

  // Same with the size, if it's unknown (which it should not be) then
  // we can't do anything.
  auto sizeOr = resolveSize();
  assert(sizeOr.has_value() && "SIMDType must have a statically-known size");

  return dtype.getSizeInBytes(*sizeOr);
}

FailureOr<bool> SIMDType::equals(Location loc, Attribute tag, void *lhsData,
                                 void *rhsData) const {
  // Everything in a SIMDType must be static, so we can just directly compare
  // the data.
  DType dtype = resolveDType();
  assert(dtype != DType::invalid && "SIMDType must have a valid dtype");

  Optional<int64_t> sizeOr = resolveSize();
  assert(sizeOr.has_value() && "SIMDType must have a statically-known size");

  return dataEquals(loc, dtype, *sizeOr, lhsData, rhsData);
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

LogicalResult
BufferType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   TypedAttr size, TypedAttr dtype) {
  if (size && !size.getType().isIndex())
    return emitError() << "size parameter for buffer must have type `index`";
  if (dtype && !dtype.getType().isa<DTypeType>())
    return emitError() << "type parameter for buffer must be a !kgen.dtype";
  return success();
}

void BufferType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getSize());
  walkAttrsFn(getDType());
}

Type BufferType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                             ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 2 && replTypes.empty());
  return BufferType::get(getContext(), replAttrs[0], replAttrs[1]);
}

Optional<int64_t> BufferType::resolveSize() const {
  if (auto intAttr = getSize().dyn_cast_or_null<IntegerAttr>())
    return intAttr.getInt();
  return {};
}

BufferType BufferType::get(TypedAttr size, TypedAttr dtype) {
  return get(size.getContext(), size, dtype);
}

BufferType BufferType::get(MLIRContext *ctx, int64_t size, DType dtype) {
  return get(OpBuilder(ctx).getIndexAttr(size),
             DTypeConstantAttr::get(ctx, dtype));
}

/// This implements `OpaqueObjectInterface::populate`. It generates a buffer
/// object, and furthermore allocates memory for the buffer's backing storage
/// and places that in the pointer field of the buffer structure itself.
LogicalResult BufferType::populate(Location loc, InputGenKind kind,
                                   Attribute tag, void *obj) const {
  // FIXME: This doesn't currently handle dynamic size/type buffers - we need
  //        the tag to contain size, type, or both. Come up with a nice
  //        attribute structure that enables that use case.

  // Resolve the dtype.
  DType dtype = resolveDType();
  if (dtype == DType::invalid)
    return emitError(loc)
           << "Buffers with unbound dtype are not yet supported: " << *this;

  auto sizeOr = resolveSize();
  if (!sizeOr.has_value())
    return emitError(loc) << "Buffers with unbound size are not yet supported: "
                          << *this;

  int64_t numElements = *sizeOr;
  auto *ptr = (std::byte *)malloc(dtype.getSizeInBytes(numElements));

  // When the number of elements is statically-knowable, the object is just a
  // pointer.
  *((std::byte **)obj) = ptr;

  // Do the fill.
  return doFill(loc, kind, dtype, numElements, ptr);
}

/// This implements `OpaqueObjectInterface::destroy`. This deallocates any
/// memory allocated in `populate`.
void BufferType::destroy(Attribute tag, void *obj) const {
  free(*((uint8_t **)obj));
}

/// This implements `OpaqueObjectInterface::getSizeInBytes`. We don't care about
/// the buffer's allocation, we care about the size of the buffer itself.
FailureOr<size_t> BufferType::getSizeInBytes(Location loc,
                                             Attribute tag) const {
  // The size of a buffer is at worst size of (length, pointer, dtype).
  size_t size = sizeof(intptr_t) + sizeof(void *) + sizeof(int8_t);
  if (resolveSize().has_value())
    size -= sizeof(intptr_t);

  if (resolveDType() != M::DType::invalid)
    size -= sizeof(int8_t);

  return size;
}

/// This method compares two instances of data held in a buffer of a given type.
/// This is a deep comparison.
FailureOr<bool> BufferType::equals(Location loc, Attribute tag, void *lhsData,
                                   void *rhsData) const {
  // TODO: Much like above, this doesn't handle the dynamic-size or
  //       dynamic-dtype buffers.
  DType dtype = resolveDType();
  if (dtype == DType::invalid)
    return emitError(loc)
           << "Buffers with unbound dtype are not yet supported: " << *this;

  Optional<int64_t> sizeOr = resolveSize();
  if (!sizeOr.has_value())
    return emitError(loc) << "Buffers with unbound size are not yet supported: "
                          << *this;

  return dataEquals(loc, dtype, *sizeOr, lhsData, rhsData);
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

PointerType PointerType::get(TypedAttr elementType) {
  return PointerType::get(elementType.getContext(), elementType);
}

PointerType PointerType::get(Type elementType) {
  return PointerType::get(TypeConstantAttr::get(elementType));
}

LogicalResult
PointerType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                    TypedAttr dtype) {
  if (dtype && !dtype.getType().isa<MLIRTypeType>())
    return emitError() << "type parameter for pointer must be a !kgen.mlirtype";
  return success();
}

void PointerType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getElementType());
}

Type PointerType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                              ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 1 && replTypes.empty());
  return PointerType::get(replAttrs[0]);
}

Type PointerType::resolveElementType() const {
  if (auto typeCst = getElementType().dyn_cast_or_null<TypeConstantAttr>())
    return typeCst.getValue();
  return nullptr;
}

/// Implements `OpaqueObjectInterface::populate`. Because we don't know anything
/// about the pointer's size, for now, we will leave this as impossible to
/// populate. This could be changed in the future by passing the size of the
/// backing buffer into `tag`.
LogicalResult PointerType::populate(Location loc, InputGenKind kind,
                                    Attribute tag, void *obj) const {
  return emitError(loc) << "could not populate type: " << *this;
}

/// This implements `OpaqueObjectInterface::destroy`. This deallocates any
/// memory allocated in `populate`. For now, we can leave it alone because
/// PointerType cannot be populated.
void PointerType::destroy(Attribute tag, void *obj) const { return; }

/// Implements `OpaqueObjectInterface::populate`. We care about the size of a
/// pointer, and all pointers have the same size on a given platform.
FailureOr<size_t> PointerType::getSizeInBytes(Location loc,
                                              Attribute tag) const {
  // FIXME: This is incorrect once we start talking about cross-compilation.
  //        c.f. #2717
  return sizeof(void *);
}

FailureOr<bool> PointerType::equals(Location loc, Attribute tag, void *lhsData,
                                    void *rhsData) const {
  return mlir::emitError(loc) << "could not compare pointers of: " << *this;
}

//===----------------------------------------------------------------------===//
// Dialect Type Parsing and Printing
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#define GET_TYPEDEF_CLASSES
#include "KGEN/MetaDialect/MetaTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// MetaDialect type support
//===----------------------------------------------------------------------===//

void MetaDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/MetaDialect/MetaTypes.cpp.inc"
      >();
}
