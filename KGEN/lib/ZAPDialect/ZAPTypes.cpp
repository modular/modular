//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ZAPDialect/ZAPTypes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ZAPDialect/ZAPDialect.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace ZAP;

//===----------------------------------------------------------------------===//
// ZAPDialect
//===----------------------------------------------------------------------===//

void ZAPDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/ZAPDialect/ZAPTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// MemoryLikeType
//===----------------------------------------------------------------------===//

template <typename Op>
static Optional<DType> getResolvedDType(Op *op) {
  if (auto dtype =
          op->getDType().template dyn_cast_or_null<DTypeConstantAttr>())
    return dtype.getDType();
  return {};
}

template <typename Op>
static POP::PointerType getPointerType(Op *op) {
  Type elementType;
  if (TypedAttr dtype = op->getDType())
    elementType = POP::ScalarType::get(dtype);
  else
    elementType = POP::ScalarType::get(op->getContext(), DType::invalid);

  return POP::PointerType::get(elementType);
}

template <typename Op>
static POP::ScalarType getElementType(Op *op) {
  assert(op->getDType() && "expected buffer with known dtype");
  return POP::ScalarType::get(op->getDType());
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

LogicalResult
BufferType::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
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

Optional<int64_t> BufferType::getResolvedSize() const {
  if (auto intAttr = dyn_cast_if_present<IntegerAttr>(getSize()))
    return intAttr.getInt();
  return {};
}

Optional<DType> BufferType::getResolvedDType() const {
  if (auto dtype = dyn_cast_if_present<DTypeConstantAttr>(getDType()))
    return dtype.getDType();
  return {};
}

POP::PointerType BufferType::getPointerType() const {
  return ::getPointerType(this);
}

POP::ScalarType BufferType::getElementType() const {
  return ::getElementType(this);
}

BufferType BufferType::get(TypedAttr size, TypedAttr dtype) {
  return get(size.getContext(), size, dtype);
}

BufferType BufferType::get(MLIRContext *ctx, int64_t size, DType dtype) {
  return get(OpBuilder(ctx).getIndexAttr(size),
             DTypeConstantAttr::get(ctx, dtype));
}

//===----------------------------------------------------------------------===//
// TensorType
//===----------------------------------------------------------------------===//

/// Print an array of parameter values that either has an index type or is null
/// (which prints as a `?`).
void printOptionalIndicesParamValue(AsmPrinter &p, ArrayRef<TypedAttr> value) {
  p << "[";
  interleaveComma(
      value, p, [&](TypedAttr attr) { printOptionalIndexParamValue(p, attr); });
  p << "]";
}

/// Parse an array of parameter values that is known to be an index type or a
/// `?` which results in a null attribute.
ParseResult
parseOptionalIndicesParamValue(AsmParser &p,
                               FailureOr<SmallVector<TypedAttr>> &result) {
  result.emplace();
  return p.parseCommaSeparatedList(
      AsmParser::Delimiter::Square, [&]() -> ParseResult {
        FailureOr<TypedAttr> attr;
        if (auto err = parseOptionalIndexParamValue(p, attr); failed(err))
          return err;
        result->emplace_back(attr.value());
        return success();
      });
}

LogicalResult
TensorType::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
                   ArrayRef<TypedAttr> shape, TypedAttr dtype) {
  if (shape.empty())
    return emitError() << "shape parameter for tensor must not be empty";
  if (shape.size() > TensorType::getMaximumRank())
    return emitError()
           << "shape parameter exceeds the maximum rank of the tensor type";
  for (auto size : shape) {
    if (size && !size.getType().isIndex())
      return emitError() << "size parameter for tensor must have type `index`";
    auto sizeInt = dyn_cast_if_present<IntegerAttr>(size);
    if (sizeInt && sizeInt.getInt() <= 0)
      return emitError() << "size parameter for tensor must be positive";
  }
  if (dtype && !dtype.getType().isa<DTypeType>())
    return emitError() << "type parameter for tensor must be a !kgen.dtype";
  return success();
}

void TensorType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (TypedAttr shape : getShape())
    walkAttrsFn(shape);
  walkAttrsFn(getDType());
}

Type TensorType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                             ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == (getRank() + 1) && replTypes.empty());
  SmallVector<TypedAttr> castedShapeAttrs;
  for (auto attr : llvm::drop_end(replAttrs)) {
    castedShapeAttrs.push_back(attr.cast<TypedAttr>());
    // Reject attempts to change an operand to something that isn't a TypedAttr.
    if (!castedShapeAttrs.back())
      return {};
  }

  return TensorType::get(getContext(), castedShapeAttrs, replAttrs.back());
}

size_t TensorType::getRank() const { return getShape().size(); }

Optional<int64_t> TensorType::getResolvedSize() const {
  int64_t size = 1;
  for (TypedAttr shape : getShape()) {
    auto intAttr = dyn_cast_if_present<IntegerAttr>(shape);
    if (!intAttr)
      return {};
    size *= intAttr.getInt();
  }
  return size;
}

Optional<DType> TensorType::getResolvedDType() const {
  return ::getResolvedDType(this);
}

POP::PointerType TensorType::getPointerType() const {
  return ::getPointerType(this);
}

POP::ScalarType TensorType::getElementType() const {
  return ::getElementType(this);
}

TensorType TensorType::get(ArrayRef<TypedAttr> shape, TypedAttr dtype) {
  return get(dtype.getContext(), shape, dtype);
}

TensorType TensorType::get(MLIRContext *ctx, ArrayRef<TypedAttr> shape,
                           DType dtype) {
  return get(ctx, shape, DTypeConstantAttr::get(ctx, dtype));
}

TensorType TensorType::get(MLIRContext *ctx, ArrayRef<int64_t> shape,
                           DType dtype) {
  SmallVector<TypedAttr, 5> shapeAttr;
  llvm::transform(shape, std::back_inserter(shapeAttr),
                  [&](int64_t dim) { return Builder(ctx).getIndexAttr(dim); });
  return get(ctx, shapeAttr, dtype);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/ZAPDialect/ZAPTypes.cpp.inc"
