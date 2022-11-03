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
#include "Support/LogicalResult.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
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
static Optional<KGENDType> getResolvedDType(Op *op) {
  if (auto dtype = dyn_cast_if_present<DTypeConstantAttr>(op->getDType()))
    return dtype.getDType();
  return {};
}

template <typename Op>
static POP::PointerType getPointerType(Op *op) {
  Type elementType;
  if (TypedAttr dtype = op->getDType())
    elementType = POP::SIMDType::get(1, dtype);
  else {
    MLIRContext *ctx = op->getContext();
    elementType =
        POP::SIMDType::get(ctx, Builder(ctx).getIndexAttr(1),
                           DTypeConstantAttr::get(ctx, DType::invalid));
  }

  return POP::PointerType::get(elementType);
}

template <typename Op>
static POP::SIMDType getElementType(Op *op) {
  assert(op->getDType() && "expected buffer with known dtype");
  return POP::SIMDType::get(1, op->getDType());
}

//===----------------------------------------------------------------------===//
// NDBufferType
//===----------------------------------------------------------------------===//

/// Print an array of parameter values that either has an index type or is null
/// (which prints as a `?`).
void printOptionalIndicesParamValue(AsmPrinter &p, ArrayRef<TypedAttr> value) {
  p << '[';
  interleaveComma(
      value, p, [&](TypedAttr attr) { printOptionalIndexParamValue(p, attr); });
  p << ']';
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
NDBufferType::verify(function_ref<mlir::InFlightDiagnostic()> emitError,
                     ArrayRef<TypedAttr> shape, TypedAttr dtype) {
  if (shape.empty())
    return emitError() << "shape parameter for ndbuffer must not be empty";
  if (shape.size() > NDBufferType::getMaximumRank())
    return emitError()
           << "shape parameter exceeds the maximum rank of the ndbuffer type";
  for (auto size : shape) {
    if (size && !size.getType().isIndex())
      return emitError()
             << "size parameter for ndbuffer must have type `index`";
    auto sizeInt = dyn_cast_if_present<IntegerAttr>(size);
    if (sizeInt && sizeInt.getInt() <= 0)
      return emitError() << "size parameter for ndbuffer must be positive";
  }
  if (dtype && !dtype.getType().isa<DTypeType>())
    return emitError() << "type parameter for ndbuffer must be a !kgen.dtype";
  return success();
}

void NDBufferType::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (TypedAttr shape : getShape())
    walkAttrsFn(shape);
  walkAttrsFn(getDType());
}

Type NDBufferType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                               ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == (getRank() + 1) && replTypes.empty());
  SmallVector<TypedAttr, 5> shapeAttrs;
  shapeAttrs.reserve(replAttrs.size() - 1);
  for (auto attr : llvm::drop_end(replAttrs))
    shapeAttrs.push_back(attr);

  return NDBufferType::get(getContext(), shapeAttrs, replAttrs.back());
}

size_t NDBufferType::getRank() const { return getShape().size(); }

Optional<int64_t> NDBufferType::getResolvedSize() const {
  int64_t size = 1;
  for (TypedAttr shape : getShape()) {
    auto intAttr = dyn_cast_if_present<IntegerAttr>(shape);
    if (!intAttr)
      return {};
    size *= intAttr.getInt();
  }
  return size;
}

Optional<KGENDType> NDBufferType::getResolvedDType() const {
  return ::getResolvedDType(this);
}

POP::PointerType NDBufferType::getPointerType() const {
  return ::getPointerType(this);
}

POP::SIMDType NDBufferType::getElementType() const {
  return ::getElementType(this);
}

NDBufferType NDBufferType::get(ArrayRef<TypedAttr> shape, TypedAttr dtype) {
  return get(dtype.getContext(), shape, dtype);
}

NDBufferType NDBufferType::get(MLIRContext *ctx, ArrayRef<TypedAttr> shape,
                               KGENDType dtype) {
  return get(ctx, shape, DTypeConstantAttr::get(ctx, dtype));
}

NDBufferType NDBufferType::get(MLIRContext *ctx, ArrayRef<int64_t> shape,
                               KGENDType dtype) {
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
