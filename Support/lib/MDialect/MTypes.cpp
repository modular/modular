//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MTypes.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

/// Parse rank 1 dimension followed by an 'x'.
static ParseResult parseSizeX(AsmParser &p, FailureOr<int64_t> &size) {
  SmallVector<int64_t> dims;
  llvm::SMLoc curLoc = p.getCurrentLocation();
  if (p.parseDimensionList(dims, /*allowDynamic=*/false))
    return failure();
  if (dims.size() != 1)
    return p.emitError(curLoc, "expected a single dimension");
  size = dims.front();
  return success();
}

static void printSizeX(AsmPrinter &p, int64_t size) { p << size << 'x'; }

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Support/MDialect/MTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// MDialect
//===----------------------------------------------------------------------===//

void MDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Support/MDialect/MTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

/// Ensure the array size is non-negative.
LogicalResult ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                                int64_t size, Type elementType) {
  if (size < 0)
    return emitError() << "invalid array size: " << size;
  return success();
}

/// An array type always has rank 1.
bool ArrayType::hasRank() const { return true; }

/// The shape of an array is always [size].
ArrayRef<int64_t> ArrayType::getShape() const {
  // We need to return a const reference.
  return static_cast<detail::ArrayTypeStorage *>(getImpl())->size;
}

/// Clone the type. Expect the shape to always be rank 1.
ShapedType ArrayType::cloneWith(Optional<ArrayRef<int64_t>> shape,
                                Type elementType) const {
  assert(!shape || shape->size() == 1);
  if (shape)
    return get(shape->front(), elementType);
  return get(getSize(), elementType);
}

ArrayType ArrayType::get(int64_t size, Type elementType) {
  return get(elementType.getContext(), size, elementType);
}
