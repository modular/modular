//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITAttrs.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// ArgumentDefaultAttr
//===----------------------------------------------------------------------===//

/// Reject default arguments with negative indices.
LogicalResult
DefaultArgumentAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            IntegerAttr index, TypedAttr value) {
  if (index.getValue().isNegative())
    return emitError() << "index cannot be negative";

  return success();
}

/// Reject default argument arrays that include multiple defaults for the same
/// argument index.
LogicalResult
DefaultArgumentArrayAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<DefaultArgumentAttr> attrs) {
  llvm::SmallDenseSet<int64_t> indices;
  for (const DefaultArgumentAttr &attr : attrs) {
    int64_t index = attr.getIndex().getInt();
    if (!indices.insert(index).second)
      return emitError() << "cannot specify more than one default argument for "
                            "the same index "
                         << index;
  }

  return success();
}
