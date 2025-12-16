//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Modular version of LogicalResult which has some
// features that upstream MLIR does not.
//
// TODO: Upstream this into MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_LOGICALRESULT_H
#define SUPPORT_LOGICALRESULT_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/LogicalResult.h"

namespace M {
// Types.
using mlir::FailureOr;
using mlir::LogicalResult;
using mlir::ParseResult;

// Global functions.
using mlir::failed;
using mlir::failure;
using mlir::succeeded;

//===----------------------------------------------------------------------===//
// Reimplemented success()
//
// We reimplement success() so it can integrate with other types like ErrorOr
// which can't convert a failure state.

/// Utility function to generate a LogicalResult. If isSuccess is true a
/// `success` result is generated, otherwise a 'failure' result is generated.
inline LogicalResult success(bool isSuccess) {
  return LogicalResult::success(isSuccess);
}

/// This type is returned by `success()`, it is always successful!
struct SuccessType {
  // This implicitly converts to LogicalResult.
  /*implicit*/ operator LogicalResult() const { return success(true); }
  /*implicit*/ operator ParseResult() const { return success(true); }
};

/// Return a success indicator.
inline SuccessType success() { return SuccessType(); }

} // namespace M

#endif // SUPPORT_LOGICALRESULT_H
