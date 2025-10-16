//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/Error.h"
#include "mlir/IR/Diagnostics.h"

using namespace M;

void KGEN::emitLimitedError(function_ref<InFlightDiagnostic()> emitError,
                            ErrorLimit &limit) {
  ++limit.errorCount;
  if (limit.errorLimit > 0 && limit.errorCount > limit.errorLimit) {
    return;
  }

  InFlightDiagnostic diag = emitError();

  // Emit message if hits error limit.
  if (limit.errorCount == limit.errorLimit)
    diag.attachNote() << "too many errors emitted, stopping now";
}
