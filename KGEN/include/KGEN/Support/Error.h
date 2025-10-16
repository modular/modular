//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_ERROR_H
#define KGEN_SUPPORT_ERROR_H

#include "Support/LLVMCompilerForwardDecls.h"
namespace M::KGEN {
struct ErrorLimit {
  int errorLimit = 0;
  int errorCount = 0;
};

void emitLimitedError(function_ref<InFlightDiagnostic()> emitError,
                      ErrorLimit &limit);

} // namespace M::KGEN

#endif // KGEN_SUPPORT_ERROR_H
