//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// A temporary alignment check macro for debugging #26118
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_LLVM_ALIGN_TO_MACRO_H
#define SUPPORT_LLVM_ALIGN_TO_MACRO_H

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

// TODO(#26118): Remove this check once the align bug is found.
#define CHECKED_LLVM_ALIGN_TO(OUT, SIZE, ALIGN)                                \
  if ((ALIGN) == 0) {                                                          \
    auto msg =                                                                 \
        llvm::Twine{"Alignment bug hit, please report in issue #26118: "} +    \
        llvm::Twine{LLVM_PRETTY_FUNCTION};                                     \
    llvm_unreachable(msg.str().c_str());                                       \
  }                                                                            \
  (OUT) = llvm::alignTo(SIZE, ALIGN)

#endif // SUPPORT_LLVM_ALIGN_TO_MACRO_H
