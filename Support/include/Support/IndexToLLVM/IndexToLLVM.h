//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_INDEXTOLLVM_H
#define SUPPORT_INDEXTOLLVM_H

#include "mlir/Pass/Pass.h"
#include <memory>

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

namespace M::index {
#define GEN_PASS_DECL_INDEXTOLLVM
#define GEN_PASS_REGISTRATION
#include "Support/IndexToLLVM/IndexToLLVM.h.inc"
} // namespace M::index

#endif // SUPPORT_INDEXTOLLVM_H
