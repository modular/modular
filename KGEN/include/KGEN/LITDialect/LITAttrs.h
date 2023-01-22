//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITATTRS_H
#define KGEN_LITDIALECT_LITATTRS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace M::KGEN {
class DeclRefType;
} // namespace M::KGEN

#define GET_ATTRDEF_CLASSES
#include "KGEN/LITDialect/LITAttrs.h.inc"

#endif // KGEN_LITDIALECT_LITATTRS_H
