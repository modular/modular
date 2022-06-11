//===- KGENAttrs.h --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENATTRIBUTES_H
#define KGEN_KGENATTRIBUTES_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.h.inc"

namespace M::KGEN {
/// Scan the body of the specified operation checking invariants on parameters,
/// diagnosing errors and returning failure if so.  This is used by verifiers
/// for ops with bodies, like kgen.generator.
LogicalResult checkParametersInOpBody(Operation *op);
} // namespace M::KGEN

#endif // KGEN_KGENATTRIBUTES_H
