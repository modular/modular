//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENINTERFACES_H
#define KGEN_KGENDIALECT_KGENINTERFACES_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"

namespace M::KGEN {
class DeclInterface;
class KGENCallOpInterface;

/// Iterator type for iterating call operation region body operations.
using CallRegionBodyIterator =
    llvm::mapped_iterator<Region *, DeclInterface (*)(Region &)>;

namespace impl {
LogicalResult verifyCallOp(KGENCallOpInterface op);
} // namespace impl
} // namespace M::KGEN

#include "KGEN/KGENDialect/KGENInterfaces.h.inc"

#endif // KGEN_KGENDIALECT_KGENINTERFACES_H
