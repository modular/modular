//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENINTERFACES_H
#define KGEN_KGENDIALECT_KGENINTERFACES_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"

namespace M::KGEN {
class DeclInterface;
class KGENCallOpInterface;

namespace impl {
LogicalResult verifyCallOp(KGENCallOpInterface op);
LogicalResult verifyIfTopLevel(DeclInterface decl,
                               SymbolTableCollection &symtab);
} // namespace impl
} // namespace M::KGEN

#include "KGEN/KGENDialect/KGENInterfaces.h.inc"

#endif // KGEN_KGENDIALECT_KGENINTERFACES_H
