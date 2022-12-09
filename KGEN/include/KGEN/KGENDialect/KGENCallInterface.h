//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENCALLINTERFACE_H
#define KGEN_KGENDIALECT_KGENCALLINTERFACE_H

#include "KGEN/KGENDialect/KGENDeclInterface.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"

namespace M::KGEN {
/// Iterator type for iterating call operation region body operations.
using CallRegionBodyIterator =
    llvm::mapped_iterator<Region *, DeclInterface (*)(Region &)>;
} // namespace M::KGEN

#include "KGEN/KGENDialect/KGENCallInterface.h.inc"

#endif // KGEN_KGENDIALECT_KGENCALLINTERFACE_H
