//===- MetaOps.h ----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Meta dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_METAOPS_H
#define KGEN_KGENDIALECT_METAOPS_H

#include "GenericML/Support/TensorEltType.h"
#include "KGEN/MetaDialect/MetaTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "KGEN/MetaDialect/Meta.h.inc"

#endif // KGEN_KGENDIALECT_METAOPS_H
