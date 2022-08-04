//===- POPOps.h -----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Meta dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_POPOPS_H
#define KGEN_KGENDIALECT_POPOPS_H

#include "KGEN/MetaDialect/MetaTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "KGEN/POPDialect/POP.h.inc"

#endif // KGEN_KGENDIALECT_POPOPS_H
