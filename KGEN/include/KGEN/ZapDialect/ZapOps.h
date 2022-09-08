//===- KGEN/ZapDialect/ZapOps.h -------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ZAPDIALECT_ZAPOPS_H
#define KGEN_ZAPDIALECT_ZAPOPS_H

#include "KGEN/MetaDialect/MetaTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/ZapDialect/Zap.h.inc"

#endif // KGEN_ZAPDIALECT_ZAPOPS_H
