//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_CODIALECT_COOPS_H
#define KGEN_CODIALECT_COOPS_H

#include "KGEN/CODialect/COTypes.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/CODialect/CO.h.inc"

#endif // KGEN_CODIALECT_COOPS_H
