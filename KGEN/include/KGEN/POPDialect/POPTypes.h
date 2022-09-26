//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPTYPES_H
#define KGEN_POPDIALECT_POPTYPES_H

#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/SubElementInterfaces.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "KGEN/POPDialect/POPTypes.h.inc"

#endif // GEN_POPDIALECT_POPTYPES_H
