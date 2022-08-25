//===- KGEN/POPDialect/POPAttrs.h -----------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPATTRS_H
#define KGEN_POPDIALECT_POPATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/SubElementInterfaces.h"

#include "KGEN/POPDialect/POPEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "KGEN/POPDialect/POPAttrs.h.inc"

#endif // GEN_POPDIALECT_POPATTRS_H
