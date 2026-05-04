//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPATTRS_H
#define KGEN_POPDIALECT_POPATTRS_H

#include "KGEN/KGENDialect/KGENAttrInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/POPDialect/POPEnums.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/IPInt.h"
#include "Support/IPRational.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/raw_ostream.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/POPDialect/POPAttrs.h.inc"

#endif // GEN_POPDIALECT_POPATTRS_H
