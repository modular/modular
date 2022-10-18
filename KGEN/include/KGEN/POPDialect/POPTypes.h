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

//===----------------------------------------------------------------------===//
// Pretty Type Parsing and Printing
//===----------------------------------------------------------------------===//

namespace M::KGEN::POP {
/// Try to parse a pretty type or a standard MLIR type. A pretty type is a POP
/// type without the dialect prefix.
ParseResult parsePrettyType(AsmParser &p, FailureOr<TypedAttr> &typeExpr);
/// Try to print a pretty type or a standard MLIR type. A pretty type is a POP
/// type without the dialect prefix.
void printPrettyType(AsmPrinter &p, TypedAttr typeExpr);
} // namespace M::KGEN::POP

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KGEN/POPDialect/POPTypes.h.inc"

#endif // GEN_POPDIALECT_POPTYPES_H
