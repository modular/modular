//===- KGENAttrs.h --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENATTRIBUTES_H
#define KGEN_KGENATTRIBUTES_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

// Pull in all enum type definitions and utility function declarations.
#include "KGEN/KGENDialect/KGENEnums.h.inc"

namespace M::KGEN {
inline raw_ostream &operator<<(raw_ostream &os, PEO opcode) {
  return os << stringifyEnum(opcode);
}

/// Scan the body of the specified operation checking invariants on parameters,
/// diagnosing errors and returning failure if so.  This is used by verifiers
/// for ops with bodies, like kgen.generator.
LogicalResult checkParametersInOpBody(Operation *op);

/// Parse a "colon type" production if present or default to si64 if not.  This
/// is commonly used in our parameter representation.
ParseResult parseColonTypeOrSI64(OpAsmParser &parser, Type &type);

/// print `: <type>` or elide it entirely if type is an si64.
void printColonTypeOrSI64(OpAsmPrinter &p, Type type);

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the printed syntax easier to grok.
void printParamValue(OpAsmPrinter &p, Attribute value, Type type);

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the parsed syntax easier to grok.
ParseResult parseParamValue(OpAsmParser &p, Attribute &value, Type type);

ParseResult parseTypedParamValue(OpAsmParser &p, Attribute &value,
                                 Type &resultType);
void printTypedParamValue(OpAsmPrinter &p, Operation *, Attribute value,
                          Type resultType);

} // namespace M::KGEN

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.h.inc"

#endif // KGEN_KGENATTRIBUTES_H
