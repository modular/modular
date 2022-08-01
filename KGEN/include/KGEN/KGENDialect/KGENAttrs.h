//===- KGENAttrs.h --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the core KGEN attribute classes, provides implementation
// logic for working with them, and helpers for defining operations that take
// them.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENATTRIBUTES_H
#define KGEN_KGENATTRIBUTES_H

#include "Support/ForwardDecls.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

// Pull in all enum type definitions and utility function declarations.
#include "KGEN/KGENDialect/KGENEnums.h.inc"

namespace M::KGEN {
class ParamDeclAttr;
inline raw_ostream &operator<<(raw_ostream &os, POC opcode) {
  return os << stringifyEnum(opcode);
}

/// Given a kernel, generator, or generator interface operation, return an array
/// of `ParamDeclAttr`s for the inputs and the array of `ParamDeclAttr`s for the
/// result parameters.  A kernel will always never have input parameters.
std::pair<ArrayRef<Attribute>, ArrayRef<Attribute>>
getDeclParameterInfo(Operation *decl);

/// Return the `paramDecls` array of ParamDeclAttr values if the specified
/// operation has it, or an empty array otherwise.
ArrayRef<Attribute> getParamDecls(Operation *op);

/// Return the `paramDecls` array of ParamDeclAttr values if the specified
/// operation has it, or an empty array otherwise.  This handles casting each
/// element of the attribute list, which requires building a new SmallVector.
SmallVector<ParamDeclAttr, 4> getParamDeclsCasted(Operation *op);

/// Given a kernel, generator or interface operation, return the constraints
/// imposed on it.  For a kernel this is always empty.
ArrayRef<Attribute> getDeclConstraints(Operation *decl);

//===----------------------------------------------------------------------===//
// Parameter Printing and Parsing
//

/// Parse a "colon type" production if present or default to `index` type if
/// not.  This is commonly used in our parameter representation.
ParseResult parseColonTypeOrIndex(AsmParser &parser, Type &type);

/// print `: <type>` or elide it entirely if type is an `index` type.
void printColonTypeOrIndex(AsmPrinter &p, Type type);

/// Print a parameter name correctly, using a double quoted syntax if it
/// conflicts with an MLIR or KGEN keyword, or a bareword otherwise.
void printParamName(AsmPrinter &p, StringRef name);

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the printed syntax easier to grok.
void printParamValue(AsmPrinter &p, Attribute value);

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the parsed syntax easier to grok.
ParseResult parseParamValue(AsmParser &p, TypedAttr &value, Type type);

/// Print a parameter value that is known to be an index type.
void printIndexParamValue(AsmPrinter &p, Attribute value);

/// Parse a parameter value that is known to be an index type.
ParseResult parseIndexParamValue(AsmParser &p, FailureOr<TypedAttr> &value);

/// Print a parameter value that either has an index type or is null (which
/// prints as a `?`).
void printOptionalIndexParamValue(AsmPrinter &p, Attribute value);

/// Parse a parameter value that is known to be an index type or a `?` which
/// results in a null attribute.
ParseResult parseOptionalIndexParamValue(AsmParser &p,
                                         FailureOr<TypedAttr> &result);

} // namespace M::KGEN

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.h.inc"

#endif // KGEN_KGENATTRIBUTES_H
