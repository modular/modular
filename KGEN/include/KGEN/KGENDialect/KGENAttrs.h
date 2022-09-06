//===- KGEN/KGENDialect/KGENAttrs.h ---------------------------------------===//
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
class ConstraintAttr;
class ConcreteTypeConstantAttr;
class ParameterizedTypeConstantAttr;
class DTypeConstantAttr;
class ParamDeclAttr;

inline raw_ostream &operator<<(raw_ostream &os, POC opcode) {
  return os << stringifyEnum(opcode);
}

inline raw_ostream &operator<<(raw_ostream &os, InputGenKind opcode) {
  return os << stringifyEnum(opcode);
}

/// Return the `paramDecls` array of ParamDeclAttr values if the specified
/// operation has it, or an empty array otherwise.
ArrayRef<ParamDeclAttr> getParamDecls(Operation *op);

/// We expect all parameter expressions to simplify down to concrete constants
/// after elaboration.  We don't want anything left as a ParamOperatorAttr or
/// ParamDeclRefAttr or ParameterizedTypeConstantAttr.
static inline bool isSimpleConstant(Attribute attr) {
  return attr.isa<FloatAttr, IntegerAttr, StringAttr, DTypeConstantAttr,
                  ConcreteTypeConstantAttr>();
}

//===----------------------------------------------------------------------===//
// Parameter Printing and Parsing
//

/// Return the string form for an attribute value that is printed in a <>
/// context in the .mlir file.
std::string getParamAsString(TypedAttr value);

/// Parse a "colon type" production if present or default to `index` type if
/// not.  This is commonly used in our parameter representation.
ParseResult parseColonTypeOrIndex(AsmParser &parser, Type &type);

/// Print `: <type>` or elide it entirely if type is an `index` type.
void printColonTypeOrIndex(AsmPrinter &p, Type type);

/// Print a parameter name correctly, using a double quoted syntax if it
/// conflicts with an MLIR or KGEN keyword, or a bareword otherwise.
void printParamName(StringRef name, raw_ostream &os);
void printParamName(AsmPrinter &p, StringRef name);

/// Parse a parameter name as either a keyword or double quoted string.
ParseResult parseParamName(AsmParser &p, StringAttr &name);
ParseResult parseParamName(AsmParser &p, FailureOr<StringAttr> &name);

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the printed syntax easier to grok.
void printParamValue(AsmPrinter &p, TypedAttr value, Type type = {});
void printParamValue(TypedAttr value, raw_ostream &os);

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the parsed syntax easier to grok.
ParseResult parseParamValue(AsmParser &p, TypedAttr &value, Type type);
ParseResult parseParamValue(AsmParser &p, FailureOr<TypedAttr> &value,
                            Type type);

/// Print a parameter value that is known to have `type` type.
void printTypeParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `type` type.
ParseResult parseTypeParamValue(AsmParser &p, FailureOr<TypedAttr> &value);

/// Print a parameter value that is known to have `index` type.
void printIndexParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `index` type.
ParseResult parseIndexParamValue(AsmParser &p, FailureOr<TypedAttr> &value);

/// Print a parameter value that either has an index type or is null (which
/// prints as a `?`).
void printOptionalIndexParamValue(AsmPrinter &p, Attribute value);

/// Parse a parameter value that is known to be an index type or a `?` which
/// results in a null attribute.
ParseResult parseOptionalIndexParamValue(AsmParser &p,
                                         FailureOr<TypedAttr> &result);

/// Print a histogram.
void printHistogram(AsmPrinter &p,
                    ArrayRef<std::pair<Attribute, uint64_t>> histogram);
/// Parse a histogram.
FailureOr<SmallVector<std::pair<Attribute, uint64_t>>>
parseHistogram(AsmParser &p);

} // namespace M::KGEN

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.h.inc"

#endif // KGEN_KGENATTRIBUTES_H
