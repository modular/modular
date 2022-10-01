//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares utility functions primarily for parsing, printing and
// verifying KGEN related operations and types.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENUTILS_H
#define KGEN_KGENDIALECT_KGENUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Types.h"

namespace mlir {
class FunctionOpInterface;
}

namespace M::KGEN {
class ConstraintArrayAttr;
class ParamBindArrayAttr;
class ParamDeclAttr;
class GeneratorInterfaceOp;
class ParamBindArrayAttr;
class ParamDeclArrayAttr;
class KGENDeclInterface;
class TypeArrayAttr;
class SignatureType;

/// Return the string form for an attribute value that is printed in a <>
/// context in the .mlir file.
std::string getParamAsString(Attribute value);

/// Parse a type in a KGEN context, handling sugar like "dtype" for
/// "!kgen.dtype" etc.
ParseResult parseKGENType(AsmParser &parser, Type &type);

/// Print `type` using KGEN specific type sugars.
void printKGENType(raw_ostream &os, Type type);

/// Parse a "colon type" production if present or default to `index` type if
/// not.  This is commonly used in our parameter representation.
ParseResult parseColonTypeOrIndex(AsmParser &parser, Type &type);

/// Print `: <type>` or elide it entirely if type is an `index` type.
void printColonTypeOrIndex(raw_ostream &os, Type type);

//===----------------------------------------------------------------------===//
// Parameter Printing and Parsing
//===----------------------------------------------------------------------===//

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
ParseResult parseParamValue(AsmParser &p, FailureOr<TypedAttr> &result,
                            Type type);

/// Parse ":type 42" or "42" and default to index type.
ParseResult parseParamValueDefaultingToIndex(AsmParser &p, TypedAttr &value);

/// Print a parameter value that is known to have `dtype` type.
void printDTypeParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `dtype` type.
ParseResult parseDTypeParamValue(AsmParser &p, FailureOr<TypedAttr> &value);

/// Print a parameter value that is known to have `type` type.
void printTypeParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `type` type.
ParseResult parseTypeParamValue(AsmParser &p, FailureOr<TypedAttr> &value);

/// Print a parameter value that is known to have `index` type.
void printIndexParamValue(AsmPrinter &p, Operation *op, Attribute value);
void printIndexParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `index` type.
ParseResult parseIndexParamValue(AsmParser &p, TypedAttr &value);
ParseResult parseIndexParamValue(AsmParser &p, FailureOr<TypedAttr> &value);

/// Print a parameter value that either has an index type or is null (which
/// prints as a `?`).
void printOptionalIndexParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to be an index type or a `?` which
/// results in a null attribute.
ParseResult parseOptionalIndexParamValue(AsmParser &p,
                                         FailureOr<TypedAttr> &result);

/// Print a parameter value that either has `dtype` type or is null (which
/// prints as a `?`).
void printOptionalDTypeParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to be a `dtype` type or a `?` which
/// results in a null attribute.
ParseResult parseOptionalDTypeParamValue(AsmParser &p,
                                         FailureOr<TypedAttr> &result);

/// Parse and print a ParamDeclAttr which has syntactic form `name (: type)?`.
ParseResult parseParamDecl(AsmParser &p, FailureOr<ParamDeclAttr> &result);
ParseResult parseParamDecl(AsmParser &p, ParamDeclAttr &result);
void printParamDecl(raw_ostream &os, ParamDeclAttr decl);
void printParamDecl(AsmPrinter &p, ParamDeclAttr decl);

/// Parse and print ParamDeclArrayAttr as a canonical list of comma separated
/// information.
void printParamDecls(raw_ostream &os, ParamDeclArrayAttr decls);
ParseResult parseParamDecls(AsmParser &p, ParamDeclArrayAttr &result);

/// Parse and print a parameter specification on a generator or region type.
ParseResult parseOptionalParameterSpec(AsmParser &parser,
                                       ParamDeclArrayAttr &inputParamDecls,
                                       TypeArrayAttr &resultParamTypes);
void printOptionalParameterSpec(raw_ostream &os,
                                ParamDeclArrayAttr inputParamDecls,
                                TypeArrayAttr resultParamTypes);
void printOptionalParameterSpec(AsmPrinter &p, Operation *op,
                                ParamDeclArrayAttr paramDecls,
                                TypeArrayAttr resultParamTypes);

/// Parse and print a constraint specification if present.
ParseResult parseOptionalConstraints(OpAsmParser &p,
                                     ConstraintArrayAttr &constraints);
void printOptionalConstraints(OpAsmPrinter &p, Operation *op,
                              ConstraintArrayAttr constraints);

/// Parse and print a parameter binding list if present.
ParseResult parseParamBinds(AsmParser &p, ParamBindArrayAttr &paramBinds);
void printParamBinds(AsmPrinter &p, ParamBindArrayAttr paramBinds);

//===----------------------------------------------------------------------===//
// Logic shared between funcs, generators, and generator interfaces
//===----------------------------------------------------------------------===//

enum class GeneratorOrFuncKind {
  func,
  generator,
  interface,
  precompiled,

  // HLKGEN dialect
  hlgenerator,
};

/// Parse the MLIR syntax for a kgen.generator, kgen.func and related
/// operators.
ParseResult parseGeneratorOrFunc(OpAsmParser &parser, OperationState &result,
                                 GeneratorOrFuncKind opKind);
void printGeneratorOrFunc(OpAsmPrinter &p, mlir::FunctionOpInterface op);

/// Check that the specified generator/interfaces matches signature
/// information with the other interface.
LogicalResult verifyDeclMatchesInterface(const char *originatorName,
                                         KGENDeclInterface originatorDecl,
                                         const char *interfaceName,
                                         GeneratorInterfaceOp interfaceDecl);

/// Check that the specified declaration signatures match, checking the
/// parameter and value type information.
LogicalResult verifyDeclSignaturesMatch(const char *originatorName,
                                        SignatureType originatorSignature,
                                        Location originatorLoc,
                                        const char *interfaceName,
                                        SignatureType targetSignature,
                                        Location targetLoc);

/// Check that the parameter declarations match.
LogicalResult
verifyParamDeclsMatch(const char *originatorName,
                      ArrayRef<ParamDeclAttr> originatorParamDecls,
                      Location originatorLoc, const char *targetName,
                      ParamDeclArrayAttr targetParamDecls, Location targetLoc);

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENUTILS_H
