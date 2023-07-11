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

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/OpImplementation.h"

namespace M::KGEN {
class DeclInterface;
class FuncInterface;

/// This struct represents the data for an exported symbol.
struct ExportedSymbol {
  ExportedSymbol(bool isCExport = false) : isCExport(isCExport) {}

  /// A flag indicating whether the export is a C export.
  bool isCExport = false;
};

/// This type is a bit of a mouthful, add a useful alias for it.
using ExportMap = llvm::MapVector<StringAttr, ExportedSymbol>;

/// Given a module operation, return its exported symbols and aliases.
ExportMap getExportedSymbols(ModuleOp module);

/// Return the string form for an attribute value that is printed in a <>
/// context in the .mlir file.
std::string getParamAsString(Attribute value);

/// Parse a parameter of type kgen.string.
ParseResult parseStringParam(AsmParser &p, TypedAttr &value);

/// Print a parameter of type kgen.string.
void printStringParam(AsmPrinter &p, Operation *, Attribute value);

/// Parse a type in a KGEN context, handling sugar like "dtype" for
/// "!kgen.dtype" etc.
ParseResult parseKGENType(AsmParser &parser, Type &type);

/// Try to parse a specific KGEN type.
template <typename T>
ParseResult parseKGENType(AsmParser &parser, T &type) {
  Type value;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (failed(parseKGENType(parser, value)))
    return failure();
  if (auto expectedType = dyn_cast<T>(value))
    return type = expectedType, success();
  return parser.emitError(loc, "wrong KGEN type");
}

/// Print `type` using KGEN specific type sugars.
void printKGENType(AsmPrinter &p, Type type);
void printKGENType(raw_ostream &os, Type type);

/// Parse a "colon type" production if present or default to `index` type if
/// not.  This is commonly used in our parameter representation.
ParseResult parseColonTypeOrIndex(AsmParser &parser, Type &type);

/// Print `: <type>` or elide it entirely if type is an `index` type.
void printColonTypeOrIndex(AsmPrinter &p, Type type);

//===----------------------------------------------------------------------===//
// Parameter Printing and Parsing
//===----------------------------------------------------------------------===//

/// Print a parameter name correctly, using a double quoted syntax if it
/// conflicts with an MLIR or KGEN keyword, or a bareword otherwise.
void printParamName(AsmPrinter &p, StringRef name);

/// Parse a parameter name as either a keyword or double quoted string.
ParseResult parseParamName(AsmParser &p, StringAttr &name);

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the printed syntax easier to grok.
void printParamValue(AsmPrinter &p, TypedAttr value, Type type = {});
void printParamValue(AsmPrinter &p, Operation *op, TypedAttr value,
                     Type type = {});

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the parsed syntax easier to grok.
ParseResult parseParamValue(AsmParser &p, TypedAttr &value, Type type);

/// Parse ":type 42" or "42" and default to index type.
ParseResult parseParamValueDefaultingToIndex(AsmParser &p, TypedAttr &value);

/// Print a parameter value that is known to have `dtype` type.
void printDTypeParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `dtype` type.
ParseResult parseDTypeParamValue(AsmParser &p, TypedAttr &value);

/// Print a parameter value that is known to have `type` type.
void printTypeParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `type` type.
ParseResult parseTypeParamValue(AsmParser &p, TypedAttr &value);

/// Print a parameter value that is known to have `index` type.
void printIndexParamValue(AsmPrinter &p, Operation *op, Attribute value);
void printIndexParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `index` type.
ParseResult parseIndexParamValue(AsmParser &p, TypedAttr &value);

/// Parse a index-or-colon-type and then a parameter value of that type.
ParseResult parseColonTypeParamValue(AsmParser &p, TypedAttr &value);
void printColonTypeParamValue(AsmPrinter &p, TypedAttr value);

/// Parse and print a ParamDeclAttr which has syntactic form `name (: type)?`.
ParseResult parseParamDecl(AsmParser &p, ParamDeclAttr &result);
void printParamDecl(AsmPrinter &p, ParamDeclAttr decl);

/// Parse and print ParamDeclArrayAttr as a canonical list of comma separated
/// information.
void printParamDecls(AsmPrinter &p, ArrayRef<ParamDeclAttr> decls);
ParseResult parseParamDecls(AsmParser &p, ParamDeclArrayAttr &result);

/// Parse and print a parameter specification on a generator or region type. The
/// parameter spec includes input parameter declarations and types and
/// optionally result parameter declarations and types.
ParseResult parseOptionalParameterSpec(AsmParser &parser,
                                       ParamDeclArrayAttr &inputParamDecls);
ParseResult parseOptionalParameterSpec(AsmParser &parser,
                                       ParamDeclArrayAttr &inputParamDecls,
                                       ParamDeclArrayAttr &resultParamDecls);
void printOptionalParameterSpec(AsmPrinter &p,
                                ArrayRef<ParamDeclAttr> inputParamDecls,
                                ArrayRef<ParamDeclAttr> resultParams = {});
void printOptionalParameterSpec(AsmPrinter &p, Operation *op,
                                ArrayRef<ParamDeclAttr> inputParamDecls);

/// Parse and print an operand and result type list with metadata.
OptionalParseResult parseOptionalSignature(AsmParser &p,
                                           SignatureType &signature);
ParseResult parseSignature(AsmParser &p, SignatureType &signature);
ParseResult parseSignatureValues(AsmParser &p, TypeArrayAttr inputParamTypes,
                                 TypeArrayAttr resultParamTypes,
                                 SignatureType &signature);
void printSignature(AsmPrinter &p, SignatureType signature);
void printSignatureValues(AsmPrinter &p, SignatureType signature);

/// Parse and print a function signature with optional metadata.
ParseResult parseFunctionSignature(OpAsmParser &p,
                                   SmallVectorImpl<OpAsmParser::Argument> &args,
                                   ParamDeclArrayAttr &inputParams,
                                   ParamDeclArrayAttr &resultParams,
                                   FunctionType &functionType,
                                   SignatureType &signature);
void printFunctionSignature(OpAsmPrinter &p, Region &region,
                            ArrayRef<ParamDeclAttr> inputParams,
                            ArrayRef<ParamDeclAttr> resultParams,
                            FunctionType functionType, SignatureType signature,
                            StringArrayAttr valueParamNames = {});

/// Parse the always_inline related keywords if present.
ParseResult parseOptionalAlwaysInline(OpAsmParser &parser,
                                      AlwaysInlineLevelAttr &attr);
void printOptionalAlwaysInline(OpAsmPrinter &p, AlwaysInlineLevelAttr level);

/// Parse and print a constraint specification if present.
ParseResult parseOptionalConstraints(OpAsmParser &p,
                                     ConstraintArrayAttr &constraints);
void printOptionalConstraints(OpAsmPrinter &p, Operation *op,
                              ArrayRef<ConstraintAttr> constraints);

/// Parse and print a parameter binding list if present.
ParseResult parseParamBinds(AsmParser &p, ParamBindArrayAttr &paramBinds);
void printParamBinds(AsmPrinter &p, ArrayRef<ParamBindAttr> paramBinds);

/// Parse a list of parameter bindings without result parameters in <>'s
ParseResult parseOptionalParamBindSpec(AsmParser &p,
                                       ParamBindArrayAttr &paramValues);
void printOptionalParamBindSpec(AsmPrinter &p, ParamBindArrayAttr paramValues);

/// Parse and print a list of parameter values.
ParseResult parseParameterValues(AsmParser &p, ParameterExprArrayAttr &values);
ParseResult parseParameterValues(AsmParser &p,
                                 SmallVectorImpl<TypedAttr> &values);
void printParameterValues(OpAsmPrinter &p, Operation *op,
                          ParameterExprArrayAttr values);
void printParameterValues(AsmPrinter &p, ArrayRef<TypedAttr> values);

/// Parse and print a parametric callee and result parameter declarations.
ParseResult parseParametricCallee(OpAsmParser &p, TypedAttr &callee,
                                  ParamDeclArrayAttr &paramDecls);
void printParametricCallee(OpAsmPrinter &p, Operation *, TypedAttr callee,
                           ParamDeclArrayAttr paramDecls);

/// Parse an align parameter if present.
ParseResult parseOptionalAlignmentParamValue(AsmParser &p, TypedAttr &result);
void printOptionalAlignmentParamValue(AsmPrinter &p, Operation *op,
                                      TypedAttr alignment);

template <typename SequenceType>
ParseResult parseSequenceElements(AsmParser &p, SmallVector<TypedAttr> &values,
                                  SequenceType type) {
  auto elementType = ParamRefType::get(type.getElementType());
  return p.parseCommaSeparatedList(
      [&] { return parseParamValue(p, values.emplace_back(), elementType); });
}

template <typename SequenceType>
void printSequenceElements(AsmPrinter &p, ArrayRef<TypedAttr> values,
                           SequenceType type) {
  llvm::interleaveComma(values, p,
                        [&](TypedAttr value) { printParamValue(p, value); });
}

ParseResult parsePrettyType(AsmParser &p, TypedAttr &typeExpr);
void printPrettyType(AsmPrinter &p, TypedAttr typeExpr);

//===----------------------------------------------------------------------===//
// Logic shared between funcs, generators, and generator interfaces
//===----------------------------------------------------------------------===//

enum class GeneratorOrFuncKind { func, generator };

/// Parse the MLIR syntax for a kgen.generator, kgen.func and related
/// operators.
ParseResult parseGeneratorOrFunc(OpAsmParser &parser, OperationState &result,
                                 GeneratorOrFuncKind opKind);
void printGeneratorOrFunc(OpAsmPrinter &p, FuncInterface op);

/// Check that the specified declaration signatures match, checking the
/// parameter and value type information.
LogicalResult verifyDeclSignaturesMatch(StringRef originatorName,
                                        SignatureType originatorSignature,
                                        Location originatorLoc,
                                        StringRef interfaceName,
                                        SignatureType targetSignature,
                                        Location targetLoc);

/// Check that the parameter declarations match.
LogicalResult
verifyParamDeclsMatch(StringRef paramKind, StringRef originatorName,
                      ArrayRef<ParamDeclAttr> originatorParamDecls,
                      Location originatorLoc, StringRef targetName,
                      ArrayRef<ParamDeclAttr> targetParamDecls,
                      Location targetLoc);

/// Check that the parameter bindings match the declarations.
LogicalResult
verifyParamDeclsMatch(StringRef paramKind, StringRef originatorName,
                      ArrayRef<ParamBindAttr> binds, Location originatorLoc,
                      StringRef targetName, ArrayRef<ParamDeclAttr> decls,
                      Location targetLoc);

/// Check that the op has exactly one block in its region, or it's been cached.
LogicalResult verifyOneBlockOrCached(Operation *op);

/// Check the parameter result types.
LogicalResult checkResultParameterTypes(Operation *op,
                                        ArrayRef<TypedAttr> resultParams,
                                        DeclInterface decl);

/// Check the value and parameter result types.
LogicalResult checkResultArgumentTypes(Operation *op,
                                       ArrayRef<TypedAttr> resultParams,
                                       FuncInterface func);

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENUTILS_H
