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
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/OpImplementation.h"

namespace M::KGEN {
class DeclInterface;
class FuncInterface;

/// This struct represents the data for an exported symbol.
struct ExportedSymbol {
  ExportedSymbol(ExportKind kind, bool isData = false)
      : kind(kind), isData(isData) {}

  /// The export kind of the symbol.
  ExportKind kind;
  /// True if the symbol is a global variable.
  bool isData;
};

/// This type is a bit of a mouthful, add a useful alias for it.
using ExportMap = llvm::MapVector<StringAttr, ExportedSymbol>;

/// Given a module operation, return its exported symbols and aliases.
ExportMap getExportedSymbols(ModuleOp module);

/// Return the string form for an attribute value that is printed in a <>
/// context in the .mlir file. In diagnostics contexts, MLIR and KGEN keywords
/// are not escaped with *"...".
std::string getParamAsString(Attribute value, bool forDiag = false);

/// Print the value as colon type parameter value into a string.
StringAttr getParamTypeAsString(TypedAttr value);

/// Print the type as a KGEN type.
StringAttr getTypeAsString(Type type);

/// Parse a parameter of type kgen.string.
ParseResult parseStringParam(AsmParser &p, TypedAttr &value);

/// Print a parameter of type kgen.string.
void printStringParam(AsmPrinter &p, Operation *, Attribute value);

/// Parse a type in a KGEN context, handling sugar like "dtype" for
/// "!kgen.dtype" etc.
ParseResult parseKGENType(AsmParser &parser, Type &type);
OptionalParseResult parseOptionalKGENType(AsmParser &parser, Type &type);

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

/// Returns whether the given type could be the type of a KGEN type expression.
bool isTypeExprType(Type type);

/// Returns whether the given attribute is a KGEN type expression.
bool isTypeExpr(TypedAttr attr);

/// Gets the common Modular environment attribute (also known as `-D` defines)
/// for the given module. This includes things like `MODULAR_PARANOID`,
/// `BUILD_TYPE`, LLCL profiling level, etc.
EnvAttr getModularEnvAttr(MLIRContext *ctx);

/// Extends the module EnvAttr with common Modular environment attribute (also
/// known as `-D` defines) for the given module. This includes things like
/// `MODULAR_PARANOID`, `BUILD_TYPE`, LLCL profiling level, etc. Note that the
/// existing EnvAttr module values take precedence here.
void extendWithModularEnvAttr(ModuleOp moduleOp);

//===----------------------------------------------------------------------===//
// Parameter Printing and Parsing
//===----------------------------------------------------------------------===//

/// Print a parameter name correctly, using a double quoted syntax if it
/// conflicts with an MLIR or KGEN keyword, or a bareword otherwise. When
/// printing a parameter name in a reference, the name must be escaped to
/// prevent collision with other parameter values, particularly types.
void printParamName(AsmPrinter &p, StringAttr name, bool isRef = false);
/// Parse a parameter name as either a keyword or double quoted string.
ParseResult parseParamName(AsmParser &p, StringAttr &name);

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the printed syntax easier to grok. In a
/// context where printing for diagnostics, we do not use the double quoted
/// syntax to escape MLIR and KGEN keywords.
void printParamValue(AsmPrinter &p, TypedAttr value, Type type = {},
                     bool forDiag = false);
void printParamValue(AsmPrinter &p, Operation *op, TypedAttr value,
                     Type type = {});

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the parsed syntax easier to grok.
ParseResult parseParamValue(AsmParser &p, TypedAttr &value, Type type);

/// Parse a parameter declaration of the form `name = value`.
ParseResult parseParamDeclaration(OpAsmParser &p, ParamDeclAttr &paramDecl,
                                  TypedAttr &value);

/// Print a parameter declaration of the form `name = value`.
void printParamDeclaration(OpAsmPrinter &p, ParamDeclAttr paramDecl,
                           TypedAttr value);

/// Parse ":type 42" or "42" and default to index type.
ParseResult parseParamValueDefaultingToIndex(AsmParser &p, TypedAttr &value);

/// Print a parameter value that is known to have `dtype` type.
void printDTypeParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `dtype` type.
ParseResult parseDTypeParamValue(AsmParser &p, TypedAttr &value);

/// Print a type parameter value. Default to `TypeType`, but allow an
/// optional type for the type value.
void printTypeParamValue(AsmPrinter &p, TypedAttr value);
/// Parse a type parameter value. Prints the type of the type value if it is not
/// an `TypeType`.
ParseResult parseTypeParamValue(AsmParser &p, TypedAttr &value);

/// Parse or print a parametric type expression and convert it to a type.
ParseResult parseParamType(AsmParser &p, Type &type);
void printParamType(AsmPrinter &p, Type type);
ParseResult parseParamTypes(AsmParser &p, SmallVectorImpl<Type> &types);
void printParamTypes(AsmPrinter &p, ArrayRef<Type> types);

/// Print an array of parameter type values.
void printTypeParamValues(AsmPrinter &p, ArrayRef<TypedAttr> values);
/// Parse an array of parameter type values.
ParseResult parseTypeParamValues(AsmParser &p, SmallVector<TypedAttr> &values);

/// Print the body of a type-value (without any surrounding brackets). Caller
/// specifies how types are printed.
void printTypeValueBody(
    AsmPrinter &p, TypeConstantAttr type,
    llvm::function_ref<void(AsmPrinter &, Type)> typePrinter);
/// Parse the body of a type-value (without any surrounding brackets). Caller
/// specifies how types are parsed.
/// If the caller knows the type has identical type-value representation, it
/// can set the additional flag to abort after the first type is parsed.
OptionalParseResult parseTypeValueBody(
    AsmParser &p, TypedAttr &value, Type type,
    llvm::function_ref<OptionalParseResult(AsmParser &, Type &)> typeParser,
    bool knownIdenticalRepresentation = false);

/// Pretty print a type-value:
/// If the type-value has identical type/value representation, just print the
/// type-value Type itself. Otherwise print the entire type-value surrounded by
/// square brackets.
LogicalResult
printSugaredTypeValue(AsmPrinter &p, TypedAttr value,
                      llvm::function_ref<void(AsmPrinter &, Type)> typePrinter);
/// Parse a pretty-printed type-value.
OptionalParseResult parseSugaredTypeValue(
    AsmParser &p, TypedAttr &value, Type type,
    llvm::function_ref<OptionalParseResult(AsmParser &, Type &)> typeParser);

/// Print a parameter value that is known to have `index` type.
void printIndexParamValue(AsmPrinter &p, Operation *op, Attribute value);
void printIndexParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `index` type.
ParseResult parseIndexParamValue(AsmParser &p, TypedAttr &value);

/// Print a parameter value that is known to have `i1` type.
void printI1ParamValue(AsmPrinter &p, Operation *op, Attribute value);
void printI1ParamValue(AsmPrinter &p, Attribute value);
/// Parse a parameter value that is known to have `i1` type.
ParseResult parseI1ParamValue(AsmParser &p, TypedAttr &value);

/// Parse a index-or-colon-type and then a parameter value of that type.
ParseResult parseColonTypeParamValue(AsmParser &p, TypedAttr &value);
void printColonTypeParamValue(AsmPrinter &p, TypedAttr value);
inline void printColonTypeParamValue(AsmPrinter &p, Operation *,
                                     TypedAttr value) {
  printColonTypeParamValue(p, value);
}

/// Parse and print a ParamDeclAttr which has syntactic form `name (: type)?`.
ParseResult parseParamDecl(AsmParser &p, ParamDeclAttr &result);
void printParamDecl(AsmPrinter &p, ParamDeclAttr decl);
inline void printParamDecl(AsmPrinter &p, Operation *, ParamDeclAttr decl) {
  printParamDecl(p, decl);
}

/// Type of hooks that customize parameter declaration printing.
using ParamDeclPrintHookTy = function_ref<void(ParamDeclAttr decl)>;

/// Type of hooks that customize parameter declaration parsing.
using ParamDeclParseHookTy =
    function_ref<ParseResult(SmallVectorImpl<ParamDeclAttr> &)>;

/// Print a ParamDeclArrayAttr as a canonical list of comma separated
/// information. If the element printing hook is provided, it is called by the
/// given parser for each element in the list, and is responsible for printing
/// the decl.
void printParamDecls(AsmPrinter &p, ArrayRef<ParamDeclAttr> decls,
                     ParamDeclPrintHookTy printElt = {});

/// Parse a ParamDeclArrayAttr as a canonical list of comma separated
/// information. If the element parsing hook is provided, it is called by the
/// given parser for each element in the list, and is responsible for parsing
/// the decl and placing it in the provided array.
ParseResult parseParamDecls(AsmParser &p, ParamDeclArrayAttr &result,
                            ParamDeclParseHookTy parseElt = {});

/// Parse and print a parameter specification on a generator or region type. The
/// parameter spec includes input parameter declarations and types and
/// optionally result parameter declarations and types. If the input element
/// parsing hook is provided, it is called by the given parser for each element
/// of the inputs, and is responsible for parsing the decl and placing it in the
/// provided array.
ParseResult parseOptionalParameterSpec(AsmParser &parser,
                                       ParamDeclArrayAttr &inputParamDecls,
                                       ParamDeclArrayAttr &resultParamDecls,
                                       ParamDeclParseHookTy parseInputElt = {});

/// Print a parameter specification on a generator or region type. The parameter
/// spec includes input parameter declarations and types and optionally result
/// parameter declarations and types. If the input element printing hook is
/// provided, it is called by the given parser for each element of the inputs,
/// and is responsible for printing the decl.
void printOptionalParameterSpec(AsmPrinter &p,
                                ArrayRef<ParamDeclAttr> inputParamDecls,
                                ArrayRef<ParamDeclAttr> resultParams = {},
                                ParamDeclPrintHookTy printInputElt = {});

/// Parse an optional argument convention, or use the given default.
ParseResult
parseArgConvention(AsmParser &p, ArgConvention &convention,
                   ArgConvention defaultConvention = ArgConvention::None);

/// Print an argument convention if not the given default.
void printArgConvention(AsmPrinter &p, ArgConvention convention,
                        ArgConvention defaultConvention = ArgConvention::None);

/// Print the parameter type signature if there are any input or result types.
/// If the input type printing hook is provided, it is called by the given
/// parser for each element of the inputs, and is responsible for printing the
/// type.
void printOptionalParamSignature(AsmPrinter &p, ArrayRef<Type> inputParamTypes,
                                 ArrayRef<Type> resultParamTypes,
                                 function_ref<void(Type)> printInputTy = {});

/// Parse a parameter signature (input/result types) if present. If the input
/// type parsing hook is provided, it is called by the given parser for each
/// element of the inputs, and is responsible for parsing the type and placing
/// it in the provided array.
ParseResult parseOptionalParamSignature(
    AsmParser &p, SmallVectorImpl<Type> &inputParamTypes,
    SmallVectorImpl<Type> &resultParamTypes,
    function_ref<ParseResult(SmallVectorImpl<Type> &)> parseInputTy = {});

ParseResult parseSignature(AsmParser &p, TypeAttr &signature);
ParseResult parseSignature(AsmParser &p, Type &signature);
ParseResult parseSignatureValues(
    AsmParser &p, function_ref<ParseResult(SmallVectorImpl<Type> &)> parseArg,
    FunctionType &values, FnEffects &effects, bool optionalResultList);
void printSignature(AsmPrinter &p, Type signatureType);
inline void printSignature(AsmPrinter &p, Operation *op, Type signatureType) {
  printSignature(p, signatureType);
}
void printSignature(AsmPrinter &p, Operation *op, TypeAttr signature);
void printSignatureValues(AsmPrinter &p, FunctionType functionType,
                          SignatureType signature);
void printSignatureValues(AsmPrinter &p, function_ref<void(unsigned)> printElt,
                          FunctionType functionType, SignatureType signature,
                          bool optionalResultList);

/// Parse a plain (i.e. non-lit) signature.
ParseResult parseKGENSignature(AsmParser &p, FunctionType &functionType,
                               SignatureType &signature);

/// Parse a function signature with optional metadata. In the assembly format,
/// the SSA value names are optional in the argument list. If they are present,
/// they are populated in `args`. The `parseNames` flag control whether the
/// signature should include the argument names.
ParseResult parseFunctionSignature(OpAsmParser &p,
                                   SmallVectorImpl<OpAsmParser::Argument> &args,
                                   ParamDeclArrayAttr &inputParams,
                                   ParamDeclArrayAttr &resultParams,
                                   FunctionType &functionType,
                                   SignatureType &signature);
/// Print a function signature with optional metadata. If `region` is non-null,
/// then the SSA value names of the region arguments are printed.
void printFunctionSignature(OpAsmPrinter &p, Region *region,
                            ArrayRef<ParamDeclAttr> inputParams,
                            ArrayRef<ParamDeclAttr> resultParams,
                            FunctionType functionType, SignatureType signature);

/// Parse the always_inline related keywords if present.
ParseResult parseOptionalInline(OpAsmParser &parser, InlineLevelAttr &attr);
void printOptionalInline(AsmPrinter &p, InlineLevel level);

/// Parse and print a decorator list if present.
ParseResult parseOptionalDecorators(AsmParser &p, DecoratorsAttr &decorators);
void printOptionalDecorators(OpAsmPrinter &p, Operation *op,
                             ArrayRef<TypedAttr> decorators);

/// Parse and print a list of parameter values.
ParseResult parseParameterValues(AsmParser &p, ParameterExprArrayAttr &values);
ParseResult parseParameterValues(AsmParser &p,
                                 SmallVectorImpl<TypedAttr> &values);
void printParameterValues(OpAsmPrinter &p, Operation *op,
                          ParameterExprArrayAttr values);
void printParameterValues(AsmPrinter &p, ArrayRef<TypedAttr> values);

/// Parse and print a parametric callee and result parameter declarations.
ParseResult parseParametricCallee(OpAsmParser &p, TypedAttr &callee);
void printParametricCallee(OpAsmPrinter &p, Operation *, TypedAttr callee);

/// Parse and print a comma separated sequence of elements.
template <typename SequenceType>
ParseResult parseSequenceElements(AsmParser &p, SmallVector<TypedAttr> &values,
                                  SequenceType type) {
  return p.parseCommaSeparatedList([&] {
    return parseParamValue(p, values.emplace_back(), type.getElementType());
  });
}

template <typename SequenceType>
void printSequenceElements(AsmPrinter &p, ArrayRef<TypedAttr> values,
                           SequenceType type) {
  llvm::interleaveComma(values, p,
                        [&](TypedAttr value) { printParamValue(p, value); });
}

//===----------------------------------------------------------------------===//
// Logic shared between funcs, generators, and generator interfaces
//===----------------------------------------------------------------------===//

enum class GeneratorOrFuncKind { func, generator };

/// Parse and print an export kind.
ParseResult parseSymbolExport(AsmParser &p, ExportKindAttr &exportKind);
void printSymbolExport(AsmPrinter &p, Operation *op, ExportKindAttr exportKind);

/// Check that the specified declaration signatures match, checking the
/// parameter and value type information.
LogicalResult verifyDeclSignaturesMatch(StringRef originatorName,
                                        SignatureType originatorSignature,
                                        Location originatorLoc,
                                        StringRef interfaceName,
                                        SignatureType targetSignature,
                                        Location targetLoc);

/// Check that the parameter bindings match the declarations.
LogicalResult
verifyParamDeclsMatch(StringRef paramKind, StringRef originatorName,
                      ArrayRef<TypedAttr> paramValues, Location originatorLoc,
                      StringRef targetName, ArrayRef<ParamDeclAttr> decls,
                      Location targetLoc);

/// Check the parameter result types.
LogicalResult checkResultParameterTypes(Operation *op,
                                        ArrayRef<TypedAttr> resultParams,
                                        DeclInterface decl);

/// Check the value and parameter result types.
LogicalResult checkResultArgumentTypes(Operation *op,
                                       ArrayRef<TypedAttr> resultParams,
                                       FuncInterface func);

/// Whether the decorator's name is (starts with) the specific annotation.
bool hasDecorator(ArrayRef<TypedAttr> decorators, StringRef annotation);

/// Whether the generator operation contains any decorator with any of the given
/// annotations.
bool hasAnyDecorator(ArrayRef<TypedAttr> decorators,
                     ArrayRef<StringLiteral> annotations);

ParseResult parseRegionWithArgs(OpAsmParser &p, Region &region);
void printRegionWithArgs(OpAsmPrinter &p, Operation *op, Region &region);

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENUTILS_H
