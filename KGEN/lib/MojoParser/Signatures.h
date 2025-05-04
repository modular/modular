//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains logic for parsing and type checking and IR building of
// signatures for structs, functions, and function types.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_SIGNATURES_H
#define KGEN_MOJOPARSER_SIGNATURES_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/Lexer.h"

namespace mlir {
class raw_indented_ostream;
} // namespace mlir

namespace M::KGEN::LIT {
class ASTDecl;
class ASTType;
class ExprEmitter;
class ExprNode;
class FnTypeGeneratorType;
class ParserBase;

//===----------------------------------------------------------------------===//
// Argument and Parameter List Parsing
//===----------------------------------------------------------------------===//

/// This specifies the handling of keyword arguments in a list.
enum class KWArgHandling {
  kInferred,            //< before a standalone '//'
  kPositionalOnly,      //< before a standalone '/'
  kPositionalOrKeyword, //< before a standalone '*'
  kKeywordOnly          //< after a standalone '*'
};

enum class KWArgMarkerInfo {
  kNotMarker,  //< This is a normal argument.
  kSlashSlash, //< This argument is a standalone '//' marker.
  kSlash,      //< This argument is a standalone '/' marker.
  kStar,       //< This argument is a standalone '*' marker.
};

enum class ArgListKind {
  kParamList,         //< parameter list like `[x: Int, y: Int]`
  kArgList,           //< argument list like `(x: Int, y: Int)`
  kFnTypeArgList,     //< fn type, like `fn (Int, y: Float)`
  kFnTypeParamList,   //< fn type, like `fn [Int, y: Float](x: Int)`
  kBareLambdaArgList, //< argument list like `lambda x, y: x+y`
};

/// Specify variadic argument kind, e.g. `*x` or `**x`.
enum VarArgKind {
  /// Not a variadic argument, e.g. `x` or `x: Int`.
  None,
  /// A homogeneously typed variadic argument, e.g. `*x` or `*x: Int`.
  VarArg,
  /// A heterogeneously typed variadic argument, e.g. `*x: *Ts`.
  PackVarArg,
  /// A variadic keywords argument, e.g. `**x`.
  KWVarArg
};

/// Parsing support for a function argument and parameter:
///
/// argument_list      ::= argument ("," argument)*
/// argument           ::= "/" | "*"
/// argument           ::= [argument_convention] [argument_variadic] identifier
///                        [argument_type] ["=" expression]
/// argument_convention ::= "owned" | "read" | "mut"
/// argument_variadic  ::= "*" | "**"
/// argument_type      ::= ":" star_expression
///
/// Note that this  type is stored in a bump pointer allocated ExprNode, so
/// it cannot have a destructor!
struct ParsedArgument {
  SMLoc loc;
  LexerCursor cursor;
  // Specify argument passing convention, e.g. owned/mut etc.
  enum PAArgConvention {
    kConventionUnspec = 0,      // Nothing specified
    kConventionMut = 1,         // mut x
    kConventionOwned = 2,       // owned x
    kConventionRead = 3,        // read x
    kConventionRef = 4,         // ref [origin, addrspace] x
    kConventionOut = 5,         // __init__(out self) argument
    kConventionByRefResult = 6, // No syntax: result slot
  } convention = kConventionUnspec;

  // After type checking, this will hold the KGEN convention to use.
  ArgConvention kgenConvention = ArgConvention(128);

  // For variadics and packs, this is the declared argument convention, even
  // those the variadic type is passed another way.
  ArgConvention kgenVariadicConvention = ArgConvention(128);

  VarArgKind vararg = VarArgKind::None;
  StringAttr name;
  ExprNode *typeExpr = nullptr;
  ExprNode *initExpr = nullptr;
  // If this is a ref convention, this specifies the origin expression.
  ExprNode *refOriginExpr = nullptr;

  /// This gets set to true when there is a /diagnosed/ error that should
  /// prevent subsequent references to this argument.
  mutable bool isErroneous = false;

  KWArgHandling kwArgHandling = KWArgHandling::kPositionalOrKeyword;

  ParseResult parse(ParserBase &p, KWArgMarkerInfo &markerInfo,
                    ArgListKind kind);

  /// Map KWArgHandling to the PassingKind enum of the LIT dialect.
  PassingKind getKWArgHandlingAsPassingKind() const;

  /// Print the argument for debugging.
  void print(mlir::raw_indented_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

//===----------------------------------------------------------------------===//
// ParsedParamList
//===----------------------------------------------------------------------===//

/// This is all the state built up when parsing the parameter signature for a
/// parameterized declaration, (e.g. a function or struct).
class ParsedParamList {
public:
  /// The full ParsedArgument for each parameter.
  SmallVector<ParsedArgument> params;

  /// Parse a parameter signature if present.
  ///
  /// param_signature    ::= "[" param_list ("->" param_result_types)? "]"
  /// param_list   ::= argument_list | "(" ")"
  /// param_result_types ::= expression ("," expression)*
  ParseResult parseParametersIfPresent(ParserBase &p, ArgListKind kind);
};

/// This contains the result state from type checking a parameter signature.
class TypeCheckedParamList {
public:
  /// This is the declaration that we do name lookup against.
  ASTDecl &declScope;
  SharedState &shared;

  /// Type check each of the parameters from 'parsedParams' into their
  /// decomposed representation.
  TypeCheckedParamList(ArrayRef<ParsedArgument> parsedParams,
                       ASTDecl &declScope);

  /// Get an PogListAttr for this parameter list.
  PogListAttr getParamListAttr();

  // These are the results of type checking 'params' in typeCheck.
  /// One ParamDeclAttr for each parameter being declared.
  SmallVector<ParamDeclAttr> paramDeclAttrs;
  SmallVector<StringAttr> names;
  SmallVector<PassingKind> passingKinds;
  SmallVector<VariadicKind> variadicKinds;

  /// Default values for positional and positionalOrKeyword params.
  SmallVector<TypedAttr> defaultPosParams;
  /// Default values for keyword-only params.
  SmallVector<TypedAttr> defaultKwOnlyParams;
};

//===----------------------------------------------------------------------===//
// ParsedArgumentList
//===----------------------------------------------------------------------===//

/// This is all the state built up when parsing a function signature.
class ParsedArgumentList {
public:
  /// Any arguments specified.
  SmallVector<ParsedArgument> parsedArgs;
  /// The result specifier if present.
  ParsedArgument resultArg;
  FnEffects effects;

  /// Parse an argument list, including the parentheses around them. This also
  /// parses 'raises' and other effects.
  ParseResult parseArgumentListAndEffects(ParserBase &p, ArgListKind kind);

  /// Parse the result specifier starting with a `->` if present.
  void parseResultIfPresent(ParserBase &p,
                            std::optional<size_t> stmtIndent = std::nullopt);
};

/// This contains the result state from type checking a parameter signature.
class TypeCheckedFnSignature {
public:
  /// Emit the argument types, default values, and result type and determine
  /// the argument conventions.
  ///
  /// 'fnDecl' will be null when this is a function type, which doesn't have a
  /// declaration.
  TypeCheckedFnSignature(TypeCheckedParamList &paramList,
                         ParsedArgumentList &argList,
                         const ExprNode *originExpr, ASTDecl *fnDecl,
                         SpecialFunctionInfo &fnInfo);
  TypeCheckedParamList &paramList;
  ParsedArgumentList &argList;

  /// For methods, this is the default type of Self.  For global functions this
  /// is null.
  ASTType selfType;

  // This is the type checked declared argument type, e.g. "String" or "Int".
  SmallVector<Type> argTypes;
  /// Default values for positional and positionalOrKeyword args.
  SmallVector<TypedAttr> defaultPosArgs;
  /// Default values for keyword-only arguments.
  SmallVector<TypedAttr> defaultKwOnlyArgs;
  ASTType resultType;

  /// This is the type checked argument types with argument conventions and
  /// origins applied, e.g. "!lit.ref<String>" or "!kgen.variadic<Int>"
  SmallVector<Type> fullArgTypes;
  SmallVector<ParamDeclAttr> implicitOriginDecls;

  /// This is the result type + variant for throwing functions.  This is what
  /// finally gets treated as the ABI for the function.
  ASTType fullResultType;

  /// This is an optional origin set parameter, representing the origins of
  /// the function captures.
  TypedAttr captureOrigins;
  /// Whether `@__unsafe_disable_nested_origin_exclusivity` was specified:
  /// nested origins are not considered in exclusivity checking.
  /// TODO: Generalize this to mutation sets.
  bool isNestedOriginExclusivityCheckingDisabled = false;

  /// This performs any special checks over the declaration based on its name
  /// and whether it is a method.  This happens after decorator processing
  /// because that is how defs work in Python.
  ///
  /// If this function detects a problem, it marks the decl as erroneous and
  /// resets the SpecialFunctionInfo.
  void verifyFunctionNameBinding(ASTDecl &decl, StringAttr name,
                                 SpecialFunctionInfo &fnInfo) const;

  /// Return a FunctionType with the specified argTypes and resultType.
  FunctionType getFunctionType() const;

  /// Form a LIT signature packaging up all the stuff we need to know about this
  /// type checked function.
  FnTypeGeneratorType getFnTypeGeneratorType() const;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_SIGNATURES_H
