//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_PARSEDARGUMENT_H
#define KGEN_MOJOPARSER_PARSEDARGUMENT_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "KGEN/MojoParser/Lexer.h"

namespace M::KGEN::LIT {
class ASTDecl;
class ASTType;
class ExprEmitter;
class ExprNode;
class ParserBase;

//===----------------------------------------------------------------------===//
// Argument and Parameter List Parsing
//===----------------------------------------------------------------------===//

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

/// Parsing support for a function argument and input parameter:
///
/// argument_list      ::= argument ("," argument)*
/// argument           ::= "/" | "*"
/// argument           ::= [argument_convention] [argument_variadic] identifier
///                        [argument_type] ["=" expression]
/// argument_convention ::= "owned" | "borrowed" | "inout"
/// argument_variadic  ::= "*" | "**"
/// argument_type      ::= ":" star_expression
struct ParsedArgument {
  SMLoc loc;
  LexerCursor cursor;
  // Specify argument passing convention, e.g. owned/byref etc.
  enum {
    kConventionUnspec = 0,         // Nothing specified
    kConventionInOut = 1,          // inout x
    kConventionOwned = 2,          // owned x
    kConventionBorrowed = 3,       // borrowed x
    kConventionInOutResult = 4,    // No syntax: result slot
    kConventionInitSelfResult = 5, // No syntax: __init__(inout self) argument
  } convention = kConventionUnspec;

  // After type checking, this will hold the KGEN convention to use.
  ValueInputConvention kgenConvention = ValueInputConvention(128);

  VarArgKind vararg = VarArgKind::None;
  StringAttr name;
  const ExprNode *typeExpr = nullptr;
  ExprNode *initExpr = nullptr;

  /// This gets set to true when there is a /diagnosed/ error that should
  /// prevent subsequent references to this argument.
  bool isErroneous = false;

  /// This specifies the handling of keyword arguments in a list.
  enum class KWArgHandling {
    kPositionalOnly,      //< before a standalone '/'
    kPositionalOrKeyword, //< before a standalone '*'
    kKeywordOnly          //< after a standalone '*'
  } kwArgHandling = KWArgHandling::kPositionalOrKeyword;

  enum class KWArgMarkerInfo {
    kNotMarker, //< This is a normal argument.
    kSlash,     //< This argument is a standalone '/' marker.
    kStar,      //< This argument is a standalone '*' marker.
  };

  enum class ArgListKind {
    kParamList,         //< parameter list like `[x: Int, y: Int]`
    kArgList,           //< argument list like `(x: Int, y: Int)`
    kFnTypeArgList,     //< fn type, like `fn (Int, y: Float)`
    kFnTypeParamList,   //< fn type, like `fn [Int, y: Float](x: Int)`
    kBareLambdaArgList, //< argument list like `lambda x, y: x+y`
  };

  ParseResult parse(ParserBase &p, KWArgMarkerInfo &markerInfo,
                    ArgListKind kind);

  /// This method handles the function argument list for a Python function.
  /// Python has some pretty interesting rules where standalone '*' and '/'
  /// markers (when used in place of an argument) actually change the
  /// interpretation of other argument definitions by specifying how they behave
  /// w.r.t. keyword arguments.  We resolve these here so the client doesn't
  /// have to deal with them.
  ///
  /// This classification logic is described here:
  ///   https://peps.python.org/pep-0570/#how-to-teach-this
  ///
  static ParseResult parseAndResolvePresentArgumentList(
      ParserBase &p, SmallVectorImpl<ParsedArgument> &args, ArgListKind kind);

  /// Parse an argument list, including the parentheses around them.  The
  /// argument list is allowed to be empty.  If `fnEffects` is non-null, then
  /// this parses 'raises' and other effects.
  static ParseResult parseAndResolveParenthesizedArgumentList(
      ParserBase &p, SmallVectorImpl<ParsedArgument> &args, ArgListKind kind,
      FnEffects &fnEffects);

  /// Process parsed parameter arguments into input parameters by determining
  /// the correct parameter types, conventions, and default parameter values.
  /// The unmangled parameter names are also collected.
  static void processParameterInputArgs(
      ExprEmitter &emitter, ASTDecl &declScope, ArrayRef<ParsedArgument> args,
      SmallVectorImpl<ParamDeclAttr> &params,
      SmallVectorImpl<StringAttr> &names,
      SmallVectorImpl<PassingKind> &passingKinds,
      SmallVectorImpl<TypedAttr> &defaultPosParams, bool &paramVarArg);

  /// Emit the argument types, default values, and result type and determine
  /// the argument conventions.
  static ASTType emitFunctionArgumentsAndResults(
      function_ref<ParseResult()> reportError, ExprEmitter &typeEmitter,
      SmallVectorImpl<StringAttr> &inputParamNames,
      SmallVectorImpl<PassingKind> &inputParamPassingKinds,
      SmallVectorImpl<ParamDeclAttr> &inputParamDecls,
      const ExprNode *resultTypeExpr, FnEffects &effects,
      SmallVectorImpl<ParsedArgument> &args, SmallVectorImpl<Type> &argTypes,
      SmallVectorImpl<TypedAttr> &defaultPosArgs, bool isDef, SMLoc resultLoc,
      ASTDecl *fnDecl = nullptr,
      SpecialFunctionInfo fnInfo = SpecialFunctionInfo(),
      function_ref<void()> processSignature = [] {});

  /// Map KWArgHandling to the PassingKind enum of the LIT dialect.
  static PassingKind mapToPassingKind(KWArgHandling handling);
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARSEDARGUMENT_H
