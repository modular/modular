//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains logic for parsing and type checking and IR building of function
// signatures.  This is used both for fn/def declarations, but also for function
// type syntax.
//
//===----------------------------------------------------------------------===//

#include "Signatures.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ParserBase.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Argument and Parameter List Parsing
//===----------------------------------------------------------------------===//

ParseResult ParsedArgument::parse(ParserBase &p, KWArgMarkerInfo &markerInfo,
                                  ArgListKind kind) {
  loc = p.getToken().getLoc();
  cursor = p.getLexer().getCursor();

  // Any owned/borrowed/inout keyword sets convention.
  if (p.consumeIf(Token::kw_owned))
    convention = kConventionOwned;
  else if (p.consumeIf(Token::kw_borrowed))
    convention = kConventionBorrowed;
  else if (p.consumeIf(Token::kw_inout))
    convention = kConventionInOut;
  while (p.getToken().isAny(Token::kw_owned, Token::kw_borrowed,
                            Token::kw_inout)) {
    p.emitTokenError("argument already has a convention specified");
    p.consumeToken();
  }

  markerInfo = KWArgMarkerInfo::kNotMarker;

  // The first token of an argument may be a standalone '*' or '/' marker, and
  // the '*' may also be part of a varargs specification.  Check for these
  // first.
  if (p.consumeIf(Token::slash)) {
    markerInfo = KWArgMarkerInfo::kSlash;
    return success();
  }
  if (p.consumeIf(Token::star)) {
    if (p.getToken().isAny(Token::comma, Token::r_paren, Token::r_square)) {
      markerInfo = KWArgMarkerInfo::kStar;
      return success();
    }
    vararg = VarArgKind::VarArg;
  } else if (p.consumeIf(Token::star_star)) {
    vararg = VarArgKind::KWVarArg;
    kwArgHandling = KWArgHandling::kKeywordOnly;
  }

  // When parsing a function type, the name is optional.
  if (kind == ArgListKind::kFnTypeArgList ||
      kind == ArgListKind::kFnTypeParamList) {
    StringAttr maybeArgName;
    SMLoc nextLocation;
    if (succeeded(p.parseOptionalIdentifier(maybeArgName, Token::colon,
                                            &nextLocation))) {
      name = maybeArgName;
      loc = nextLocation;
    }
  } else {
    if (p.parseIdentifier(name, "expected parameter name", &loc)) {
      // TODO: Scan ahead for better recovery.
      return failure();
    }
  }

  // Parse an optional type annotation: `":" ["*"] expression`. Omit the colon
  // if a name was not specified.  Bare lambda arg lists do not allow types.
  if (kind != ArgListKind::kBareLambdaArgList) {
    if (!name || p.consumeIf(Token::colon)) {
      SMLoc starLoc = p.getToken().getLoc();
      if (p.getToken().getKind() == Token::star) {
        if (vararg != VarArgKind::VarArg) {
          InflightDiag diag = p.emitError(
              starLoc, "only variadic arguments' types can be unpacked");
          if (name) {
            diag.attachNote(loc)
                << "'" << name.getValue() << "' is not a variadic argument";
          }
        }
        vararg = VarArgKind::PackVarArg;
      }
      ExprNode *typeExprNode;
      if (p.parseStarredItem(typeExprNode))
        return failure();
      typeExpr = typeExprNode;
    }
  }

  // Set the name to empty string if it wasn't specified.
  if (!name)
    name = StringAttr::get(p.getContext());

  // Parse an optional default argument value: `"=" expression`.
  SMLoc equalLoc;
  if (p.consumeIf(Token::equal, &equalLoc)) {
    if (p.parseExpression(initExpr))
      return failure();

    if (convention == kConventionInOut) {
      p.emitError(equalLoc, "inout arguments may not have defaults")
          << initExpr->getRange();
      initExpr = nullptr;
    }

    // Default args and varargs don't mix.
    if (vararg != VarArgKind::None) {
      p.emitError(equalLoc, "variadic arguments may not have defaults")
          << initExpr->getRange();
      initExpr = nullptr;
    }
  }
  return success();
}

ParseResult ParsedArgument::parseAndResolvePresentArgumentList(
    ParserBase &p, SmallVectorImpl<ParsedArgument> &args, ArgListKind kind) {
  // Figure out where to stop scanning.
  SmallVector<Token::Kind, 2> stopTokens;
  switch (kind) {
  case ArgListKind::kParamList:
  case ArgListKind::kFnTypeParamList:
    stopTokens.append({Token::r_square, Token::minus_greater});
    break;
  case ArgListKind::kFnTypeArgList:
  case ArgListKind::kArgList:
    stopTokens.push_back(Token::r_paren);
    break;
  case ArgListKind::kBareLambdaArgList:
    stopTokens.push_back(Token::colon);
    break;
  }

  // As we parse all of the arguments and the keyword arguments and markers, we
  // resolve the markers and check the invariants.  Python's parameter grammar
  // embeds checking for `/` and `*` into it, but we do this ad-hoc for
  // simplicity, according to the following rules:
  //
  //   1) Only one '/' and '*' marker may exist in the parameter list.
  //   2) They are specified in that order.
  //   3) `/` cannot be first, and '*' cannot be last in the list.
  //
  // See this for more information:
  // https://peps.python.org/pep-0570/#how-to-teach-this
  bool hasSlashMarker = false, hasStarMarker = false;
  auto defaultKWArgHandling = KWArgHandling::kPositionalOrKeyword;

  // This is invoked when we see a '/' marker.
  StringRef argOrParam =
      kind == ArgListKind::kParamList || kind == ArgListKind::kFnTypeParamList
          ? "parameter"
          : "argument";
  auto handleSlashMarker = [&](SMLoc loc) {
    if (hasSlashMarker) {
      p.emitError(loc, "cannot have two '/' markers in the same ")
          << argOrParam << " list";
      return;
    }
    if (hasStarMarker) {
      p.emitError(loc, "cannot specify '/' marker after '*' marker in ")
          << argOrParam << " list";
      return;
    }
    if (args.empty()) {
      p.emitError(loc, "'/' marker cannot be used at the start of the ")
          << argOrParam << " list";
    }

    // Ok, process it by changing all arguments we've seen to be positional
    // only.  The remaining ones will stay kPositionalOrKeyword though.
    for (ParsedArgument &arg : args)
      arg.kwArgHandling = KWArgHandling::kPositionalOnly;
    hasSlashMarker = true;
  };

  // This is invoked when we see a '*' marker or '*arg' argument.
  auto handleStarMarker = [&](SMLoc loc, bool isMarker) {
    if (hasStarMarker) {
      p.emitError(loc, "cannot have two '*' markers in the same ")
          << argOrParam << " list";
    }

    // Diagnose '*' marker at end of argument list for completeness.
    if (p.getToken().isAny(stopTokens) && isMarker) {
      p.emitError(loc, "'*' marker is not allowed at end of ")
          << argOrParam << " list";
    }

    // From now on, any parsed arguments are keyword only.
    defaultKWArgHandling = KWArgHandling::kKeywordOnly;
    hasStarMarker = true;
  };

  // This parses either an argument or a keyword argument specifier.
  bool foundName = false;
  auto parseArgument = [&]() -> ParseResult {
    KWArgMarkerInfo marker = KWArgMarkerInfo::kNotMarker;
    ParsedArgument arg;
    arg.kwArgHandling = defaultKWArgHandling;
    if (arg.parse(p, marker, kind))
      return failure();

    // If this argument is just a marker, process it.
    if (marker == KWArgMarkerInfo::kSlash)
      return handleSlashMarker(arg.loc), success();
    if (marker == KWArgMarkerInfo::kStar)
      return handleStarMarker(arg.loc, /*isMarker=*/true), success();

    if (arg.name.empty()) {
      if (foundName) {
        return p.emitError(arg.loc, "unnamed ")
               << argOrParam << " cannot follow named " << argOrParam;
      }
      if (hasSlashMarker || hasStarMarker) {
        return p.emitError(arg.loc, "unnamed ")
               << argOrParam << " cannot follow '/' or '*'";
      }
    } else {
      foundName = true;
    }

    // Otherwise, if this is a varargs marker, handle it as a marker and an
    // argument.
    if (arg.vararg == VarArgKind::VarArg ||
        arg.vararg == VarArgKind::PackVarArg)
      handleStarMarker(arg.loc, /*isMarker=*/false);

    // If we have a **arg then it must be the last argument.
    if (arg.vararg == VarArgKind::KWVarArg && p.getToken().isNot(stopTokens)) {
      p.emitError(arg.loc, "'**' marker must be at end of ")
          << argOrParam << " list";
      arg.vararg = VarArgKind::None;
    }

    if (arg.vararg == VarArgKind::KWVarArg)
      p.emitError(arg.loc, "variadic keyword argument not supported yet");

    // Otherwise just remember the argument.
    args.push_back(arg);
    return success();
  };

  // Parse a list of arguments and keyword argument specifiers.  Each argument
  // will leave its `kwargHandling` default initialized.
  if (p.parseCommaSeparatedList(parseArgument, stopTokens))
    return failure();

  // We allow specifying signatures with only positional-only arguments if all
  // the argument names are omitted, i.e. `fn(Int, Int) -> Int` is the same as
  // `fn(Int, Int, /) -> Int`.
  bool allUnnamedPosOnly = !foundName && !hasSlashMarker && !hasStarMarker;
  for (ParsedArgument &arg : args) {
    if (!arg.name.empty() ||
        arg.kwArgHandling == KWArgHandling::kPositionalOnly || arg.vararg)
      continue;
    if (!allUnnamedPosOnly)
      return p.emitError(arg.loc, "unnamed ")
             << argOrParam << " must be positional-only";
    arg.kwArgHandling = KWArgHandling::kPositionalOnly;
  }

  // TODO(#21950): we currently don't allow keyword-only args after variadics.
  auto hasVarArg = [](const ParsedArgument &arg) { return arg.vararg; };
  if (!args.empty() &&
      args.back().kwArgHandling == KWArgHandling::kKeywordOnly &&
      llvm::any_of(args, hasVarArg)) {
    p.emitError(args.back().loc,
                "keyword-only arguments after variadics not supported yet");
  }
  return success();
}

/// Parse an argument list, including the parentheses around them.  The
/// argument list is allowed to be empty.  If `fnEffects` is non-null, then this
/// parses 'raises' and other effects.
ParseResult ParsedArgument::parseAndResolveParenthesizedArgumentList(
    ParserBase &p, SmallVectorImpl<ParsedArgument> &args, ArgListKind kind,
    FnEffects &fnEffects) {

  if (p.parseToken(Token::l_paren, "expected '(' for argument list"))
    return failure();

  if (!p.consumeIf(Token::r_paren)) {
    if (parseAndResolvePresentArgumentList(p, args, kind) ||
        p.parseToken(Token::r_paren, "expected ')' in argument list"))
      return failure();
  }

  // If the client supports function effects, parse them as well.
  // Parse other function effects.
  while (p.getToken().isIdentifier()) {
    SMLoc loc = p.getToken().getLoc();
    StringRef spelling = p.getToken().getSpelling();

    auto handleEffect = [&](auto hasFn, auto setFn) {
      if ((fnEffects.*hasFn)())
        p.emitError(loc, "function effect '")
            << spelling << "' was already specified";
      (fnEffects.*setFn)(true);
    };

    if (spelling == "raises") {
      handleEffect(&FnEffects::isThrows, &FnEffects::setThrows);
    } else if (spelling == "capturing") {
      handleEffect(&FnEffects::isCapturing, &FnEffects::setCapturing);
    } else if (spelling == "escaping") {
      handleEffect(&FnEffects::isEscaping, &FnEffects::setEscaping);
    } else {
      // If this isn't a known effect, then it could be an error like a missing
      // colon at the end of a function declaration.  If so, emit a nice error
      // and recover cleanly.
      if (p.getToken().isStartOfLine() && kind == ArgListKind::kArgList) {
        // Otherwise maybe it was misspelled, just eat it.
        p.emitError(p.getTokenLocOrEndOfPreviousLineIfOnNewLine(),
                    "missing ':' at end of function signature");
        return failure();
      }

      // Otherwise maybe it was misspelled, just eat it.
      p.emitError(loc, "unknown function effect '")
          << spelling << "', expected 'raises', 'capturing', or 'escaping'";
    }

    p.consumeIdentifier();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Parameter signature implementation
//===----------------------------------------------------------------------===//

/// Helper to emit a consistent error message when a required argument or
/// parameter follows a optional one.
static InflightDiag emitOptionalAfterRequired(ExprEmitter &emitter,
                                              const ParsedArgument &arg,
                                              StringRef argOrParam) {
  std::string kindStr =
      arg.kwArgHandling == ParsedArgument::KWArgHandling::kKeywordOnly
          ? "keyword-only"
          : "positional";
  return emitter.emitError(arg.loc, "required ")
         << kindStr << " " << argOrParam << " follows optional " << kindStr
         << " " << argOrParam;
}

/// Core implementation of the parameter argument parsing logic.
void ParsedParamSignature::typeCheck() {
  // Resolve each of the parameter declarations.
  ExprEmitter emitter(shared, declScope, EC_Type);

  bool seenPosInitExpr = false;
  bool seenKwOnlyInitExpr = false;
  for (const ParsedArgument &arg : parsedParams) {
    // Check for things supported in arguments that are not supported in
    // parameters.

    ASTType type;
    if (arg.typeExpr)
      type = emitter.emitExprType(arg.typeExpr);
    else
      emitter.emitError(arg.loc, "parameters must always have a type");
    if (!type)
      type = emitter.shared.getTypeCheckErrorType();

    VarArgKind vararg = arg.vararg;
    if (vararg == VarArgKind::PackVarArg)
      emitter.emitError(arg.loc, "parameters may not be variadic packs");

    if (vararg == VarArgKind::VarArg && !type.isTypeCheckErrorType()) {
      // TODO: What convention should we use for parameter varargs?
      type = VariadicType::get(type, ValueInputConvention::BorrowedInReg);
      isVarArgs = true;
    }

    if (arg.kwArgHandling == ParsedArgument::KWArgHandling::kKeywordOnly)
      seenPosInitExpr = false;

    if (const ExprNode *initExpr = arg.initExpr) {
      Type paramType = type;
      PValue value =
          emitter.emitExprPValue(initExpr, EC_DefaultParam, paramType);
      if (!value)
        return;
      if (arg.kwArgHandling == ParsedArgument::KWArgHandling::kKeywordOnly) {
        defaultKwOnlyParams.push_back(value);
        seenKwOnlyInitExpr = true;
      } else {
        defaultPosParams.push_back(value);
        seenPosInitExpr = true;
      }

    } else if (seenPosInitExpr || seenKwOnlyInitExpr) {
      emitOptionalAfterRequired(emitter, arg, "parameter")
          << arg.typeExpr->getRange();
    }

    // TODO: Parameter decls should support conventions at some point.
    if (arg.convention != ParsedArgument::kConventionUnspec)
      emitter.emitError(arg.loc, "parameters must always be passed by-value");

    // Bind the parsed type expression so references from other parameters
    // can be resolved. The parameter names in ParamDeclAttr are mangled with
    // the location so that parameter names in mojo are unique in the IR.
    auto newDecl =
        ParamDeclAttr::get(declScope.getUniqueParamNameNew(arg.name), type);
    paramDeclAttrs.push_back(newDecl);

    // The unmangled names are also collected to aid keyword parameter binding.
    passingKinds.emplace_back(
        ParsedArgument::mapToPassingKind(arg.kwArgHandling));
    names.push_back(arg.name);

    ASTDecl &resolvedDecl = emitter.getDeclResolver().addFullyResolvedDecl(
        PValue(ParamDeclRefAttr::get(newDecl)), arg.name, arg.loc, &declScope);
    emitter.shared.notifyListenerOnParameterDecl(resolvedDecl, arg.loc);
  }
}

/// param_signature    ::= "[" param_list ("->" param_result_types)? "]"
/// param_list   ::= argument_list | "(" ")"
/// param_result_types ::= expression ("," expression)*
ParseResult ParsedParamSignature::parseOptionalParameterSignature(
    ParserBase &p, SmallVectorImpl<ParsedArgument> &params,
    ParsedArgument::ArgListKind kind) {
  // Check to see if a parameter signature exists at all.
  if (!p.consumeIf(Token::l_square) || p.consumeIf(Token::r_square))
    return success();

  // Parse an actual parameter list.
  if (ParsedArgument::parseAndResolvePresentArgumentList(p, params, kind))
    return failure();

  return p.parseToken(Token::r_square, "expected ']' for parameter list");
}

/// Given a type that potentially has all of its parameters unbound, implicitly
/// add the parameter declarations to the function input parameters.
static ASTType
addImplicitTypeParams(SharedState &shared, ASTDecl &declScope, ASTType type,
                      const ParsedArgument &arg,
                      SmallVectorImpl<StringAttr> &inputParamNames,
                      SmallVectorImpl<PassingKind> &inputParamPassingKinds,
                      SmallVectorImpl<ParamDeclAttr> &inputParamDecls) {
  // Check if the type has unbound parameters.
  auto metatype = dyn_cast_or_null<MetaTypeType>(type.getMetaType());
  if (!metatype)
    return type;
  ArrayRef<Type> inputParams = metatype.getSignature().getInputParamTypes();
  if (inputParams.empty())
    return type;

  SmallVector<TypedAttr> paramValues;
  for (Type type : inputParams) {
    auto funcDecl = ParamDeclAttr::get(
        declScope.getUniqueParamNameNew(arg.name, /*isUserDefinedDecl=*/false),
        type);
    inputParamNames.push_back(StringAttr::get(type.getContext()));
    inputParamPassingKinds.push_back(PassingKind::Implicit);
    inputParamDecls.push_back(funcDecl);
    paramValues.push_back(ParamDeclRefAttr::get(funcDecl));
  }
  return BindTypeAttr::get(PValue(type), paramValues);
}

ASTType ParsedArgument::emitFunctionArgumentsAndResults(
    function_ref<ParseResult()> reportError, ExprEmitter &typeEmitter,
    SmallVectorImpl<StringAttr> &inputParamNames,
    SmallVectorImpl<PassingKind> &inputParamPassingKinds,
    SmallVectorImpl<ParamDeclAttr> &inputParamDecls,
    const ExprNode *resultTypeExpr, FnEffects &effects,
    SmallVectorImpl<ParsedArgument> &args, SmallVectorImpl<Type> &argTypes,
    SmallVectorImpl<TypedAttr> &defaultPosArgs,
    SmallVectorImpl<TypedAttr> &defaultKwOnlyArgs, bool isDef, SMLoc resultLoc,
    ASTDecl *fnDecl, SpecialFunctionInfo fnInfo) {
  SharedState &shared = typeEmitter.shared;
  ASTDecl &sigDecl = typeEmitter.declScope;
  // If this definition is a struct/class member, compute the self type.
  ASTType selfType;
  if (fnDecl) {
    ASTDecl *parent = fnDecl->getParentDecl();
    if (isa<StructDeclOp, TraitDeclOp>(*parent)) {
      // The parent decl must be fully resolved in order to resolve any of its
      // members.
      assert(parent->resolvedness == DeclResolvedness::fully);
      selfType = parent->getSelfType();
    }
  }

  // HACK: Create a dummy value to assign to argument declarations during
  // argument and result type emission.
  MLIRContext *ctx = typeEmitter.shared.getContext();
  SmallVector<OwningOpRef<ParamConstantOp>> argVals;
  auto makeDummy = [&](Type type) -> Value {
    return *argVals.emplace_back(OpBuilder(ctx).create<ParamConstantOp>(
        UnknownLoc::get(ctx), UnboundAttr::get(type)));
  };

  // Resolve all argument types, generating type check error types for any types
  // that could not be correctly resolved.
  bool seenPosInitExpr = false;
  bool seenKwOnlyInitExpr = false;
  for (auto [idx, arg] : llvm::enumerate(args)) {
    ASTType type;
    if (arg.typeExpr) {
      // Emit the argument type. Allow argument types to be "automatically"
      // parameterized: if the type is fully unbound, its input parameters are
      // appended to the function input parameters.
      type = typeEmitter.emitExprType(arg.typeExpr, /*allowUnbound=*/true);

      // If the type couldn't be emitted, mark this argument erroneous (so uses
      // within the body of the function don't trigger secondary errors) and
      // mark the function erroneous so calls to it won't resolve.  Put in a
      // placeholder type so we can continue type checking.
      if (!type) {
        if (reportError())
          return {};
        type = shared.getTypeCheckErrorType();
        arg.isErroneous = true;
      }
      type = addImplicitTypeParams(shared, sigDecl, type, arg, inputParamNames,
                                   inputParamPassingKinds, inputParamDecls);
    } else if (!idx && selfType && !cast<LIT::FuncOp>(fnDecl).getIsStatic()) {
      // If this is the 'self' argument in a struct, default the type to Self.
      type = selfType;
    } else if (isDef) {
      // In 'def', arguments with no types default to 'object'.
      type = shared.lookupObjectType(arg.loc, sigDecl);
      if (!type) {
        if (reportError())
          return {};
        type = shared.getTypeCheckErrorType();
        arg.isErroneous = true;
      }
    } else {
      // In an 'fn' we report an error.
      shared.emitError(arg.loc, "'fn' argument type must be specified")
          << SourceRange(arg.loc, arg.loc);
      if (reportError())
        return {};
      type = shared.getTypeCheckErrorType();
      arg.isErroneous = true;
    }
    assert(type && "must have an argument type");
    argTypes.push_back(type);

    // Determine the required function effects from the conventions.
    if (arg.vararg == VarArgKind::VarArg)
      effects.setVarArgs();
    else if (arg.vararg == VarArgKind::PackVarArg)
      effects.setPackVarArgs();
    else if (arg.vararg == VarArgKind::KWVarArg)
      effects.setKWVarArgs();

    // If no convention was explicitly specified, provide a default.  We default
    // to borrowed in an 'fn' or owned in a 'def'.
    if (arg.convention == ParsedArgument::kConventionUnspec) {
      arg.convention = isDef ? ParsedArgument::kConventionOwned
                             : ParsedArgument::kConventionBorrowed;

      // FIXME(owned varargs): we don't support owned varargs, so pass varargs
      // as borrowed instead for def's for now to hackaround this.
      if (isDef && arg.vararg != VarArgKind::None)
        arg.convention = ParsedArgument::kConventionBorrowed;
    }

    if (arg.kwArgHandling == KWArgHandling::kKeywordOnly)
      seenPosInitExpr = false;

    // Emit default argument values.
    if (const ExprNode *initExpr = arg.initExpr) {
      PValue value =
          typeEmitter.emitExprPValue(initExpr, EC_DefaultArgument, type);
      if (!value)
        return {};
      if (arg.kwArgHandling == KWArgHandling::kKeywordOnly) {
        defaultKwOnlyArgs.push_back(value);
        seenKwOnlyInitExpr = true;
      } else {
        defaultPosArgs.push_back(value);
        seenPosInitExpr = true;
      }
    } else if (seenPosInitExpr || seenKwOnlyInitExpr) {
      InflightDiag diag =
          emitOptionalAfterRequired(typeEmitter, arg, "argument");
      // Depending on `reportError`, the type might also be missing.
      if (arg.typeExpr)
        diag << arg.typeExpr->getRange();
    }

    // Add the declaration for the argument, now that is has been resolved. Use
    // a placeholder value to allow the value to be referenced, but in function
    // body resolution, it will be replaced with the actual function argument
    // SSA value.
    if (!arg.name.empty()) {
      typeEmitter.getDeclResolver().addFullyResolvedDecl(
          SRValue(makeDummy(type)), arg.name, arg.loc, &typeEmitter.declScope);
    }
  }

  // Compute the result type. If it is memory-only, insert it into the argument
  // list to be added to the signature.
  ASTType resultType;
  if (!resultTypeExpr) {
    resultType = shared.getNoneType();
    // Don't insert the return value for certain special functions.
    if (isDef && !fnInfo.hasNoneResult() && !fnInfo.isInitializer()) {
      // Insert an object memory-only result type.
      ParsedArgument resultArg;
      resultArg.loc = resultLoc;
      resultArg.name = StringAttr::get(shared.getContext(), "__result__");
      resultArg.convention = ParsedArgument::kConventionInOutResult;
      resultArg.kwArgHandling = ParsedArgument::KWArgHandling::kPositionalOnly;
      args.insert(args.begin(), resultArg);
      argTypes.insert(argTypes.begin(),
                      shared.lookupObjectType(resultLoc, sigDecl));
      if (!argTypes.front()) {
        if (reportError())
          return {};
        argTypes.front() = shared.getTypeCheckErrorType();
      }
    }
  } else if (resultTypeExpr->kind == ExprNode::kNoneLiteral) {
    // If the result type is a `None` literal, then convert it to NoneType.
    resultType = shared.getNoneType();
  } else {
    resultType = typeEmitter.emitExprType(resultTypeExpr);
    // On error, a diagnostic will be emitted, but we don't want to kill the
    // entire function definition.  We won't be able to correctly type check any
    // calls to this function though.
    if (!resultType) {
      if (reportError())
        return {};
      resultType = shared.getTypeCheckErrorType();
    }

    // Memory-only types get passed as the first argument to the function
    // by-reference.
    TypeConvention rp =
        resultType.getRegisterPassability(resultTypeExpr->getLoc(), shared);
    if (rp == TypeConvention::MemoryOnly) {
      // Synthesize a result argument for this, and use None as the actual
      // function result.
      ParsedArgument resultArg;
      resultArg.loc = resultTypeExpr->getLoc();
      resultArg.name = StringAttr::get(shared.getContext(), "__result__");
      resultArg.convention = ParsedArgument::kConventionInOutResult;
      resultArg.kwArgHandling = ParsedArgument::KWArgHandling::kPositionalOnly;
      resultArg.typeExpr = resultTypeExpr;
      args.insert(args.begin(), resultArg);
      argTypes.insert(argTypes.begin(), resultType);
      resultType = shared.getNoneType();
    } else if (rp != TypeConvention::RegisterPassableTrivial) {
      // We know the result type of the function is register passable (because
      // otherwise it would be promoted to an argument).  If the result of the
      // function is a non-trivial type, mark the function effect as having an
      // owned result so ownership tracking will notice it.
      effects.setOwnedRegisterResult();
    }
  }
  return resultType;
}

void DeclResolver::computeArgumentConventions(
    MutableArrayRef<ParsedArgument> args, MutableArrayRef<Type> argTypes,
    SmallVectorImpl<ParamDeclAttr> &implicitLifetimeDecls, ASTDecl &declScope) {
  // This closure is called for argument conventions that don't allow
  // variadics.
  auto rejectVariadic = [&](size_t argNo, const char *kind) -> bool {
    auto &arg = args[argNo];
    // If the arg isn't variadic, then it's fine.
    if (arg.vararg == VarArgKind::None)
      return false;
    // Emit an error and remember this error.
    if (!arg.isErroneous) {
      shared.emitError(arg.loc)
          << "'" << kind << "' arguments cannot be variadic";
      arg.isErroneous = true;
    }

    // Switch to a convention that is supportable.
    arg.convention = ParsedArgument::kConventionBorrowed;
    arg.kgenConvention = ValueInputConvention::BorrowedInReg;
    return true;
  };

  for (auto [i, arg, argType] : llvm::enumerate(args, argTypes)) {
    switch (arg.convention) {
    case ParsedArgument::kConventionUnspec:
      llvm_unreachable("should be resolved by now");
    case ParsedArgument::kConventionOwned:
      // Memory-only owned argument are passed with a layer of indirection and
      // use a specific convention to model this.
      if (ASTType(argType).isRegisterPassable(arg.loc, shared)) {
        arg.kgenConvention = ValueInputConvention::OwnedInReg;
        rejectVariadic(i, "owned");
      } else {
        arg.kgenConvention = ValueInputConvention::OwnedInMem;
      }
      break;
    case ParsedArgument::kConventionBorrowed:
      // Memory-only owned argument are passed with a layer of indirection and
      // use a specific convention to model this.
      if (ASTType(argType).isRegisterPassable(arg.loc, shared))
        arg.kgenConvention = ValueInputConvention::BorrowedInReg;
      else
        arg.kgenConvention = ValueInputConvention::BorrowedInMem;
      break;
    case ParsedArgument::kConventionInOut:
      arg.kgenConvention = ValueInputConvention::ByRef;
      break;
    case ParsedArgument::kConventionInOutResult:
      arg.kgenConvention = ValueInputConvention::ByRefResult;
      break;
    case ParsedArgument::kConventionInitSelfResult:
      // We also force the passing kind of self to positional-only.
      arg.kwArgHandling = ParsedArgument::KWArgHandling::kPositionalOnly;
      arg.kgenConvention = ValueInputConvention::InitSelf;
      break;
    }

    // Values passed by memory need an associated lifetime parameter, and
    // need to be passed by reference.
    if (SignatureType::hasAddress(arg.kgenConvention)) {
      // Given a memory argument named "foo" we give the implicit lifetime a
      // name of "`foo".  We do this because of Rust precedent, but also
      // because you can't spell this identifier in Mojo, even with backticks!
      StringAttr lifetimeName;
      if (arg.name) {
        lifetimeName = declScope.getAnonymousLifetimeFor(
            arg.name.str(), /*dontRenameOutermost=*/true);
      } else { // Used by function types, for example.
        lifetimeName = declScope.getAnonymousLifetimeFor(
            Twine(llvm::utostr(i)) + "_unnamed",
            /*dontRenameOutermost=*/true);
      }

      // The reference is immutable when borrowing, mutable otherwise.
      bool isMutable = arg.convention != ParsedArgument::kConventionBorrowed;

      auto lifetimeDecl = ParamDeclAttr::get(
          lifetimeName, LifetimeType::get(shared.getContext(), isMutable));
      implicitLifetimeDecls.push_back(lifetimeDecl);

      argType = RefType::get(
          argType, ParamDeclRefAttr::get(lifetimeName, lifetimeDecl.getType()));
    }

    // If this is a valid vararg argument, then we pass it as a variadic type.
    // The convention is to pass as a register value, in the case of a memory
    // value, we're passing the array of pointers by value.
    if (arg.vararg == VarArgKind::VarArg) {
      argType = VariadicType::get(argType, arg.kgenConvention);
      arg.kgenConvention = ValueInputConvention::BorrowedInReg;
    }
  }
}
