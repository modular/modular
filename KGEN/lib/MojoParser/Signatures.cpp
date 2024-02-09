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

    if (convention == kConventionInOut ||
        convention == kConventionInitSelfResult) {
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

/// This method handles the function argument list for a Python function.
/// Python has some pretty interesting rules where standalone '*' and '/'
/// markers (when used in place of an argument) actually change the
/// interpretation of other argument definitions by specifying how they behave
/// w.r.t. keyword arguments.  We check these here so the client doesn't
/// have to deal with them.
///
/// This classification logic is described here:
///   https://peps.python.org/pep-0570/#how-to-teach-this
///
static ParseResult
parseArgOrParamList(ParserBase &p, SmallVectorImpl<ParsedArgument> &parsedArgs,
                    ArgListKind kind) {
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
    if (parsedArgs.empty()) {
      p.emitError(loc, "'/' marker cannot be used at the start of the ")
          << argOrParam << " list";
    }

    // Ok, process it by changing all arguments we've seen to be positional
    // only.  The remaining ones will stay kPositionalOrKeyword though.
    for (ParsedArgument &arg : parsedArgs)
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
    auto marker = KWArgMarkerInfo::kNotMarker;
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
      if (foundName)
        return p.emitError(arg.loc, "unnamed ")
               << argOrParam << " cannot follow named " << argOrParam;

      if (hasSlashMarker || hasStarMarker)
        return p.emitError(arg.loc, "unnamed ")
               << argOrParam << " cannot follow '/' or '*'";
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
      return p.emitError(arg.loc, "'**' marker must be at end of ")
             << argOrParam << " list";
    }

    if (arg.vararg == VarArgKind::KWVarArg)
      return p.emitError(arg.loc,
                         "variadic keyword argument not supported yet");

    // Otherwise just remember the argument.
    parsedArgs.push_back(arg);
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
  for (ParsedArgument &arg : parsedArgs) {
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
  if (!parsedArgs.empty() &&
      parsedArgs.back().kwArgHandling == KWArgHandling::kKeywordOnly &&
      llvm::any_of(parsedArgs, hasVarArg)) {
    return p.emitError(
        parsedArgs.back().loc,
        "keyword-only arguments after variadics not supported yet");
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
  std::string kindStr = arg.kwArgHandling == KWArgHandling::kKeywordOnly
                            ? "keyword-only"
                            : "positional";
  return emitter.emitError(arg.loc, "required ")
         << kindStr << " " << argOrParam << " follows optional " << kindStr
         << " " << argOrParam;
}

/// Core implementation of the parameter argument parsing logic.
TypeCheckedParamList::TypeCheckedParamList(
    ArrayRef<ParsedArgument> parsedParams, ASTDecl &declScope,
    SharedState &shared)
    : declScope(declScope), shared(shared) {
  // Resolve each of the parameter declarations.
  ExprEmitter emitter(shared, declScope, EC_Type);

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
      type = VariadicType::get(type, ArgConvention::BorrowedInReg);
      isVarArgs = true;
    }

    if (const ExprNode *initExpr = arg.initExpr) {
      Type paramType = type;
      PValue value =
          emitter.emitExprPValue(initExpr, EC_DefaultParam, paramType);
      if (!value) {
        arg.isErroneous = true;
        value = PValue(UnknownAttr::get(paramType));
      }

      if (arg.kwArgHandling == KWArgHandling::kKeywordOnly)
        defaultKwOnlyParams.push_back(value);
      else
        defaultPosParams.push_back(value);

    } else {
      // Diagnose an invalid missing default argument: if we have any positional
      // defaults, then we require all the rest to have defaults until the
      // keyword-only section.
      //
      // If we've had any in the keyword-only section, we continue to require
      // them.  FIXME: Why? There is no ambiguity with some keyword-only
      // arguments having defaults.
      if ((!defaultPosParams.empty() &&
           arg.kwArgHandling != KWArgHandling::kKeywordOnly) ||
          !defaultKwOnlyParams.empty()) {
        emitOptionalAfterRequired(emitter, arg, "parameter")
            << arg.typeExpr->getRange();
      }
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
    passingKinds.push_back(arg.getKWArgHandlingAsPassingKind());
    names.push_back(arg.name);

    ASTDecl &resolvedDecl = emitter.getDeclResolver().addFullyResolvedDecl(
        PValue(ParamDeclRefAttr::get(newDecl)), arg.name, arg.loc, &declScope);
    emitter.shared.notifyListenerOnParameterDecl(resolvedDecl, arg.loc);
  }
}

/// param_signature    ::= "[" param_list ("->" param_result_types)? "]"
/// param_list   ::= argument_list | "(" ")"
/// param_result_types ::= expression ("," expression)*
ParseResult ParsedParamList::parseOptionalParameters(ParserBase &p,
                                                     ArgListKind kind) {
  // Check to see if a parameter signature exists at all.
  if (!p.consumeIf(Token::l_square) || p.consumeIf(Token::r_square))
    return success();

  // Parse an actual parameter list.
  if (parseArgOrParamList(p, params, kind))
    return failure();

  return p.parseToken(Token::r_square, "expected ']' for parameter list");
}

/// Given a type that potentially has all of its parameters unbound, implicitly
/// add the parameter declarations to the function parameters.
static ASTType addImplicitTypeParams(SharedState &shared, ASTDecl &declScope,
                                     ASTType type, const ParsedArgument &arg,
                                     TypeCheckedParamList &paramList) {
  // Check if the type has unbound parameters.
  auto metatype = dyn_cast_or_null<MetaTypeType>(type.getMetaType());
  if (!metatype)
    return type;
  ArrayRef<Type> params = metatype.getSignature().getParamTypes();
  if (params.empty())
    return type;

  SmallVector<TypedAttr> paramValues;
  for (Type type : params) {
    auto funcDecl = ParamDeclAttr::get(
        declScope.getUniqueParamNameNew(arg.name, /*isUserDefinedDecl=*/false),
        type);
    paramList.names.push_back(StringAttr::get(type.getContext()));
    paramList.passingKinds.push_back(PassingKind::Implicit);
    paramList.paramDeclAttrs.push_back(funcDecl);
    paramValues.push_back(ParamDeclRefAttr::get(funcDecl));
  }
  return BindTypeAttr::get(PValue(type), paramValues);
}

//===----------------------------------------------------------------------===//
// Function Signature Parsing
//===----------------------------------------------------------------------===//

/// Parse an argument list, including the parentheses around them.  This also
/// parses 'raises' and other effects.
ParseResult ParsedArgumentList::parseArgumentListAndEffects(ParserBase &p,
                                                            ArgListKind kind) {

  // If this is a bare lambda argument list, it won't be parenthesized and won't
  // have effects.
  if (kind == ArgListKind::kBareLambdaArgList)
    return parseArgOrParamList(p, parsedArgs, kind);

  if (p.parseToken(Token::l_paren, "expected '(' for argument list"))
    return failure();

  if (!p.consumeIf(Token::r_paren)) {
    if (parseArgOrParamList(p, parsedArgs, kind) ||
        p.parseToken(Token::r_paren, "expected ')' in argument list"))
      return failure();
  }

  // If the client supports function effects, parse them as well.
  // Parse other function effects.
  while (p.getToken().isIdentifier()) {
    SMLoc loc = p.getToken().getLoc();
    StringRef spelling = p.getToken().getSpelling();

    auto handleEffect = [&](auto hasFn, auto setFn) {
      if ((effects.*hasFn)())
        p.emitError(loc, "function effect '")
            << spelling << "' was already specified";
      (effects.*setFn)(true);
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

/// This function creates a new anonymous lifetime decl for the specified
/// argument, and wraps the type with a reference using that lifetime.
static std::pair<ParamDeclAttr, RefType>
getRefAndLifetimeForAddressArgument(ParsedArgument &arg, size_t idx, Type type,
                                    ASTDecl &declScope) {
  // Given a memory argument named "foo" we give the implicit lifetime a
  // name of "`foo".  We do this because of Rust precedent, but also
  // because you can't spell this identifier in Mojo, even with backticks!
  StringAttr lifetimeName;
  if (arg.name) {
    lifetimeName = declScope.getAnonymousLifetimeFor(
        arg.name.str(), /*dontRenameOutermost=*/true);
  } else { // Used by function types, for example.
    lifetimeName =
        declScope.getAnonymousLifetimeFor(Twine(llvm::utostr(idx)) + "_unnamed",
                                          /*dontRenameOutermost=*/true);
  }

  // The reference is immutable when borrowing, mutable otherwise.
  bool isMutable = arg.convention != ParsedArgument::kConventionBorrowed;
  auto lifetimeDecl = ParamDeclAttr::get(
      lifetimeName, LifetimeType::get(lifetimeName.getContext(), isMutable));

  auto refType = RefType::get(
      type, ParamDeclRefAttr::get(lifetimeName, lifetimeDecl.getType()));
  return {lifetimeDecl, refType};
}

/// Type check each argument in turn, resolving their type and default
/// initializer value.  Arguments in Mojo can refer to previous arguments in
/// their type+default value expressions as PValues, so we need to ensure that
/// they are emitted and have declarations registered in the scope so that later
/// lookups can find them.
static void typeCheckOneArgument(size_t idx, ASTType selfType, bool isDef,
                                 bool isStaticMethod, ASTDecl *fnDecl,
                                 TypeCheckedFnSignature &tcSignature) {
  auto &arg = tcSignature.argList.parsedArgs[idx];

  ASTDecl &declScope = tcSignature.paramList.declScope;
  SharedState &shared = tcSignature.paramList.shared;
  ExprEmitter typeEmitter(shared, declScope, EC_Type);

  // Start by computing the declared type of the argument.
  ASTType type;
  if (arg.typeExpr) {
    // Emit the argument type. Allow argument types to be "automatically"
    // parameterized: if the type is fully unbound, its parameters are
    // appended to the function parameters.
    type = typeEmitter.emitExprType(arg.typeExpr, /*allowUnbound=*/true);

    // If the type couldn't be emitted, mark this argument erroneous (so uses
    // within the body of the function don't trigger secondary errors) and
    // mark the function erroneous so calls to it won't resolve.  Put in a
    // placeholder type so we can continue type checking.
    if (!type) {
      type = shared.getTypeCheckErrorType();
      arg.isErroneous = true;
    }
    type = addImplicitTypeParams(shared, declScope, type, arg,
                                 tcSignature.paramList);
  } else if (idx == 0 && selfType &&
             // FIXME: This is incorrect, the @static_method decorators haven't
             // been applied yet.
             !isStaticMethod) {
    // If this is the 'self' argument in a struct, default the type to Self.
    type = selfType;
  } else if (isDef) {
    // In 'def', arguments with no types default to 'object'.
    type = shared.lookupObjectType(arg.loc, declScope);
    if (!type) {
      type = shared.getTypeCheckErrorType();
      arg.isErroneous = true;
    }
  } else {
    // In an 'fn' we report an error.
    shared.emitError(arg.loc, "'fn' argument type must be specified")
        << SourceRange(arg.loc, arg.loc);
    type = shared.getTypeCheckErrorType();
    arg.isErroneous = true;
  }
  assert(type && "must have an argument type");
  tcSignature.argTypes.push_back(type);

  // Check if the argument is a parametric function.
  if (auto fType = dyn_cast<LITSignatureType>(type)) {
    if (fType.getNumParams() != 0) {
      arg.isErroneous = true;
      shared.emitError(shared.diags.translateLocation(arg.typeExpr->getLoc()),
                       "parametric functions may not be used as arguments; "
                       "consider passing as a parameter instead");
    }
  }

  // Determine the required function effects from the conventions.
  if (arg.vararg == VarArgKind::VarArg)
    tcSignature.argList.varEffects.setVarArgs();
  else if (arg.vararg == VarArgKind::PackVarArg)
    tcSignature.argList.varEffects.setPackVarArgs();
  else if (arg.vararg == VarArgKind::KWVarArg)
    tcSignature.argList.varEffects.setKWVarArgs();

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

  // Emit default argument values if present.
  if (const ExprNode *initExpr = arg.initExpr) {
    PValue value =
        typeEmitter.emitExprPValue(initExpr, EC_DefaultArgument, type);
    if (!value) {
      arg.isErroneous = true;
      value = PValue(UnknownAttr::get(type));
    }
    if (arg.kwArgHandling == KWArgHandling::kKeywordOnly)
      tcSignature.defaultKwOnlyArgs.push_back(value);
    else
      tcSignature.defaultPosArgs.push_back(value);
  } else {
    // Diagnose an invalid missing default argument: if we have any positional
    // defaults, then we require all the rest to have defaults until the
    // keyword-only section.
    //
    // If we've had any in the keyword-only section, we continue to require
    // them.  FIXME: Why? There is no ambiguity with some keyword-only
    // arguments having defaults.
    if ((!tcSignature.defaultPosArgs.empty() &&
         arg.kwArgHandling != KWArgHandling::kKeywordOnly) ||
        !tcSignature.defaultKwOnlyArgs.empty()) {
      InflightDiag diag =
          emitOptionalAfterRequired(typeEmitter, arg, "argument");
      if (arg.typeExpr)
        diag << arg.typeExpr->getRange();
    }
  }

  // Now that we have the declared type and default value sorted, apply the
  // argument convention to compute the full type for the argument.
  switch (arg.convention) {
  case ParsedArgument::kConventionUnspec:
    llvm_unreachable("should be resolved by now");
  case ParsedArgument::kConventionInOutResult:
    llvm_unreachable("shouldn't occur in an argument list");
  case ParsedArgument::kConventionOwned:
    // Memory-only owned argument are passed with a layer of indirection and
    // use a specific convention to model this.
    if (!type.isRegisterPassable(arg.loc, shared)) {
      arg.kgenConvention = ArgConvention::OwnedInMem;
      break;
    }
    arg.kgenConvention = ArgConvention::OwnedInReg;

    // Reject variadic owned register arguments, we can't support them yet.
    // TODO(ownership): Enable 'owned' @register_passable variadics.
    if (arg.vararg != VarArgKind::None) {
      // Emit an error and remember this error.
      if (!arg.isErroneous) {
        shared.emitError(arg.loc)
            << "'owned' @register_passable arguments cannot be variadic";
        arg.isErroneous = true;
      }

      // Switch to a convention that is supportable.
      arg.convention = ParsedArgument::kConventionBorrowed;
      arg.kgenConvention = ArgConvention::BorrowedInReg;
    }
    break;
  case ParsedArgument::kConventionBorrowed:
    if (type.isRegisterPassable(arg.loc, shared))
      arg.kgenConvention = ArgConvention::BorrowedInReg;
    else
      arg.kgenConvention = ArgConvention::BorrowedInMem;
    break;
  case ParsedArgument::kConventionInOut:
    arg.kgenConvention = ArgConvention::ByRef;
    break;
  case ParsedArgument::kConventionInitSelfResult:
    arg.kgenConvention = ArgConvention::InitSelf;
    break;
  }

  // Values passed by memory need an associated lifetime parameter, and
  // need to be passed by reference.
  Type fullType;
  if (!SignatureType::hasAddress(arg.kgenConvention)) {
    fullType = type;
  } else {
    auto [lifetimeDecl, refType] =
        getRefAndLifetimeForAddressArgument(arg, idx, type, declScope);
    tcSignature.implicitLifetimeDecls.push_back(lifetimeDecl);
    fullType = refType;
  }

  // If this is a valid vararg argument, then we pass it as a variadic type.
  // The convention is to pass as a register value, in the case of a memory
  // value, we're passing the array of pointers by value.
  if (arg.vararg == VarArgKind::VarArg) {
    fullType = VariadicType::get(fullType, arg.kgenConvention);
    arg.kgenConvention = ArgConvention::BorrowedInReg;
  }
  tcSignature.fullArgTypes.push_back(fullType);

  // Add the declaration for the argument, now that is has been resolved. Use
  // a placeholder value to allow the value to be referenced, but in function
  // body resolution, it will be replaced with the actual function argument
  // SSA value.
  //
  // Names are always present for function bodies, but can be missing in
  // function types.  In that case, there are obviously no dependent values on
  // it, because they can't be named.
  if (arg.name.empty())
    return;

  // Create the block argument that will eventually represent this function
  // argument.  If we're generating this argument for a function, put it into
  // its entry block. Otherwise it is a function type:We allocate the argument
  // into a holding block owned by SharedState so it isn't leaked.
  Block &blockOwningArg = fnDecl ? *cast<LIT::FuncOp>(fnDecl).getBody()
                                 : shared.getArgumentOwningBlock();
  BlockArgument blockArg =
      blockOwningArg.addArgument(fullType, shared.translateLocation(arg.loc));

  DeclIRValue argIRValue;
  switch (arg.kgenConvention) {
  case ArgConvention::ByRef:
  case ArgConvention::InitSelf:
  case ArgConvention::ByRefResult:
  case ArgConvention::OwnedInMem:
    // OwnedInMem passes ownership of the argument into the callee so we
    // can directly mutate it if we want to.
    argIRValue = MLValue(blockArg);
    break;
  case ArgConvention::OwnedInReg:
    // NOTE: we're treating this as an SRValue, even though it will get
    // wrapped and turned into an SLValue within the body.
    argIRValue = SRValue(blockArg);
    break;
  case ArgConvention::BorrowedInReg:
    argIRValue = SBValue(blockArg);
    break;
  case ArgConvention::BorrowedInMem:
    argIRValue = MBValue(blockArg);
    break;
  case ArgConvention::None:
    llvm_unreachable("none convention not permitted in lit");
  }

  // FIXME: This is not setting the correct type for Variadics.  We shouldn't
  // expose something like !kgen.variadic to subsequent arguments, we should
  // expose VariadicListMem.  This will require moving the VariadicList
  // formation to the caller side.
  typeEmitter.getDeclResolver().addFullyResolvedDecl(
      argIRValue, arg.name, arg.loc, &typeEmitter.declScope);
}

/// Type check the result type for the function.  `resultTypeExpr` will be
/// non-null if explicitly specified in source code, and the `resultLoc` will
/// always be valid point for end of the argument list.
static void typeCheckResult(const ExprNode *resultTypeExpr, SMLoc resultLoc,
                            bool isDef, const SpecialFunctionInfo &fnInfo,
                            ASTDecl *fnDecl,
                            TypeCheckedFnSignature &tcSignature) {
  ASTDecl &declScope = tcSignature.paramList.declScope;
  SharedState &shared = tcSignature.paramList.shared;

  ASTType resultType;
  if (!resultTypeExpr) {
    // If the result type wasn't specified, we default to either "None" or
    // "object" depending on whether this is a def.
    resultType = shared.getNoneType();

    // If this is a 'def', then we want to default to 'object' unless this is a
    // known function that doesn't support that.
    if (isDef && !fnInfo.hasNoneResult() && !fnInfo.isInitializer()) {
      resultType = shared.lookupObjectType(resultLoc, declScope);
      if (!resultType)
        resultType = shared.getTypeCheckErrorType();
    }
  } else if (resultTypeExpr->kind == ExprNode::kNoneLiteral) {
    // If the result type is a `None` literal, then convert it to NoneType.
    resultType = shared.getNoneType();
  } else {
    ExprEmitter typeEmitter(shared, declScope, EC_Type);
    resultType = typeEmitter.emitExprType(resultTypeExpr);

    // On error, a diagnostic will be emitted, but we don't want to kill the
    // entire function definition.  We won't be able to correctly type check any
    // calls to this function though.
    if (!resultType)
      resultType = shared.getTypeCheckErrorType();
  }

  // Remember the user-declared result type.
  tcSignature.resultType = resultType;

  // Now that we have the user's result type, compute the full type of the
  // result, which can can be different when memory only, when throwing, etc.
  ASTType fullResultType = resultType;

  // If it is memory-only, pass it indirectly as the first argument to the
  // function by-reference.
  TypeConvention rp = resultType.getRegisterPassability(resultLoc, shared);
  if (rp == TypeConvention::MemoryOnly) {
    // FIXME(#26008): Async functions with memory-only do not work.
    if (tcSignature.argList.effects.isAsync()) {
      shared.emitError(
          resultLoc, "async functions do not support memory-only results yet");
      tcSignature.argList.effects.setAsync(false);
    }

    // Synthesize a result argument for this, and use None as the actual
    // function result.
    ParsedArgument resultArg;
    resultArg.loc = resultLoc;
    resultArg.name = StringAttr::get(shared.getContext(), "__result__");
    resultArg.convention = ParsedArgument::kConventionInOutResult;
    resultArg.kgenConvention = ArgConvention::ByRefResult;
    resultArg.kwArgHandling = KWArgHandling::kPositionalOnly;
    resultArg.typeExpr = resultTypeExpr;

    // Insert this into the FRONT of the argument list.
    // TODO(#30134): pass byref result at the end of the list.
    tcSignature.argList.parsedArgs.insert(
        tcSignature.argList.parsedArgs.begin(), resultArg);
    tcSignature.argTypes.insert(tcSignature.argTypes.begin(), resultType);

    // Compute the full type for this new argument and a lifetime for it.
    auto [lifetimeDecl, refType] = getRefAndLifetimeForAddressArgument(
        resultArg, 0, resultType, declScope);
    tcSignature.implicitLifetimeDecls.insert(
        tcSignature.implicitLifetimeDecls.begin(), lifetimeDecl);
    tcSignature.fullArgTypes.insert(tcSignature.fullArgTypes.begin(), refType);

    // If this is for a lit.func declaration (as opposed to a function type),
    // add a block argument for this.  We don't register this for name lookup
    // though, we don't want it to conflict with user identifiers, and it is
    // never looked up directly.
    if (fnDecl) {
      Block &body = *cast<LIT::FuncOp>(fnDecl).getBody();
      (void)body.insertArgument(0U, refType,
                                shared.translateLocation(resultLoc));
    }

    // We know the ABI result will be None now, which is trivial.
    fullResultType = shared.getNoneType();
    rp = TypeConvention::RegisterPassableTrivial;
  }

  if (tcSignature.argList.effects.isThrows()) {
    Type errorType = shared.getBuiltinErrorType(declScope, resultLoc);
    fullResultType = VariantType::get({errorType, fullResultType});
    // The result is always owned because the function returns a variant
    // containing an Error, which is nontrivial.
    rp = TypeConvention::RegisterPassable;
  }
  tcSignature.fullResultType = fullResultType;

  // If the result of the function is a non-trivial type, mark the function
  // effect as having an owned result so ownership tracking will notice it.
  if (rp != TypeConvention::RegisterPassableTrivial)
    tcSignature.argList.effects.setOwnedRegisterResult();
}

/// Emit the argument types, default values, and result type and determine
/// the argument conventions.
///
/// 'fnDecl' will be null when this is a function type, which doesn't have a
/// declaration.
TypeCheckedFnSignature::TypeCheckedFnSignature(TypeCheckedParamList &paramList,
                                               ParsedArgumentList &argList,
                                               const ExprNode *resultTypeExpr,
                                               SMLoc resultLoc, bool isDef,
                                               ASTDecl *fnDecl,
                                               SpecialFunctionInfo &fnInfo)
    : paramList(paramList), argList(argList) {

  SharedState &shared = paramList.shared;
  ExprEmitter typeEmitter(shared, paramList.declScope, EC_Type);

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

  // If this is a well-known function like `__init__`, perform early semantic
  // checks and clarify what special function it really is.

  // __*init__ methods are weird - for memory-only results we define
  // init in convention Python style, but for @register_passable values, we
  // return it.  We handle this by mapping them to different enumerators so
  // things downstream have stronger invariants.
  //
  // This logic happens before type checking, so we need to be very careful
  // to only process it if defined correctly.  We let downstream checks diagnose
  // the errors.
  if ((fnInfo.kind == SpecialFunctionKind::kInit ||
       fnInfo.kind == SpecialFunctionKind::kCopyInit ||
       fnInfo.kind == SpecialFunctionKind::kMoveInit) &&
      selfType) {
    // If this is a memory-only type, then the self argument is actually passed
    // with a special thing, it is written inout, but it isn't really.
    if (!selfType.isRegisterPassable(fnDecl->getLoc(), shared)) {
      if (!argList.parsedArgs.empty() && argList.parsedArgs[0].convention ==
                                             ParsedArgument::kConventionInOut) {
        auto &selfArg = argList.parsedArgs[0];
        selfArg.convention = ParsedArgument::kConventionInitSelfResult;
        // We also force the passing kind of self to positional-only.
        if (selfArg.kwArgHandling == KWArgHandling::kPositionalOrKeyword)
          selfArg.kwArgHandling = KWArgHandling::kPositionalOnly;
      }
    } else {
      if (fnInfo.kind == SpecialFunctionKind::kCopyInit)
        fnInfo = SpecialFunctionInfo::get(SpecialFunctionKind::kCopyInitReg);
      else if (fnInfo.kind == SpecialFunctionKind::kInit)
        fnInfo = SpecialFunctionInfo::get(SpecialFunctionKind::kInitReg);
      else {
        assert(fnInfo.kind == SpecialFunctionKind::kMoveInit);
        fnDecl->hasReferenceError = true;
        paramList.shared.emitError(fnDecl->getLoc(), "'")
            << fnInfo.name
            << "' is not supported for @register_passable types, they "
               "are always movable by copying a register";
        fnInfo = SpecialFunctionInfo();
      }
    }
  }

  // __new__ and __init__ in register types are implicitly static.
  if (fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod)
    cast<LIT::FuncOp>(fnDecl).setIsStatic(true);

  // True if this is a static method.
  // FIXME: This is completely wrong, @static_method decorator hasn't been
  // applied yet.
  bool isStaticMethod = selfType && cast<LIT::FuncOp>(fnDecl).getIsStatic();

  // Resolve all argument types, generating type check error types for any types
  // that could not be correctly resolved.
  for (size_t i = 0, e = argList.parsedArgs.size(); i != e; ++i)
    typeCheckOneArgument(i, selfType, isDef, isStaticMethod, fnDecl, *this);

  // Compute the result type.
  typeCheckResult(resultTypeExpr, resultLoc, isDef, fnInfo, fnDecl, *this);
}

FunctionType TypeCheckedFnSignature::getFunctionType() const {
  return FunctionType::get(fullResultType.mlirType.getContext(), fullArgTypes,
                           {fullResultType.mlirType});
}

/// Form a LIT signature packaging up all the stuff we need to know about this
/// type checked function.
LITSignatureType TypeCheckedFnSignature::getLITSignatureType() const {
  MLIRContext *ctx = paramList.shared.getContext();

  SmallVector<StringAttr> argNames;
  SmallVector<PassingKind> argPassingKinds;
  SmallVector<ArgConvention> argConventions;
  argNames.reserve(argList.parsedArgs.size());
  argPassingKinds.reserve(argList.parsedArgs.size());
  argConventions.reserve(argList.parsedArgs.size());
  for (const ParsedArgument &arg : argList.parsedArgs) {
    argPassingKinds.push_back(arg.getKWArgHandlingAsPassingKind());
    argNames.push_back(arg.name);
    argConventions.push_back(arg.kgenConvention);
  }

  auto metadata = FnMetadataAttr::get(
      ArgParamListAttr::get(ctx, argNames, argPassingKinds, defaultPosArgs,
                            defaultKwOnlyArgs, /*variadicIndices=*/{},
                            /*packIndices=*/{}),
      ArgParamListAttr::get(ctx, paramList.names, paramList.passingKinds,
                            paramList.defaultPosParams,
                            paramList.defaultKwOnlyParams,
                            /*variadicIndices=*/{}, /*packIndices=*/{}),
      implicitLifetimeDecls.size(), argList.varEffects);

  /// Silence internal verifier errors when constructing types from the parser.
  /// We don't want to show these to the user.
  auto silenceErrors = [ctx] {
    InFlightDiagnostic diag = mlir::emitError(UnknownLoc::get(ctx));
    diag.abandon();
    return diag;
  };

  FunctionType functionType = getFunctionType();
  return SignatureType::remapToSignature(
      paramList.paramDeclAttrs, {}, functionType, argConventions,
      argList.effects, metadata, silenceErrors);
}
