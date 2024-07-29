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
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/ParserBase.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "llvm/ADT/StringExtras.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

/// Process the lifetime expression in a `ref [...] T` reference specifier.
/// T is specified as 'type' and this returns the result !lit.ref type.
static ASTType processLifetimeSpecifier(const ExprNode *lifetimeExpr,
                                        ASTType type, StringRef valueName,
                                        TypeCheckedParamList &paramList,
                                        bool isResult) {
  SharedState &shared = paramList.shared;

  // For errors, return "RefType(TypeCheckErrorType)" to maintain the invariant
  // that all "ref" values have RefType, but their RValue type is an error.
  auto hadError = [&]() -> ASTType {
    return RefType::getImmortal(shared.getTypeCheckErrorType(), /*isMut*/ true);
  };

  // Propagate already disgnosed errors.
  if (isa<TypeCheckErrorType>(type))
    return hadError();

  ExprEmitter emitter(shared, paramList.declScope, EC_Lifetime);

  // If the lifetime expression is syntactically a 2-element tuple, then
  // take it apart into a lifetime and address space.
  ExprNode *addrSpaceExpr = nullptr;
  if (auto *tuple = dyn_cast<TupleNode>(lifetimeExpr)) {
    if (tuple->exprs.size() != 2) {
      emitter.emitError(tuple->getLoc())
          << "expected specifier with one lifetime or a lifetime and an "
             "address space"
          << lifetimeExpr->getRange();
      return hadError();
    }

    lifetimeExpr = tuple->exprs[0];
    addrSpaceExpr = tuple->exprs[1];
  }

  // Emit the lifetime expression if it is a normal expression.
  PValue lifetime;
  if (lifetimeExpr->kind != ExprNode::kDiscardLiteral) {
    lifetime = emitter.emitExprPValue(lifetimeExpr, EC_Lifetime);
  } else {
    // We need to add two parameters to this function, one for the mutability
    // of type Bool and one for the lifetime.
    auto addParam = [&](const Twine &name, Type type) -> TypedAttr {
      auto paramDecl =
          ParamDeclAttr::get(paramList.declScope.mangleParamName(name), type);
      paramList.names.push_back(StringAttr::get(type.getContext()));
      paramList.passingKinds.push_back(PassingKind::Implicit);
      paramList.paramDeclAttrs.push_back(paramDecl);
      return ParamDeclRefAttr::get(paramDecl);
    };

    if (isResult) {
      emitter.emitError(lifetimeExpr->getLoc())
          << "cannot infer lifetime for a function result"
          << lifetimeExpr->getRange();
      return hadError();
    }

    auto isMut = addParam(valueName + "_is_mut",
                          IntegerType::get(shared.getContext(), 1));
    lifetime = addParam(valueName + "_is_lifetime", LifetimeType::get(isMut));
  }
  if (!lifetime)
    return hadError();

  if (!isa<LifetimeType>(lifetime.getType())) {
    emitter.emitError(lifetimeExpr->getLoc())
        << "result reference lifetime has unexpected type "
        << lifetime.getType() << lifetimeExpr->getRange();
    return hadError();
  }

  // If we have an address space, emit it.
  TypedAttr addrSpace;
  if (addrSpaceExpr) {
    auto addrSpaceCValue = emitter.emitExprCValue(addrSpaceExpr, EC_Lifetime);
    // Invoke __mlir_index__ to get Int/AddressSpace to what we need.
    if (addrSpaceCValue && !isa<IndexType>(addrSpaceCValue.getRValueType())) {
      ValueDest dest(EC_Lifetime);
      addrSpaceCValue = emitter.emitNamedMethodCall(
          "__mlir_index__", {{{addrSpaceCValue, addrSpaceExpr}}}, dest,
          CallSyntax::kMethodCall, addrSpaceExpr);
    }

    addrSpace =
        emitter.emitPValue({addrSpaceCValue, addrSpaceExpr}, EC_Lifetime).get();

    if (addrSpace && !isa<IndexType>(addrSpace.getType())) {
      emitter.emitError(lifetimeExpr->getLoc())
          << "INTERNAL ERROR: __mlir_index didn't return a value of index type"
          << addrSpace.getType() << lifetimeExpr->getRange();
      return hadError();
    }
  }
  if (!addrSpace)
    addrSpace = IntegerAttr::get(IndexType::get(shared.getContext()), 0);

  return RefType::get(type, lifetime, addrSpace);
}

//===----------------------------------------------------------------------===//
// Argument and Parameter List Parsing
//===----------------------------------------------------------------------===//

ParseResult ParsedArgument::parse(ParserBase &p, KWArgMarkerInfo &markerInfo,
                                  ArgListKind kind) {
  loc = p.getToken().getLoc();
  cursor = p.getLexer().getCursor();

  // Any owned/borrowed/inout/ref keyword sets convention.
  // TODO: Turn all of these into soft keywords.
  if (p.consumeIf(Token::kw_owned))
    convention = kConventionOwned;
  else if (p.consumeIf(Token::kw_borrowed))
    convention = kConventionBorrowed;
  else if (p.consumeIf(Token::kw_inout))
    convention = kConventionInOut;
  else if (p.getToken().is(Token::kw_ref)) {
    if (succeeded(p.parseRefSpecifier(refLifetimeExpr)))
      convention = kConventionRef;
  }

  while (p.getToken().isAny(Token::kw_owned, Token::kw_borrowed,
                            Token::kw_inout, Token::kw_ref)) {
    p.emitTokenError("argument already has a convention specified");
    p.consumeToken();
  }

  markerInfo = KWArgMarkerInfo::kNotMarker;

  // The first token of an argument may be a standalone '*', '/', or '//'
  // marker, and the '*' may also be part of a varargs specification.  Check for
  // these first.
  if (p.consumeIf(Token::slash)) {
    markerInfo = KWArgMarkerInfo::kSlash;
    return success();
  }
  if (p.getToken().isAny(Token::slash_slash)) {
    if (kind != ArgListKind::kParamList &&
        kind != ArgListKind::kFnTypeParamList) {
      p.emitTokenError("'//' can only be used in parameter lists to denote "
                       "inferred parameters");
    }
    p.consumeToken();
    markerInfo = KWArgMarkerInfo::kSlashSlash;
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
    if (succeeded(p.parseOptionalIdentifier(maybeArgName, Token::colon)))
      name = maybeArgName;
  } else {
    StringRef argOrParam =
        kind == ArgListKind::kParamList || kind == ArgListKind::kFnTypeParamList
            ? "parameter"
            : "argument";
    if (p.parseIdentifier(name, "expected " + argOrParam + " name", &loc)) {
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
        p.consumeToken(Token::star);
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

PassingKind ParsedArgument::getKWArgHandlingAsPassingKind() const {
  // Result slots are not handled through normal call argument resolution.
  if (SignatureType::isResultSlot(kgenConvention))
    return PassingKind::Implicit;

  switch (kwArgHandling) {
  case KWArgHandling::kInferred:
    return PassingKind::Inferred;
  case KWArgHandling::kPositionalOnly:
    return PassingKind::PosOnly;
  case KWArgHandling::kKeywordOnly:
    return PassingKind::KwOnly;
  case KWArgHandling::kPositionalOrKeyword:
    return PassingKind::PosOrKw;
  }
  llvm_unreachable("unhandled KWArgHandling");
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
  bool hasSlashSlashMarker = false, hasSlashMarker = false,
       hasStarMarker = false;
  auto defaultKWArgHandling = KWArgHandling::kPositionalOrKeyword;

  StringRef argOrParam =
      kind == ArgListKind::kParamList || kind == ArgListKind::kFnTypeParamList
          ? "parameter"
          : "argument";

  // This is invoked when we see a '//' marker.
  auto handleSlashSlashMarker = [&](SMLoc loc) {
    if (hasSlashSlashMarker) {
      p.emitError(loc, "cannot have two '//' markers in the same ")
          << argOrParam << " list";
      return;
    }
    if (hasSlashMarker) {
      p.emitError(loc, "cannot specify '//' marker after '/' marker in ")
          << argOrParam << " list";
      return;
    }
    if (hasStarMarker) {
      p.emitError(loc, "cannot specify '//' marker after '*' marker in ")
          << argOrParam << " list";
      return;
    }
    if (parsedArgs.empty()) {
      p.emitError(loc, "'//' marker cannot be used at the start of the ")
          << argOrParam << " list";
    }

    // Ok, process it by changing all parameter we've seen to be inferred only.
    // The remaining ones will stay kPositionalOrKeyword.
    for (ParsedArgument &arg : parsedArgs) {
      arg.kwArgHandling = KWArgHandling::kInferred;
      if (arg.initExpr) {
        p.emitError(arg.loc, "inferred parameters may not have defaults")
            << arg.initExpr->getRange();
        arg.initExpr = nullptr;
      }
    }

    hasSlashSlashMarker = true;
  };

  // This is invoked when we see a '/' marker.
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

    // Ok, process it by changing all arguments we've seen that aren't inferred
    // to be positional only. The remaining ones will stay kPositionalOrKeyword.
    for (ParsedArgument &arg : parsedArgs)
      if (arg.kwArgHandling != KWArgHandling::kInferred)
        arg.kwArgHandling = KWArgHandling::kPositionalOnly;
    hasSlashMarker = true;
  };

  // This is invoked when we see a '*' marker or '*arg' argument.
  auto handleStarMarker = [&](SMLoc loc, bool isMarker) -> ParseResult {
    if (hasStarMarker) {
      return p.emitError(loc, "cannot have two '*' markers in the same ")
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

    return success();
  };

  // This parses either an argument or a keyword argument specifier.
  bool foundName = false;
  bool foundKwargs = false;
  auto parseArgument = [&]() -> ParseResult {
    auto marker = KWArgMarkerInfo::kNotMarker;
    ParsedArgument arg;
    arg.kwArgHandling = defaultKWArgHandling;
    if (arg.parse(p, marker, kind))
      return failure();

    // If we have a **arg then it must be the last argument.
    if (foundKwargs) {
      return p.emitError(arg.loc, "'**' marker must be at end of ")
             << argOrParam << " list";
    }

    // If this argument is just a marker, process it.
    if (marker == KWArgMarkerInfo::kSlashSlash) {
      handleSlashSlashMarker(arg.loc);
      return success();
    }
    if (marker == KWArgMarkerInfo::kSlash) {
      handleSlashMarker(arg.loc);
      return success();
    }
    if (marker == KWArgMarkerInfo::kStar)
      return handleStarMarker(arg.loc, /*isMarker=*/true);

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
      if (failed(handleStarMarker(arg.loc, /*isMarker=*/false)))
        return failure();

    if (arg.vararg == VarArgKind::KWVarArg) {
      foundKwargs = true;

      if (kind == ArgListKind::kParamList ||
          kind == ArgListKind::kFnTypeParamList) {
        return p.emitError(arg.loc,
                           "variadic keyword parameters not supported yet");
      }
      if (arg.convention != ParsedArgument::kConventionUnspec &&
          arg.convention != ParsedArgument::kConventionOwned) {
        return p.emitError(
            arg.loc,
            "non-owned variadic keyword arguments are not supported yet");
      }
    }

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
        arg.kwArgHandling == KWArgHandling::kPositionalOnly ||
        arg.kwArgHandling == KWArgHandling::kInferred || arg.vararg)
      continue;
    if (!allUnnamedPosOnly)
      return p.emitError(arg.loc, "unnamed ")
             << argOrParam << " must be positional-only";
    arg.kwArgHandling = KWArgHandling::kPositionalOnly;
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

/// Helper to emit a default argument/parameter value. Variadic and pack
/// arguments/parameters get a placeholder default iff there are already
/// defaults in the given array of default (i.e. only if a variadic comes after
/// an optional argument/parameter).
static LogicalResult
emitDefaultIfPossible(const ParsedArgument &arg, ASTType type,
                      SmallVectorImpl<TypedAttr> &defaultPos,
                      SmallVectorImpl<TypedAttr> &defaultKwOnly,
                      ExprEmitter &emitter, ExprContext exprContext) {
  SmallVectorImpl<TypedAttr> &defaults =
      arg.kwArgHandling == KWArgHandling::kKeywordOnly ? defaultKwOnly
                                                       : defaultPos;
  auto emitDefaultIfPossible = [&]() -> PValue {
    if (const ExprNode *initExpr = arg.initExpr) {
      if (PValue value = emitter.emitExprPValue(initExpr, exprContext, type))
        return value;
      arg.isErroneous = true;
      return UnknownAttr::get(type);
    }

    // If we have a variadic argument, we add a placeholder default value so
    // that invariants about default values always correspond to the trailing
    // arguments. This allows us the have default values before a variadic.
    if (arg.vararg != VarArgKind::None && !defaults.empty())
      return UnknownAttr::get(mlir::NoneType::get(type.mlirType.getContext()));
    return {};
  };

  if (PValue value = emitDefaultIfPossible()) {
    defaults.push_back(value);
    return success();
  }
  return failure();
}

TypeCheckedParamList::TypeCheckedParamList(
    ArrayRef<ParsedArgument> parsedParams, ASTDecl &declScope,
    SharedState &shared)
    : TypeCheckScopeInfo{declScope, /*isParamContext=*/true, shared} {
  // Resolve each of the parameter declarations.
  ExprEmitter emitter(shared, declScope, EC_Type);
  for (auto [idx, arg] : llvm::enumerate(parsedParams)) {
    // Check for things supported in arguments that are not supported in
    // parameters.
    ASTType type;
    if (arg.typeExpr) {
      type = emitter.emitExprType(arg.typeExpr);
    } else {
      emitter.emitError(arg.loc, "parameters must always have a type");
      arg.isErroneous = true;
    }
    if (!type)
      type = emitter.shared.getTypeCheckErrorType();

    VarArgKind vararg = arg.vararg;
    if (vararg == VarArgKind::PackVarArg)
      emitter.emitError(arg.loc, "parameters may not be variadic packs");

    if (vararg == VarArgKind::VarArg && !type.isTypeCheckErrorType()) {
      // TODO: What convention should we use for parameter varargs?
      type = VariadicType::get(type, ArgConvention::BorrowedInReg);
      variadicIndices.push_back(idx);
    }

    if (failed(emitDefaultIfPossible(arg, type, defaultPosParams,
                                     defaultKwOnlyParams, emitter,
                                     EC_DefaultParam))) {
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
        if (arg.kgenConvention != ArgConvention::ByRefResult &&
            arg.kgenConvention != ArgConvention::ByRefError) {
          emitOptionalAfterRequired(emitter, arg, "parameter")
              << arg.typeExpr->getRange();
        }
      }
    }

    // TODO: Parameter decls should support conventions at some point.
    if (arg.convention != ParsedArgument::kConventionUnspec)
      emitter.emitError(arg.loc, "parameters must always be passed by-value");

    // Bind the parsed type expression so references from other parameters
    // can be resolved. The parameter names in ParamDeclAttr are mangled with
    // the location so that parameter names in mojo are unique in the IR.
    auto newDecl = ParamDeclAttr::get(
        declScope.mangleUserDefinedParamName(arg.name), type);
    paramDeclAttrs.push_back(newDecl);

    // The unmangled names are also collected to aid keyword parameter binding.
    passingKinds.push_back(arg.getKWArgHandlingAsPassingKind());
    names.push_back(arg.name);

    ASTDecl &resolvedDecl = emitter.getDeclResolver().addFullyResolvedDecl(
        PValue(ParamDeclRefAttr::get(newDecl)), arg.name, arg.loc, &declScope);
    emitter.shared.notifyListenerOnParameterDecl(resolvedDecl, arg.loc);
  }
}

PogListAttr TypeCheckedParamList::getParamListAttr() {
  return PogListAttr::get(
      shared.getContext(),
      PogListAttr::toPogs(names, passingKinds, variadicIndices),
      defaultPosParams, defaultKwOnlyParams);
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
static ASTType addImplicitTypeParams(ASTType type,
                                     TypeCheckedParamList &paramList) {
  // Check if the type has unbound parameters.
  auto metatype = dyn_cast_or_null<AnyStructType>(type.getMetaType());
  if (!metatype)
    return type;
  ArrayRef<Type> params = metatype.getSignature().getParamTypes();
  if (params.empty())
    return type;
  PogListAttr paramListAttr = metatype.getSignature().getParamListAttrs();

  SmallVector<TypedAttr> paramValues;
  ParserParamEvaluator evaluator(*paramList.shared.declResolver);
  for (auto [idx, type] : llvm::enumerate(params)) {
    auto funcDecl = ParamDeclAttr::get(paramList.declScope.mangleParamName(
                                           paramListAttr.getName(idx).strref()),
                                       evaluator.getReboundType(type));

    paramList.names.push_back(StringAttr::get(type.getContext()));
    paramList.passingKinds.push_back(PassingKind::Implicit);
    paramList.paramDeclAttrs.push_back(funcDecl);
    paramValues.push_back(ParamDeclRefAttr::get(funcDecl));
    evaluator.addInputValue(paramValues.back());
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
/// argument, and wraps the type with a RefType using that lifetime.
static RefType makeImplicitRefTypeForArg(const ParsedArgument &arg, size_t idx,
                                         Type type,
                                         TypeCheckedFnSignature &tcSignature) {
  ASTDecl &declScope = tcSignature.paramList.declScope;

  StringAttr lifetimeName;
  if (arg.name) {
    lifetimeName = declScope.mangleParamName(arg.name.strref());
  } else { // Used by function types, for example.
    lifetimeName =
        declScope.mangleParamName(Twine(llvm::utostr(idx)) + "_unnamed");
  }

  // The reference is immutable when borrowing, mutable otherwise.
  bool isMutable = arg.convention != ParsedArgument::kConventionBorrowed &&
                   arg.convention != ParsedArgument::kConventionUnspec;
  auto lifetimeDecl = ParamDeclAttr::get(
      lifetimeName, LifetimeType::get(lifetimeName.getContext(), isMutable));

  // Tell the signature about the new lifetime decl.
  tcSignature.implicitLifetimeDecls.push_back(lifetimeDecl);

  return RefType::get(
      type, ParamDeclRefAttr::get(lifetimeName, lifetimeDecl.getType()));
}

// If this argument is a pack vararg like "*args: *Ts" then the argument
// expression is "Ts", and the star before it was syntactically parsed.
// This expression must be a PValue of variadic metatype.  We need to
// process it into a VariadicPack.
static Type
typeCheckVariadicPackTypeSpecifier(ParsedArgument &arg, size_t argIdx,
                                   ExprEmitter &emitter,
                                   TypeCheckedFnSignature &tcSignature) {
  assert(arg.vararg == VarArgKind::PackVarArg &&
         "this applies to pack arguments");

  PValue param = emitter.emitExprPValue(arg.typeExpr, EC_Type);
  if (!param) // Error emitting the expression is already diagnosed.
    return {};

  // Make sure the param value is a variadic list of types.
  VariadicType paramVariadicType =
      dyn_cast<VariadicType>(param.getRValueType().mlirType);
  if (!paramVariadicType) {
    emitter.emitError(arg.typeExpr->getLoc(),
                      "pack argument type list must reference a variadic list")
        << arg.typeExpr->getRange();
    return {};
  }
  Type elementType = paramVariadicType.getElementType();
  if (!isa<AnyStructType, TypeType, TraitType>(elementType)) {
    emitter.emitError(arg.typeExpr->getLoc(),
                      "argument type list elements must be types")
        << arg.typeExpr->getRange();
    return {};
  }

  if (isa<TypeType>(elementType)) {
    emitter.emitError(arg.loc)
        << "variadic pack elements declared as 'AnyTrivialRegType' are removed,"
        << " please declare elements as 'AnyType' instead of "
           "'AnyTrivialRegType'";
    return {};
  }

  // Arguments passed by memory need an associated lifetime parameter, and need
  // to be passed by reference.
  RefType refType =
      makeImplicitRefTypeForArg(arg, argIdx, elementType, tcSignature);

  // Form a VariadicPack type.
  ASTType variadicPackType =
      emitter.shared.getBuiltinVariadicPackType(emitter.declScope, arg.loc);

  // Sanity check the returned VariadicPack declaration.
  if (isa<TypeCheckErrorType>(variadicPackType.mlirType))
    return {};
  auto packStruct = dyn_cast_if_present<StructDeclOp>(
      variadicPackType.getDecl(emitter.shared));
  if (!packStruct || packStruct.getParams().size() != 4 ||
      !packStruct.getParams()[0].getType().isInteger(1) ||
      !isa<LifetimeType>(packStruct.getParams()[1].getType()) ||
      !isa<AnyTraitType>(packStruct.getParams()[2].getType()) ||
      !isa<VariadicType>(packStruct.getParams()[3].getType())) {
    emitter.emitError(arg.loc, "malformed VariadicPack");
    return {};
  }
  // The default element_trait param type is
  // !lit.anytrait<<@stdlib::@builtin::@anytype::@AnyType>>
  // reflecting that it takes any trait like Stringable.
  auto elementTraitParamTy = packStruct.getParams()[2].getType();

  // If the declared type of the pack elements is a trait subtype of AnyType,
  // it will be that traits metatype.  Downcast to the same type, but with
  // !lit.anytrait<AnyType> type.
  TypedAttr elementTrait = PValue(elementType).get();
  if (elementTrait.getType() != elementTraitParamTy)
    elementTrait =
        ParamOperatorAttr::get(POC::Rebind, elementTrait, elementTraitParamTy);

  // Bind the VariadicPack[isMutable, lifetime, element_trait, element_types]
  // parameters.
  return packStruct.bindReference(
      {refType.isMutable(), refType.getLifetime(), elementTrait, param.get()});
}

/// Type check each argument in turn, resolving their type and default
/// initializer value.  Arguments in Mojo can refer to previous arguments in
/// their type+default value expressions as PValues, so we need to ensure that
/// they are emitted and have declarations registered in the scope so that later
/// lookups can find them.
static void typeCheckOneArgument(size_t idx, ASTType selfType, bool isDef,
                                 bool isStaticMethod, ASTDecl *fnDecl,
                                 TypeCheckedFnSignature &tcSignature) {
  ParsedArgument &arg = tcSignature.argList.parsedArgs[idx];

  ASTDecl &declScope = tcSignature.paramList.declScope;
  SharedState &shared = tcSignature.paramList.shared;
  ExprEmitter typeEmitter(shared, declScope, EC_Type);

  // Start by computing the declared type of the argument.
  ASTType type;
  if (arg.typeExpr) {
    if (arg.vararg != VarArgKind::PackVarArg) {
      // Emit the argument type. Allow argument types to be "automatically"
      // parameterized: if the type is fully unbound, its parameters are
      // appended to the function parameters.
      type = typeEmitter.emitExprType(arg.typeExpr, /*allowUnbound=*/true);
    } else {
      // Ts in "*args: *Ts" is a reference to a variadic list of types, but
      // needs to be typechecked.
      type = typeCheckVariadicPackTypeSpecifier(arg, idx, typeEmitter,
                                                tcSignature);
    }

    // If the type couldn't be emitted, mark this argument erroneous (so uses
    // within the body of the function don't trigger secondary errors) and
    // mark the function erroneous so calls to it won't resolve.  Put in a
    // placeholder type so we can continue type checking.
    if (!type) {
      type = shared.getTypeCheckErrorType();
      arg.isErroneous = true;
      arg.vararg = VarArgKind::None; // Don't break invariants on errors.
    }
    type = addImplicitTypeParams(type, tcSignature.paramList);
  } else if (idx == 0 && selfType &&
             // FIXME: This is incorrect, the @static_method decorators haven't
             // been applied yet.
             !isStaticMethod) {
    // If this is the 'self' argument in a struct, default the type to Self.
    type = selfType;
  } else if (isDef) {
    // In 'def', arguments with no types default to 'object'.
    type = shared.lookupObjectType(declScope, arg.loc);
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

  // If no convention was explicitly specified, default to 'borrowed'.
  if (arg.convention == ParsedArgument::kConventionUnspec) {
    // TODO: enable other conventions for **kwargs.
    arg.convention = arg.vararg == VarArgKind::KWVarArg
                         ? ParsedArgument::kConventionOwned
                         : ParsedArgument::kConventionBorrowed;
  }

  // Emit default argument values if present.
  if (failed(emitDefaultIfPossible(arg, type, tcSignature.defaultPosArgs,
                                   tcSignature.defaultKwOnlyArgs, typeEmitter,
                                   EC_DefaultArgument))) {
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
  case ParsedArgument::kConventionByRefResult:
    llvm_unreachable("shouldn't occur in an argument list");
  case ParsedArgument::kConventionOwned:
    if (!type.isRegisterPassable(arg.loc, shared) ||
        // VariadicListInMem supports owned, but VariadicList does not.
        arg.vararg == VarArgKind::VarArg) {
      arg.kgenConvention = ArgConvention::OwnedInMem;
      break;
    }
    arg.kgenConvention = ArgConvention::OwnedInReg;
    break;
  case ParsedArgument::kConventionRef:
    assert(arg.refLifetimeExpr && "No lifetime expr for convention!");
    if (arg.vararg != VarArgKind::None) {
      // There should be no reason this isn't supportable.
      shared.emitError(
          arg.loc, "TODO: variadics not supported with 'ref' convention yet");
      arg.vararg = VarArgKind::None;
    }
    type = processLifetimeSpecifier(arg.refLifetimeExpr, type, arg.name,
                                    tcSignature.paramList, /*isResult=*/false);
    arg.kgenConvention = ArgConvention::Ref;

    if (isa<TypeCheckErrorType>(type.getReferenceElementType()))
      arg.isErroneous = true;
    break;
  case ParsedArgument::kConventionBorrowed: {
    arg.kgenConvention = ArgConvention::BorrowedInMem;
    TypeConvention conv = type.getRegisterPassability(arg.loc, shared);
    // FIXME(MOCO-725): Borrows of non-trivial register-passable values don't
    // have lifetimes and can't be correctly tracked if captured in an async
    // function. Emit an error to avoid a footgun.
    if (arg.vararg != VarArgKind::PackVarArg &&
        conv == TypeConvention::RegisterPassable &&
        tcSignature.argList.effects.isAsync()) {
      shared.emitError(arg.loc,
                       "TODO: borrowed non-trivial register-passable arguments "
                       "are not yet supported in async functions");
    }
    // We can pass the borrowed argument in a register if it is register
    // passable, but variadics have more details.
    if (conv != TypeConvention::MemoryOnly &&
        // We MUST pass non-trivial register types with VariadicListMem,
        // but can't quite use it for all borrowed arguments yet.
        // TODO(MOCO-726): Make variadics always pass through memory.
        (arg.vararg != VarArgKind::VarArg ||
         conv == TypeConvention::RegisterPassableTrivial))
      arg.kgenConvention = ArgConvention::BorrowedInReg;
    break;
  }
  case ParsedArgument::kConventionInOut:
    arg.kgenConvention = ArgConvention::InOut;
    break;
  case ParsedArgument::kConventionInitSelfResult:
    arg.kgenConvention = ArgConvention::InitSelf;
    break;
  }

  // For packs, we figure out the declared arg convention and adjust passed
  // convention.
  if (arg.vararg == VarArgKind::PackVarArg) {
    // Remember the original declared convention, forcing to memory convention.
    // The VariadicPack itself is passed as borrowed except for owned
    // convention: this allows the callee to consume the pack.
    switch (arg.convention) {
    case ParsedArgument::kConventionRef:
    case ParsedArgument::kConventionUnspec:
    case ParsedArgument::kConventionByRefResult:
    case ParsedArgument::kConventionInitSelfResult:
      llvm_unreachable("not a pack arg convention");
    case ParsedArgument::kConventionOwned:
      arg.kgenVariadicConvention = ArgConvention::OwnedInMem;
      arg.kgenConvention = ArgConvention::OwnedInReg;
      break;
    case ParsedArgument::kConventionBorrowed:
      arg.kgenVariadicConvention = ArgConvention::BorrowedInMem;
      arg.kgenConvention = ArgConvention::BorrowedInReg;
      break;
    case ParsedArgument::kConventionInOut:
      arg.kgenVariadicConvention = ArgConvention::InOut;
      arg.kgenConvention = ArgConvention::BorrowedInReg;
      break;
    }
  }

  // Values passed by memory need an associated lifetime parameter, and need to
  // be passed by reference. For now, we don't use reference types in **kwargs.
  Type fullType;
  if (SignatureType::hasImplicitLifetime(arg.kgenConvention) &&
      arg.vararg != VarArgKind::KWVarArg) {
    fullType = makeImplicitRefTypeForArg(arg, idx, type, tcSignature);
  } else {
    fullType = type;
  }

  // If this is a valid vararg argument, then we pass it as a variadic type.
  // The convention is to pass as a register value, in the case of a memory
  // value, we're passing the array of pointers by value.
  if (arg.vararg == VarArgKind::VarArg) {
    fullType = VariadicType::get(fullType, arg.kgenConvention);
    arg.kgenConvention = ArgConvention::BorrowedInReg;
  } else if (arg.vararg == VarArgKind::KWVarArg) {
    // We build OwnedKwargsDict[ValType].
    ASTType dictType = shared.getOwnedKwargsDictType(arg.loc);

    auto dictDecl = cast<LIT::StructType>(dictType.mlirType);
    auto dictMetatype = cast<AnyStructType>(dictDecl.getMetaType());
    ArrayRef<Type> inputTypes =
        dictMetatype.getSignature().getInputParamTypes();
    if (inputTypes.size() != 1) {
      shared.emitError(arg.loc)
          << "internal compiler error: OwnedKwargsDict type has unexpected "
             "parameter signature; please file a bug";
      arg.isErroneous = true;
    }

    // If anything is wrong with the argument, we terminate before emitting a
    // type for the variadic keyword arguments.
    if (arg.isErroneous)
      return;

    auto collectionElement = cast<TraitType>(inputTypes[0]);
    PValue binding;
    if (!arg.typeExpr) {
      assert(isDef);
      // If we're in a `def` function without an explicit argument type, we need
      // a synthetic type expression.
      SyntheticNode typeExpr(arg.loc);
      binding = typeEmitter.emitPValue({fullType, typeExpr}, EC_Type,
                                       collectionElement);
    } else {
      binding = typeEmitter.emitPValue({fullType, arg.typeExpr}, EC_Type,
                                       collectionElement);
    }
    if (!binding) {
      shared.emitError(arg.loc)
          << "argument type must conform to 'CollectionElement' to be used in "
             "a keyword variadic argument";
      arg.isErroneous = true;
      return;
    }
    fullType =
        ParamRefType::get(BindTypeAttr::get(PValue(dictType), binding.get()));

    // OwnedKwargsDict is memory only and since only the callee can access it,
    // we pass it as owned.
    arg.kgenConvention = ArgConvention::OwnedInMem;
    fullType = makeImplicitRefTypeForArg(arg, idx, fullType, tcSignature);
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
  // its entry block. Otherwise it is a function type: We allocate the argument
  // into a holding block owned by SharedState so it isn't leaked.
  Block &blockOwningArg = fnDecl ? *cast<LIT::FuncOp>(fnDecl).getBody()
                                 : shared.getArgumentOwningBlock();
  BlockArgument blockArg =
      blockOwningArg.addArgument(fullType, shared.translateLocation(arg.loc));

  DeclIRValue argIRValue;
  switch (arg.kgenConvention) {
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError:
    llvm_unreachable("should never need to handle result slots");
  case ArgConvention::InOut:
  case ArgConvention::InitSelf:
  case ArgConvention::OwnedInMem:
  case ArgConvention::Ref:
  case ArgConvention::BorrowedInMem:
    // TODO: Collapse MLValue and MBValue.
    if (cast<RefType>(fullType).isMutableKnown(true))
      argIRValue = MLValue(blockArg);
    else
      argIRValue = MBValue(blockArg);
    break;
  case ArgConvention::OwnedInReg:
    // NOTE: This will get wrapped and turned into an SLValue within the body.
    argIRValue = SRValue(blockArg);
    break;
  case ArgConvention::BorrowedInReg:
    argIRValue = SBValue(blockArg);
    break;
  }

  // FIXME: This is not setting the correct type for Variadics.  We shouldn't
  // expose something like !kgen.variadic to subsequent arguments, we should
  // expose VariadicListMem.  This will require moving the VariadicList
  // formation to the caller side.
  ASTDecl &decl = typeEmitter.getDeclResolver().addFullyResolvedDecl(
      argIRValue, arg.name, arg.loc, &typeEmitter.declScope);

  // If we don't have a function decl, notify the listener immediately (function
  // arguments will be notified when they are fully resolved later).
  if (!fnDecl)
    shared.notifyListenerOnArgumentDecl(decl, arg.name, arg.loc);
}

/// Type check the result type for the function.  `resultTypeExpr` will be
/// non-null if explicitly specified in source code, and the `resultLoc` will
/// always be valid point for end of the argument list.
static void typeCheckResult(const ExprNode *resultTypeExpr,
                            const ExprNode *resultRefLifetimeExpr,
                            SMLoc resultLoc, bool isDef,
                            const SpecialFunctionInfo &fnInfo, ASTDecl *fnDecl,
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
      resultType = shared.lookupObjectType(declScope, resultLoc);
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

  // If a result lifetime is specified with `ref [life] Ty`, then form a ref
  // result.
  if (resultRefLifetimeExpr) {
    if (tcSignature.argList.effects.isAsync()) {
      // TODO(MOCO-787): Async functions don't support ref results yet. We need
      // to define a `CoroutineRef` or support perfect forwarding in generic
      // results.
      shared.emitError(resultRefLifetimeExpr->getLoc())
          << "TODO: ref results aren't supported in async functions yet";
      resultRefLifetimeExpr = nullptr;
    } else {
      resultType = processLifetimeSpecifier(
          resultRefLifetimeExpr, resultType,
          // TODO: Use the name of the return slot if present.
          "__result__", tcSignature.paramList, /*isResult*/ true);
      tcSignature.argList.effects.setRefResult(isa<RefType>(resultType));
    }
  }

  // Remember the user-declared result type.
  tcSignature.resultType = resultType;

  // Now that we have the user's result type, compute the full type of the
  // result, which can can be different when memory only, when throwing, etc.
  ASTType fullResultType = resultType;
  TypeConvention rp = resultType.getRegisterPassability(resultLoc, shared);

  // If this function throws, add a result slot for the error that may be
  // raised.
  if (tcSignature.argList.effects.isThrows()) {
    ASTType errorType =
        shared.getBuiltinErrorType(tcSignature.paramList.declScope, resultLoc);

    // Synthesize a ByRefError argument for the error.
    ParsedArgument errArg;
    errArg.loc = resultLoc;
    errArg.name = StringAttr::get(shared.getContext(), "__error__");
    errArg.convention = ParsedArgument::kConventionByRefResult;
    errArg.kgenConvention = ArgConvention::ByRefError;
    errArg.kwArgHandling = KWArgHandling::kKeywordOnly;
    errArg.typeExpr = nullptr;
    tcSignature.argList.parsedArgs.push_back(errArg);
    tcSignature.argTypes.push_back(errorType);

    RefType refType =
        makeImplicitRefTypeForArg(errArg, 0, errorType, tcSignature);
    tcSignature.fullArgTypes.push_back(refType);

    // If this is for a lit.func declaration (as opposed to a function type),
    // add a block argument for this.
    if (fnDecl) {
      Block &body = *cast<LIT::FuncOp>(fnDecl).getBody();
      (void)body.addArgument(refType, shared.translateLocation(resultLoc));
    }

    // The ABI result type is an i1 indicating the error state.
    fullResultType = Builder(shared.getContext()).getI1Type();
    // The result value is always returned through memory. Initializers don't
    // have formal results.
    if (!fnInfo.isInitializer())
      rp = TypeConvention::MemoryOnly;
  }

  // Async functions always use in-memory results.
  if (tcSignature.argList.effects.isAsync())
    rp = TypeConvention::MemoryOnly;

  // If it is memory-only, pass it indirectly as the last argument to the
  // function by-reference.
  if (rp == TypeConvention::MemoryOnly) {
    // Synthesize a ByRefResult argument for the result.
    ParsedArgument resultArg;
    resultArg.loc = resultLoc;
    resultArg.name = StringAttr::get(shared.getContext(), "__result__");
    resultArg.convention = ParsedArgument::kConventionByRefResult;
    resultArg.kgenConvention = ArgConvention::ByRefResult;
    resultArg.kwArgHandling = KWArgHandling::kKeywordOnly;
    resultArg.typeExpr = resultTypeExpr;
    tcSignature.argList.parsedArgs.push_back(resultArg);
    tcSignature.argTypes.push_back(resultType);

    // Compute the RefType for this new argument with an implicit lifetime.
    RefType refType =
        makeImplicitRefTypeForArg(resultArg, 0, resultType, tcSignature);
    tcSignature.fullArgTypes.push_back(refType);

    // If this is for a lit.func declaration (as opposed to a function type),
    // add a block argument for this.  We don't register this for name lookup
    // though, we don't want it to conflict with user identifiers, and it is
    // never looked up directly.
    if (fnDecl) {
      Block &body = *cast<LIT::FuncOp>(fnDecl).getBody();
      (void)body.addArgument(refType, shared.translateLocation(resultLoc));
    }

    // We know the ABI register result will be None now, which is trivial.
    if (!tcSignature.argList.effects.isThrows())
      fullResultType = shared.getNoneType();
  }

  tcSignature.fullResultType = fullResultType;
}

/// Emit the argument types, default values, and result type and determine
/// the argument conventions.
///
/// 'fnDecl' will be null when this is a function type, which doesn't have a
/// declaration.
TypeCheckedFnSignature::TypeCheckedFnSignature(
    TypeCheckedParamList &paramList, ParsedArgumentList &argList,
    const ExprNode *resultTypeExpr, const ExprNode *resultRefLifetimeExpr,
    SMLoc resultLoc, bool isDef, ASTDecl *fnDecl, SpecialFunctionInfo &fnInfo)
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
      selfType = parent->getTypeDeclSelf();
    }
  }

  // If this is a well-known function like `__init__`, perform early semantic
  // checks and clarify what special function it really is.
  // This logic happens before type checking, so we need to be very careful
  // to only process it if defined correctly.  We let downstream checks diagnose
  // the errors.
  if (fnInfo.isInitializer() && selfType) {
    if (!argList.parsedArgs.empty() &&
        argList.parsedArgs[0].convention == ParsedArgument::kConventionInOut) {
      // The self argument is actually passed with a special convention. It is
      // written inout, but it isn't really.
      // TODO: Introduce an 'init' convention, maybe even an 'init' keyword.
      auto &selfArg = argList.parsedArgs[0];
      selfArg.convention = ParsedArgument::kConventionInitSelfResult;
      // We also force the passing kind of self to positional-only.
      if (selfArg.kwArgHandling == KWArgHandling::kPositionalOrKeyword)
        selfArg.kwArgHandling = KWArgHandling::kPositionalOnly;
    }

    // @register_passable values are movable by passing the register around, so
    // they can't define a moveinit.
    if (fnInfo.kind == SpecialFunctionKind::kMoveInit &&
        selfType.isRegisterPassable(fnDecl->getLoc(), shared)) {
      fnDecl->setErroneous();
      shared.emitError(fnDecl->getLoc(), "'")
          << fnInfo.name
          << "' is not supported for @register_passable types, they "
             "are always movable by copying a register";
      fnInfo = SpecialFunctionInfo();
    }
  }

  // __new__ is implicitly static.
  if (fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod)
    cast<LIT::FuncOp>(fnDecl).setIsStatic(true);

  // TODO(MOCO-789): Async initializers require a `byref_result` thunk to be
  // emitted. Just forbid them for now.
  if (fnInfo.isInitializer() && argList.effects.isAsync()) {
    shared.emitError(fnDecl->getLoc())
        << "TODO: async constructors are not yet supported";
    argList.effects.setAsync(false);
  }

  // True if this is a static method.
  // FIXME: This is completely wrong, @static_method decorator hasn't been
  // applied yet.
  bool isStaticMethod = selfType && cast<LIT::FuncOp>(fnDecl).getIsStatic();

  // Resolve all argument types, generating type check error types for any types
  // that could not be correctly resolved.
  for (size_t i = 0, e = argList.parsedArgs.size(); i != e; ++i)
    typeCheckOneArgument(i, selfType, isDef, isStaticMethod, fnDecl, *this);

  // Compute the result type.
  typeCheckResult(resultTypeExpr, resultRefLifetimeExpr, resultLoc, isDef,
                  fnInfo, fnDecl, *this);
}

FunctionType TypeCheckedFnSignature::getFunctionType() const {
  return FunctionType::get(fullResultType.mlirType.getContext(), fullArgTypes,
                           {fullResultType.mlirType});
}

/// Form a LIT signature packaging up all the stuff we need to know about this
/// type checked function.
LITSignatureType TypeCheckedFnSignature::getLITSignatureType() const {
  MLIRContext *ctx = paramList.shared.getContext();

  size_t numArgs = argList.parsedArgs.size();
  SmallVector<PogMetadataAttr> argPogs;
  argPogs.reserve(numArgs);
  SmallVector<ArgConvention> argConventions;
  argConventions.reserve(numArgs);

  ssize_t argPackIndex = -1;
  std::optional<ArgConvention> argPackOrigConvention;
  for (auto [idx, arg] : llvm::enumerate(argList.parsedArgs)) {
    bool isVariadic =
        arg.vararg == VarArgKind::VarArg || arg.vararg == VarArgKind::KWVarArg;
    argPogs.emplace_back(PogMetadataAttr::get(
        arg.name, arg.getKWArgHandlingAsPassingKind(), isVariadic));
    argConventions.push_back(arg.kgenConvention);
    if (arg.vararg == VarArgKind::PackVarArg) {
      assert(argPackIndex == -1 && "only one argument pack is possible");
      argPackIndex = idx;
      argPackOrigConvention = arg.kgenVariadicConvention;
    }
  }

  auto metadata = FnMetadataAttr::get(
      PogListAttr::get(ctx, argPogs, defaultPosArgs, defaultKwOnlyArgs,
                       argPackIndex, argPackOrigConvention),
      paramList.getParamListAttr(), implicitLifetimeDecls.size());

  /// Silence internal verifier errors when constructing types from the parser.
  /// We don't want to show these to the user.
  auto silenceErrors = [ctx] {
    InFlightDiagnostic diag = mlir::emitError(UnknownLoc::get(ctx));
    diag.abandon();
    return diag;
  };

  FunctionType functionType = getFunctionType();
  return SignatureType::remapToSignature(
      paramList.paramDeclAttrs, /*resultParams=*/{}, functionType,
      argConventions, argList.effects, metadata, silenceErrors);
}
