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
#include "CallEmission.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"
#include "MojoUtils.h"
#include "ParserBase.h"

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

// Given a value of a well known type, extract the specified field.  This
// returns null if the field doesn't exist.
static TypedAttr digOutSingleField(TypedAttr value, StringRef fieldName,
                                   SMLoc loc, SharedState &shared) {
  // TODO: This is generating a StructExtractAttr the hard way.  It would be
  // way nicer to form a call to `o.__mlir_origin__()` or something like we do
  // for bools, but unfortunately that won't get inlined and simplified by the
  // call emission because Origin is parametric.  Therefore it will break
  // parameter inference.
  ASTDecl *typeDecl = ASTType(value.getType()).getDecl(shared);
  if (!typeDecl || !isa<LIT::StructType>(value.getType()))
    return {};

  // Check to see if it has the expected field of Origin.
  LookupResult lookup =
      shared.lookupAndResolveDecl(fieldName, loc, *typeDecl,
                                  /*searchParentScopes=*/false);
  if (lookup.getIfSuccess().size() != 1)
    return {};
  auto fieldOp = dyn_cast<StructFieldOp>(lookup.getIfSuccess()[0]);
  if (!fieldOp)
    return {};

  return LIT::StructExtractAttr::get(value, fieldOp);
};

/// Given a parameter that is a !lit.origin or an Origin, return the
/// underlying !lit.origin.  This returns null on failure.
TypedAttr ASTType::extractOriginOf(SMLoc loc, TypedAttr value,
                                   SharedState &shared) {
  // A raw !lit.origin always works.
  if (isa<OriginType>(value.getType()))
    return value;

  // If this is a value of Origin type, process it.
  if (auto extractVal =
          digOutSingleField(value, ORIGIN_FIELD_NAME, loc, shared))
    if (isa<OriginType>(extractVal.getType()))
      return extractVal;
  return {};
}

/// Given an expression that can be used in __origin_of or a ref expression,
/// analyze it to determine which origin it represents.  If it doesn't work,
/// emit an error and return null.
TypedAttr ExprEmitter::extractOriginOf(const ExprNode *expr, CValue value) {
  // If this is a DLValue, it may be a def argument with an unresolved box.  We
  // could materialize the box, but Python doesn't have origin_of, so we aren't
  // in a compatibility situation: just collapse to immut ref.
  if (auto dlVal = value.getIfDLValue())
    if (MBValue resolved = dlVal->emitMBValueFromDefArgument(*this))
      value = resolved;

  if (value.isMValue()) // We can get the origin of an MValue.
    return value.getMValueType().getOrigin();

  // Check for !lit.origin and Origin struct.
  if (auto pv = value.getIfPValue()) {
    if (TypedAttr result =
            ASTType::extractOriginOf(expr->getLoc(), pv.get(), shared))
      return result;
  }

  emitError(expr->getLoc())
      << "value of type " << value.getRValueType()
      << " doesn't have a memory origin" << expr->getRange();
  return {};
}

/// Process the origin expression in a `ref [...] T` reference specifier.
/// T is specified as 'type' and this returns the result !lit.ref type.
static RefType processRefOriginSpecifier(const ExprNode *origExpr, ASTType type,
                                         StringRef valueName,
                                         TypeCheckedParamList &paramList,
                                         bool isResult) {
  SharedState &shared = paramList.shared;

  // For errors, return "RefType(TypeCheckErrorType)" to maintain the invariant
  // that all "ref" values have RefType, but their RValue type is an error.
  auto hadError = [&]() -> RefType {
    return RefType::getAnyOrigin(shared.getTypeCheckErrorType(),
                                 /*isMut*/ true);
  };

  // Propagate already diagnosed errors.
  if (isa<TypeCheckErrorType>(type))
    return hadError();

  ExprEmitter emitter(paramList.declScope, EC_Origin);

  // Check to see if this is a value address space specifier.  If so, return
  // true, otherwise return false.
  auto digOutAddressSpace = [&](TypedAttr value, SMLoc loc) -> TypedAttr {
    // If the value has index type, then it is good to go.
    if (value.getType().isIndex())
      return value;

    // Check to see if this is the well-known AddressSpace struct.  If so,
    // dig out the index from within it.
    auto extractInt = digOutSingleField(value, "_value", loc, shared);
    if (!extractInt)
      return {};
    auto extractIndex = digOutSingleField(extractInt, "value", loc, shared);
    if (!extractIndex)
      return {};
    if (extractIndex.getType().isIndex())
      return extractIndex;
    return {};
  };

  // If the origin expression is syntactically a multi-element tuple, then
  // take it apart.
  ArrayRef<const ExprNode *> originExprElts;
  if (auto *tuple = dyn_cast_if_present<TupleNode>(origExpr))
    originExprElts = tuple->exprs;
  else if (origExpr)
    originExprElts = origExpr;

  // Emit the origin expression if it is a normal expression.
  TypedAttr origin;
  TypedAttr addrSpace;
  for (const ExprNode *expr : originExprElts) {
    // Ignore _'s.
    if (expr->kind == ExprNode::kDiscardLiteral)
      continue;

    // The origin expression may be any of:
    //   1) an MValue, which we take the origin from.
    //   2) a value of !lit.origin or Origin[Mut] type.
    //  In the former case, we want to evaluate the expression without
    // evaluating it, because it may involve complex nested expressions and we
    // may be in a PValue expression.
    TypedAttr thisOrigin;
    emitter.emitExpressionWithOutEvaluatingIt(
        expr, EC_Origin, [&](CValue result) {
          // Check to see if it is an address space first.
          if (auto pv = result.getIfPValue()) {
            if (auto as = digOutAddressSpace(pv.get(), expr->getLoc())) {
              if (addrSpace) {
                emitter.emitError(expr->getLoc())
                    << "multiple specification of address space isn't valid"
                    << expr->getRange();
              }
              addrSpace = as;
              return;
            }
          }
          // Otherwise it must be a !lit.origin and Origin struct.
          thisOrigin = emitter.extractOriginOf(expr, result);
        });

    // If we found an origin, add it to our set.
    if (!thisOrigin)
      continue;
    if (!origin)
      origin = thisOrigin;
    else
      origin = OriginUnionAttr::get(origin.getContext(), {origin, thisOrigin});
  }

  // If no origin is specified, then it is inferred from the callsite. Add two
  // parameters to this function: one for the mutability of type Bool and one
  // for the origin.
  if (!origin) {
    auto addParam = [&](const Twine &name, Type type) -> TypedAttr {
      auto paramDecl =
          ParamDeclAttr::get(paramList.declScope.mangleParamName(name), type);
      paramList.names.push_back(StringAttr::get(type.getContext()));
      paramList.passingKinds.push_back(PassingKind::Implicit);
      paramList.paramDeclAttrs.push_back(paramDecl);
      return ParamDeclRefAttr::get(paramDecl);
    };

    if (isResult) {
      emitter.emitError(origExpr->getLoc())
          << "cannot infer origin for a function result"
          << origExpr->getRange();
      return hadError();
    }

    auto isMut = addParam(valueName + "_is_mut",
                          IntegerType::get(shared.getContext(), 1));
    origin = addParam(valueName + "_is_origin", OriginType::get(isMut));
  }
  if (!origin)
    return hadError();

  if (!isa<OriginType>(origin.getType())) {
    emitter.emitError(origExpr->getLoc())
        << "result reference origin has unexpected type " << origin.getType()
        << origExpr->getRange();
    return hadError();
  }

  if (!addrSpace)
    addrSpace = IntegerAttr::get(IndexType::get(shared.getContext()), 0);

  return RefType::get(type, origin, addrSpace);
}

//===----------------------------------------------------------------------===//
// Argument and Parameter List Parsing
//===----------------------------------------------------------------------===//

ParseResult ParsedArgument::parse(ParserBase &p, KWArgMarkerInfo &markerInfo,
                                  ArgListKind kind) {
  loc = p.getToken().getLoc();
  cursor = p.getLexer().getCursor();

  auto handleContextualArgConvention = [&](StringRef str,
                                           PAArgConvention conv) {
    // Handle "out: Foo" as a name, not an argument convention.
    if (p.getToken().isNot(Token::colon, Token::equal, Token::r_paren,
                           Token::r_square)) {
      convention = conv;
    } else {
      // Otherwise, the "out" is the argument name.
      name = StringAttr::get(p.getContext(), str);
    }
  };

  // Any owned/read/mut/ref keyword sets convention.
  if (p.consumeIf(Token::kw_owned))
    convention = kConventionOwned;
  else if (p.getToken().is(Token::kw_ref)) {
    (void)p.parseRefSpecifier(refOriginExpr, /*isOriginRequired*/ false);
    convention = kConventionRef;
  } else if (p.consumeIfSoftIdentifier("out")) {
    handleContextualArgConvention("out", kConventionOut);
  } else if (p.consumeIfSoftIdentifier("mut")) {
    handleContextualArgConvention("mut", kConventionMut);
  } else if (p.consumeIfSoftIdentifier("read")) {
    handleContextualArgConvention("read", kConventionRead);
  }

  while (p.getToken().isAny(Token::kw_owned, Token::kw_ref)) {
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

  // Reject attempts to make variadic output arguments.
  if (vararg != VarArgKind::None && convention == kConventionOut) {
    p.emitError(loc, "'out' convention may not be variadic");
    isErroneous = true;
    vararg = VarArgKind::None;
  }

  // Parse the argument name if present.
  if (name) {
    // If we already parsed a name due to lookahead, then we are done.
  } else if (kind == ArgListKind::kFnTypeArgList ||
             kind == ArgListKind::kFnTypeParamList) {
    // When parsing a function type, the name is optional.
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

    if (convention == kConventionMut || convention == kConventionOut) {
      p.emitError(equalLoc)
          << (convention == kConventionOut ? "'out'" : "'mut'")
          << " arguments may not have defaults" << initExpr->getRange();
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
  if (isResultSlot(kgenConvention))
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
/// 'resultArg' is non-null for argument lists, and allows handling of 'out'
/// arguments.
static ParseResult
parseArgOrParamList(ParserBase &p, SmallVectorImpl<ParsedArgument> &parsedArgs,
                    ParsedArgument *resultArg, ArgListKind kind) {
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

    // If this argument is an "out" argument, process it as a result.
    if (arg.convention == ParsedArgument::kConventionOut) {
      if (!resultArg)
        return p.emitError(arg.loc, "parameters cannot be 'out'");
      if (resultArg->convention == ParsedArgument::kConventionOut)
        return p.emitError(arg.loc,
                           "function may not have multiple 'out' arguments");
      *resultArg = arg;
      return success();
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

/// Given a type that potentially has all of its parameters unbound, implicitly
/// add the parameter declarations to the function parameters. For example, a
/// struct type can be partially bound. This function implicitly adds a
/// parameter declaration to the function for each unbound struct parameter and
/// binds the struct type to reference those parameters.
///
/// For function types, if the capture origin set parameter is unbound, an
/// implicit parameter for it is added, and a function type of the capture
/// origin set parameter bound to it is returned.
///
/// Parameters can be either added to the end of the parameter as `Implicit`
/// passing-kind parameters if `append` is set (this is used for unbound
/// arguments), or added to the beginning of the parameter list as `Inferred`
/// passing-kind parameters (this is used for unbound parameters).
static ASTType addImplicitTypeParams(ASTType type,
                                     TypeCheckedParamList &paramList,
                                     bool append) {
  SmallVector<ParamDeclAttr> paramDeclAttrs;
  SmallVector<StringAttr> names;
  SmallVector<PassingKind> passingKinds;
  // Functor to insert the pending vectors into paramList, either at the front
  // or back.
  auto insertFn = [append](auto &dst, auto &src) {
    dst.insert(append ? dst.end() : dst.begin(), src.begin(), src.end());
  };
  auto commitChanges = llvm::make_scope_exit([&]() {
    // All lists guaranteed to have the same length.
    if (paramDeclAttrs.empty())
      return;
    insertFn(paramList.paramDeclAttrs, paramDeclAttrs);
    insertFn(paramList.names, names);
    insertFn(paramList.passingKinds, passingKinds);

    // If we're inserting a parameter at the start, then any variadics before it
    // will need to be updated to account for the new parameter, e.g.:
    //     fn test[*elts: Int, autoparm: StructWithParam]():
    if (!append) {
      for (auto &idx : paramList.variadicIndices)
        idx += paramDeclAttrs.size();
    }
  });

  // The parameter decl references that will be used to fully bind the type,
  // plus a parameter evaluator we use to progressively refine the type.
  SmallVector<TypedAttr> paramValues;
  ParameterEvaluator evaluator;

  // This functor adds a single parameter to the parameter list.
  auto declareAndAddParam = [&](Type type, StringRef name) {
    auto funcDecl =
        ParamDeclAttr::get(paramList.declScope.mangleParamName(name),
                           evaluator.getReboundType(type));
    names.push_back(StringAttr::get(type.getContext()));
    passingKinds.push_back(append ? PassingKind::Implicit
                                  : PassingKind::Inferred);
    paramDeclAttrs.push_back(funcDecl);
    paramValues.push_back(ParamDeclRefAttr::get(funcDecl));
    evaluator.addInputValue(paramValues.back());
  };

  // First check for a function type.
  // FIXME: We need an AnyFunction metatype.
  if (auto sig = dyn_cast<FnTypeGeneratorType>(type)) {
    TypedAttr origins = sig.getCaptureOrigins();
    if (!isa<UnboundAttr>(origins))
      return type;
    declareAndAddParam(origins.getType(), "__origins__");
    return sig.getWithCaptureOrigins(paramValues.back());
  }

  // Check for a struct type or a struct metatype.
  auto getBoundStructMetaType = [&](StructMetaType metatype) {
    // The unbound parameters will be on the struct type's signature.
    TypeSignatureType sig = metatype.getSignature();
    for (auto [idx, type] : llvm::enumerate(sig.getParamTypes()))
      declareAndAddParam(type, sig.getParamListAttrs().getName(idx));
    return metatype.bindUnbound(paramValues);
  };

  if (auto metatype = dyn_cast_or_null<StructMetaType>(type.getMetaType()))
    return getBoundStructMetaType(metatype).getType();
  if (auto metatype = dyn_cast_or_null<StructMetaType>(type))
    return getBoundStructMetaType(metatype);

  return type;
}

TypeCheckedParamList::TypeCheckedParamList(
    ArrayRef<ParsedArgument> parsedParams, ASTDecl &declScope)
    : declScope(declScope), shared(declScope.getShared()) {
  // Resolve each of the parameter declarations.
  ExprEmitter emitter(declScope, EC_Type);
  for (const ParsedArgument &arg : parsedParams) {
    // Check for things supported in arguments that are not supported in
    // parameters.
    ASTType type;
    if (arg.typeExpr) {
      type = emitter.emitExprType(arg.typeExpr, /*allowUnbound=*/true);
      type = addImplicitTypeParams(type, *this, /*append=*/false);
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
      type = VariadicType::get(type, ArgConvention::ReadReg);
      // We add the indices of all parameters to be marked as varargs. Use the
      // current number of elements in `names`, because it also includes
      // implicitly added autoparams.
      variadicIndices.push_back(names.size());
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
ParseResult ParsedParamList::parseParametersIfPresent(ParserBase &p,
                                                      ArgListKind kind) {
  // Check to see if a parameter signature exists at all.
  if (!p.consumeIf(Token::l_square) || p.consumeIf(Token::r_square))
    return success();

  // Parse an actual parameter list.
  if (parseArgOrParamList(p, params, /*resultArg=*/nullptr, kind))
    return failure();

  return p.parseToken(Token::r_square, "expected ']' for parameter list");
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
    return parseArgOrParamList(p, parsedArgs, &resultArg, kind);

  if (p.parseToken(Token::l_paren, "expected '(' for argument list"))
    return failure();

  if (!p.consumeIf(Token::r_paren)) {
    if (parseArgOrParamList(p, parsedArgs, &resultArg, kind) ||
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

/// Parse the result specifier starting with a `->` if present.
void ParsedArgumentList::parseResultIfPresent(
    ParserBase &p, std::optional<size_t> stmtIndent) {
  SMLoc arrowLoc;
  if (!p.consumeIf(Token::minus_greater, &arrowLoc)) {
    // Make sure the result arg has a location of the end of the argument if not
    // specified by an 'out' argument, so that synthesized results (none etc)
    // have a location.
    if (!resultArg.loc.isValid())
      resultArg.loc = p.getToken().getLoc();
    return;
  }

  // We may have already parsed an 'out' argument.  If so, this will be an error
  // and we may want to undo things.
  auto oldResultArg = resultArg;
  resultArg.loc = p.getToken().getLoc();

  // Parse a result reference if present.
  if (p.getToken().is(Token::kw_ref)) {
    (void)p.parseRefSpecifier(resultArg.refOriginExpr,
                              /*originRequired*/ true);
  }

  // Parse the result type expression.
  // If this result parsing fails, then we just continue on as if none was
  // specified.
  (void)p.parseExpression(resultArg.typeExpr, stmtIndent);

  // If we already had a result, emit an error but keep parsing.
  if (resultArg.convention == ParsedArgument::kConventionOut) {
    auto diag = p.emitError(resultArg.loc)
                << "function cannot have both an 'out' argument "
                   "and an explicit result type";
    // It is common to include -> None on initializers, provide a helpful
    // message.
    if (resultArg.typeExpr &&
        resultArg.typeExpr->kind == ExprNode::kNoneLiteral) {
      diag << "; remove the '-> None' to fix it";
      diag.addFixIt(FixIt::remove(
          SourceRange(arrowLoc, resultArg.typeExpr->getRangeEnd())));
      resultArg = oldResultArg;
    }
  }

  // Indicate a present result by setting its convention to 'out'.
  resultArg.convention = ParsedArgument::kConventionOut;
}

/// This function creates a new anonymous origin decl for the specified
/// argument, and wraps the type with a RefType using that origin.
static RefType makeImplicitRefTypeForArg(const ParsedArgument &arg, size_t idx,
                                         Type type, bool isMutable,
                                         TypeCheckedFnSignature &tcSignature) {
  ASTDecl &declScope = tcSignature.paramList.declScope;

  StringAttr originName;
  if (arg.name) {
    originName = declScope.mangleParamName(arg.name.strref());
  } else { // Used by function types, for example.
    originName =
        declScope.mangleParamName(Twine(llvm::utostr(idx)) + "_unnamed");
  }

  auto originDecl = ParamDeclAttr::get(
      originName, OriginType::get(originName.getContext(), isMutable));

  // Tell the signature about the new origin decl.
  tcSignature.implicitOriginDecls.push_back(originDecl);

  return RefType::get(type,
                      ParamDeclRefAttr::get(originName, originDecl.getType()));
}

// If this argument is a pack vararg like "*args: *Ts" then the argument
// expression is "Ts", and the star before it was syntactically parsed.
// This expression must be a PValue of variadic metatype.  We need to
// process it into a VariadicPack.
static ASTType
typeCheckVariadicPackTypeSpecifier(ParsedArgument &arg, size_t argIdx,
                                   ExprEmitter &emitter,
                                   TypeCheckedFnSignature &tcSignature) {
  assert(arg.vararg == VarArgKind::PackVarArg &&
         "this applies to pack arguments");

  PValue param = emitter.emitExprPValue(arg.typeExpr, EC_Type);
  if (!param) // Error emitting the expression is already diagnosed.
    return {};

  // Make sure the param value is a variadic list of types.
  auto paramVariadicType = dyn_cast<VariadicType>(param.getRValueType());
  if (!paramVariadicType) {
    emitter.emitError(arg.typeExpr->getLoc(),
                      "pack argument type list must reference a variadic list")
        << arg.typeExpr->getRange();
    return {};
  }
  Type elementType = paramVariadicType.getElementType();
  if (isa<TypeType>(elementType)) {
    emitter.emitError(arg.loc)
        << "variadic pack elements declared as 'AnyTrivialRegType' are removed,"
        << " please declare elements as 'AnyType' instead of "
           "'AnyTrivialRegType'";
    return {};
  }

  auto metaType = ASTType(elementType).getMetaType();
  if (!metaType || !isa<StructMetaType, AnyTraitType>(metaType)) {
    emitter.emitError(arg.typeExpr->getLoc(),
                      "argument type list elements must be types")
        << arg.typeExpr->getRange();
    return {};
  }

  // The reference is immutable when borrowing, mutable otherwise.
  bool isMutable = arg.convention != ParsedArgument::kConventionRead &&
                   arg.convention != ParsedArgument::kConventionUnspec;
  bool isOwned = arg.convention == ParsedArgument::kConventionOwned;

  // Arguments passed by memory need an associated origin parameter, and need
  // to be passed by reference.
  RefType refType = makeImplicitRefTypeForArg(arg, argIdx, elementType,
                                              isMutable, tcSignature);

  // Form a VariadicPack type.  Note that we cannot use ParamBindings to do this
  // as we have no way to "splat" the type list into the variadic list :-(.
  ASTType variadicPackType =
      emitter.shared.getBuiltinVariadicPackType(emitter.declScope, arg.loc);
  if (isa<TypeCheckErrorType>(variadicPackType))
    return {}; // Sanity check the returned VariadicPack declaration.
  ASTDecl *packDecl = variadicPackType.getDecl(emitter.shared);

  // We expect:
  // VariadicPack[
  //   mut: Bool, //, is_owned: Bool, origin: Origin[mut],
  //   element_trait: _AnyTypeMetaType, *element_types: element_trait]
  auto packStruct = dyn_cast_if_present<StructDeclOp>(packDecl);
  if (!packStruct || packStruct.getParams().size() != 5) {
    emitter.emitError(arg.loc, "malformed VariadicPack");
    return {};
  }
  auto typeSig = packStruct.getSignature();

#if 0
  // TODO: This should work, but cannot because the type list is parametric and
  // we have no "splat list" operator.
  ParamBindings bindings(emitter.declScope);
  bindings.add(arg.typeExpr, refType.getOrigin());
  bindings.add(arg.typeExpr, PValue(elementType));
  bindings.add(arg.typeExpr, param.get());

  ParameterExprArrayAttr bindingValuesAttr = bindings.verifyBindings(
      packStruct, typeSig, arg.typeExpr->getLoc(), /*partial=*/false);
  if (!bindingValuesAttr)
    return {};
  LIT::StructType boundType =
      packStruct.bindAll(bindingValuesAttr.getValue());
  return TypeParamAttr::get(boundType, StructMetaType::get(boundType));
#endif

  auto isMutType = typeSig.getParamTypes()[0];
  auto isOwnedType = typeSig.getParamTypes()[1];
  auto originType = typeSig.getParamTypes()[2];
  auto traitMetaType = typeSig.getParamTypes()[3];
  if (!isa<LIT::StructType>(isMutType) || !isa<LIT::StructType>(isOwnedType) ||
      !isa<LIT::StructType>(originType) || !isa<AnyTraitType>(traitMetaType) ||
      !isa<VariadicType>(typeSig.getParamTypes()[4])) {
    emitter.emitError(arg.loc, "malformed VariadicPack");
    return {};
  }

  // Use a ParameterEvaluator to figure out which (rebound) types are needed,
  // so we get the Bool type, the Origin type etc.
  ParameterEvaluator evaluator;
  PValue isMut = emitter.emitPValue({refType.isMutable(), arg.typeExpr},
                                    EC_Type, isMutType);
  if (!isMut)
    return {};
  evaluator.addInputValue(isMut);

  auto isOwnedAttr = BoolAttr::get(emitter.getContext(), isOwned);
  PValue isOwnedVal =
      emitter.emitPValue({isOwnedAttr, arg.typeExpr}, EC_Type, isOwnedType);
  if (!isOwnedVal)
    return {};
  evaluator.addInputValue(isOwnedVal);

  PValue origin =
      emitter.emitPValue({refType.getOrigin(), arg.typeExpr}, EC_Type,
                         evaluator.getReboundType(originType));
  if (!origin)
    return {};
  evaluator.addInputValue(origin);

  // The default element_trait param type is
  // !lit.anytrait<<@stdlib::@builtin::@anytype::@AnyType>>
  // reflecting that it takes any trait like Stringable.
  // If the declared type of the pack elements is a trait subtype of AnyType,
  // it will be that traits metatype.  Downcast to the same type, but with
  // !lit.anytrait<AnyType> type.
  PValue traitMT = emitter.emitPValue({PValue(elementType), arg.typeExpr},
                                      EC_Type, traitMetaType);
  if (!traitMT)
    return {};
  evaluator.addInputValue(traitMT);

  // Bind the VariadicPack[isMutable, origin, element_trait, element_types]
  // parameters.
  return packStruct.bindReference({isMut.get(), isOwnedVal.get(), origin.get(),
                                   traitMT.get(), param.get()});
}

/// Type check each argument in turn, resolving their type and default
/// initializer value.  Arguments in Mojo can refer to previous arguments in
/// their type+default value expressions as PValues, so we need to ensure that
/// they are emitted and have declarations registered in the scope so that later
/// lookups can find them.
static void typeCheckOneArgument(size_t idx, bool isStaticMethod,
                                 ASTDecl *fnDecl,
                                 TypeCheckedFnSignature &tcSignature) {
  ParsedArgument &arg = tcSignature.argList.parsedArgs[idx];

  ASTDecl &declScope = tcSignature.paramList.declScope;
  SharedState &shared = declScope.getShared();
  ExprEmitter typeEmitter(declScope, EC_Type);

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
      // needs to be type checked.
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
    type = addImplicitTypeParams(type, tcSignature.paramList, /*append=*/true);
  } else if (idx == 0 && tcSignature.selfType &&
             // FIXME: This is incorrect, the @static_method decorators haven't
             // been applied yet.
             !isStaticMethod) {
    // If this is the 'self' argument in a struct, default the type to Self.
    type = tcSignature.selfType;
  } else {
    // Otherwise, this is an error.
    // TODO: We could explore making type annotations optional in 'def'
    // functions when we have a functional "Object" type.
    shared.emitError(arg.loc, "argument type must be specified")
        << SourceRange(arg.loc, arg.loc);
    type = shared.getTypeCheckErrorType();
    arg.isErroneous = true;
  }
  assert(type && "must have an argument type");
  tcSignature.argTypes.push_back(type);

  // Check if the argument is a parametric function.
  if (auto fType = dyn_cast<FnTypeGeneratorType>(type)) {
    if (!fType.getInputParamTypes().empty()) {
      arg.isErroneous = true;
      shared.emitError(shared.diags.translateLocation(arg.typeExpr->getLoc()),
                       "parametric functions may not be used as arguments; "
                       "consider passing as a parameter instead");
    }
  }

  // If no convention was explicitly specified, default to 'read'.
  if (arg.convention == ParsedArgument::kConventionUnspec) {
    // TODO: enable other conventions for **kwargs.
    arg.convention = arg.vararg == VarArgKind::KWVarArg
                         ? ParsedArgument::kConventionOwned
                         : ParsedArgument::kConventionRead;
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
    // Owned arguments are always passed in memory, allowing us to check for
    // exclusivity and other requirements.  Register passable arguments are
    // promoted to being passed in registers after elaboration.
    arg.kgenConvention = ArgConvention::OwnedMem;
    break;
  case ParsedArgument::kConventionRef: {
    if (arg.vararg != VarArgKind::None) {
      // There should be no reason this isn't supportable.
      shared.emitError(
          arg.loc, "TODO: variadic isn't supported with 'ref' convention yet");
      arg.vararg = VarArgKind::None;
    }
    auto refType =
        processRefOriginSpecifier(arg.refOriginExpr, type, arg.name,
                                  tcSignature.paramList, /*isResult=*/false);
    type = refType;
    if (refType.isMutableKnown(true))
      arg.kgenConvention = ArgConvention::MutRef;
    else
      arg.kgenConvention = ArgConvention::Ref;

    if (isa<TypeCheckErrorType>(type.getReferenceElementType()))
      arg.isErroneous = true;
    break;
  }
  case ParsedArgument::kConventionRead: {
    arg.kgenConvention = ArgConvention::ReadMem;
    TypeConvention conv = type.getRegisterPassability(arg.loc, shared);
    // FIXME(MOCO-725): Borrows of non-trivial register-passable values don't
    // have origins and can't be correctly tracked if captured in an async
    // function. Emit an error to avoid a footgun.
    if (arg.vararg != VarArgKind::PackVarArg &&
        conv == TypeConvention::RegisterPassable &&
        tcSignature.argList.effects.isAsync()) {
      shared.emitError(
          arg.loc, "TODO: read-only non-trivial register-passable arguments "
                   "are not yet supported in async functions");
    }
    // We can pass trivial register borrowed arguments in a register.  We cannot
    // pass non-trivial ones because we cannot diagnose ownership and have other
    // lifetime issues.
    if (conv == TypeConvention::RegisterPassableTrivial)
      arg.kgenConvention = ArgConvention::ReadReg;
    break;
  }
  case ParsedArgument::kConventionMut:
    arg.kgenConvention = ArgConvention::Mut;
    break;

  case ParsedArgument::kConventionOut:
    llvm_unreachable("Should remove this");
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
    case ParsedArgument::kConventionOut:
      llvm_unreachable("not a pack arg convention");
    case ParsedArgument::kConventionOwned:
      arg.kgenVariadicConvention = ArgConvention::OwnedMem;
      arg.kgenConvention = ArgConvention::OwnedMem;
      break;
    case ParsedArgument::kConventionRead:
      arg.kgenVariadicConvention = ArgConvention::ReadMem;
      arg.kgenConvention = ArgConvention::ReadMem;
      break;
    case ParsedArgument::kConventionMut:
      arg.kgenVariadicConvention = ArgConvention::Mut;
      arg.kgenConvention = ArgConvention::ReadMem;
      break;
    }
  }

  // Values passed by memory need an associated origin parameter, and need to
  // be passed by reference. For now, we don't use reference types in **kwargs.
  Type fullType;
  if (hasImplicitOrigin(arg.kgenConvention) &&
      arg.vararg != VarArgKind::KWVarArg) {
    bool isMutable = arg.kgenConvention != ArgConvention::ReadMem;
    fullType =
        makeImplicitRefTypeForArg(arg, idx, type, isMutable, tcSignature);
  } else {
    fullType = type;
  }

  // If this is a valid vararg argument, then we pass it as a variadic type.
  // The convention is to pass as a register value, in the case of a memory
  // value, we're passing the array of pointers by value.
  if (arg.vararg == VarArgKind::VarArg) {
    fullType = VariadicType::get(fullType, arg.kgenConvention);
    arg.kgenConvention = ArgConvention::ReadReg;
  } else if (arg.vararg == VarArgKind::KWVarArg) {
    // We build OwnedKwargsDict[ValType].
    ASTType dictType = shared.getOwnedKwargsDictType(arg.loc);

    auto dictDecl = cast<LIT::StructType>(dictType.mlirType);
    // We know these are all UnboundAttrs created by
    // StructDeclOp::bindReference. The correct way is to have bindReference
    // return a GeneratorType.
    ArrayRef<TypedAttr> inputUnboundParams = dictDecl.getParamValues();
    if (inputUnboundParams.size() != 1) {
      shared.emitError(arg.loc)
          << "internal compiler error: OwnedKwargsDict type has unexpected "
             "parameter signature; please file a bug";
      arg.isErroneous = true;
    }

    // If anything is wrong with the argument, we terminate before emitting a
    // type for the variadic keyword arguments.
    if (arg.isErroneous)
      return;

    auto collectionElement = cast<TraitType>(inputUnboundParams[0].getType());
    SyntheticNode typeExpr(arg.loc);
    auto typeExprToUse = arg.typeExpr ? arg.typeExpr : &typeExpr;
    auto binding = typeEmitter.emitPValue({fullType, typeExprToUse}, EC_Type,
                                          collectionElement);
    if (!binding) {
      arg.isErroneous = true;
      return;
    }
    fullType = cast<LIT::StructType>(dictType).bindAll(binding.get());

    // OwnedKwargsDict is memory only and since only the callee can access it,
    // we pass it as owned.
    arg.kgenConvention = ArgConvention::OwnedMem;
    fullType = makeImplicitRefTypeForArg(arg, idx, fullType, /*isMutable*/ true,
                                         tcSignature);
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
  Block &blockOwningArg =
      fnDecl ? *cast<FnOp>(fnDecl).getBody() : shared.getArgumentOwningBlock();
  BlockArgument bbArg =
      blockOwningArg.addArgument(fullType, shared.translateLocation(arg.loc));

  DeclIRValue argIRValue;
  if (arg.kgenConvention == ArgConvention::ReadReg)
    argIRValue = SRValue(bbArg);
  else // Everything else is passed in memory.
    argIRValue = CValue::getMValueForRef(bbArg);

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
static void typeCheckResult(ParsedArgument resultArg,
                            const SpecialFunctionInfo &fnInfo, ASTDecl *fnDecl,
                            TypeCheckedFnSignature &tcSignature) {
  ASTDecl &declScope = tcSignature.paramList.declScope;
  SharedState &shared = tcSignature.paramList.shared;

  // Determine the result type based on what was explicitly written or what
  // the right implicit result type is.
  ASTType resultType;
  if (resultArg.typeExpr &&
      resultArg.typeExpr->kind == ExprNode::kNoneLiteral) {
    // If the result type is a `None` literal, then convert it to NoneType.
    resultType = shared.getNoneType();
  } else if (resultArg.typeExpr) {
    ExprEmitter typeEmitter(declScope, EC_Type);
    resultType = typeEmitter.emitExprType(resultArg.typeExpr);

    // On error, a diagnostic will be emitted, but we don't want to kill the
    // entire function definition.  We won't be able to correctly type check any
    // calls to this function though.
    if (!resultType)
      resultType = shared.getTypeCheckErrorType();
  } else if (fnInfo.isInitializer() &&
             resultArg.convention == ParsedArgument::kConventionOut) {
    // If this is an initializer with an 'out self' argument, infer Self.
    resultType = tcSignature.selfType;
  } else {
    // If the result type wasn't specified, we default to "None".
    resultType = shared.getNoneType();
  }

  // If a result origin is specified with `ref [life] Ty`, then form a ref
  // result.
  if (resultArg.refOriginExpr) {
    if (tcSignature.argList.effects.isAsync()) {
      // TODO(MOCO-787): Async functions don't support ref results yet. We need
      // to define a `CoroutineRef` or support perfect forwarding in generic
      // results.
      shared.emitError(resultArg.refOriginExpr->getLoc())
          << "TODO: ref results aren't supported in async functions yet";
      resultArg.refOriginExpr = nullptr;
    } else {
      resultType = processRefOriginSpecifier(
          resultArg.refOriginExpr, resultType,
          // TODO: Use the name of the return slot if present.
          "__result__", tcSignature.paramList, /*isResult*/ true);
      tcSignature.argList.effects.setRefResult(isa<RefType>(resultType));
    }
  }

  // Remember the user-declared result type.
  tcSignature.resultType = resultType;

  // Check to see if the result type has any embedded origins that refer to
  // in-memory argument origins of generic type, e.g.:
  //
  //     fn get[T: AnyType](a: T) -> Pointer[T, __origin_of(a)]:
  //        return Pointer(a)
  //
  // These origins are not allowed to be returned from the function, because
  // when instantiated with a register-passable type, argument convention
  // lowering will turn them into:
  //
  //     fn get[T: AnyType](borrow_in_reg a: T)
  //                             -> Pointer[T, __origin_of(tmp)]:
  //        var tmp = a
  //        return Reference(tmp)
  //
  // Note that we're now returning a reference to something that doesn't outlast
  // the function!
  if (auto resultOrigins =
          shared.cachedOriginFinder.findOriginsIn(resultType.mlirType);
      !resultOrigins.empty()) {
    SmallDenseMap<TypedAttr, size_t, 8> possiblyRegisterPassableOrigins;
    for (auto [idx, parsedArg, fullType] : llvm::enumerate(
             tcSignature.argList.parsedArgs, tcSignature.fullArgTypes)) {

      // Only look at mut, read, owned arguments.  RegisterPassable args
      // won't have a origin, and `ref` args are not lowered by-reg.
      if (!hasAddress(parsedArg.kgenConvention) ||
          parsedArg.kgenConvention == ArgConvention::Ref ||
          parsedArg.kgenConvention == ArgConvention::MutRef)
        continue;

      // The argument is only a potential problem if it is generic that might
      // expand to a @register_passable type.
      auto refType = cast<RefType>(fullType);
      if (!ASTType(refType.getElementType())
               .mightBeRegisterPassable(parsedArg.loc, shared))
        continue;

      // Ok, this origin is a problem.
      possiblyRegisterPassableOrigins[refType.getOrigin()] = idx;
    }

    // Now that we know all the problematic origins, check to see if any of
    // them are referenced.
    for (TypedAttr origin : resultOrigins) {
      // Don't allow mutability dropping to interfere.
      origin = OriginMutCastAttr::strip(origin);
      if (!possiblyRegisterPassableOrigins.count(origin))
        continue;

      // Oops, found a problem, report it and indicate the argument at fault.
      assert(resultArg.typeExpr && "implicit result types can't have origins");
      size_t argIdx = possiblyRegisterPassableOrigins[origin];
      const ParsedArgument &badArg = tcSignature.argList.parsedArgs[argIdx];
      auto diag = shared.emitError(resultArg.typeExpr->getLoc());
      diag << "cannot return " << badArg.name << "s origin, because it ";
      ASTType argType =
          ASTType(tcSignature.fullArgTypes[argIdx]).getReferenceElementType();
      if (argType.isRegisterPassable(badArg.loc, shared))
        diag << "has @register_passable type " << argType;
      else
        diag << "might expand to a @register_passable type";
      diag << resultArg.typeExpr->getRange()
           << SourceRange(badArg.loc, badArg.loc);
      break;
    }
  }

  // Now that we have the user's result type, compute the full type of the
  // result, which can can be different when memory only, when throwing, etc.
  ASTType fullResultType = resultType;
  TypeConvention rp = resultType.getRegisterPassability(resultArg.loc, shared);

  // If this function throws, add a result slot for the error that may be
  // raised.
  if (tcSignature.argList.effects.isThrows()) {
    ASTType errorType = shared.getBuiltinErrorType(
        tcSignature.paramList.declScope, resultArg.loc);

    // Synthesize a ByRefError argument for the error.
    ParsedArgument errArg;
    errArg.loc = resultArg.loc;
    errArg.name = StringAttr::get(shared.getContext(), "__error__");
    errArg.convention = ParsedArgument::kConventionByRefResult;
    errArg.kgenConvention = ArgConvention::ByRefError;
    errArg.kwArgHandling = KWArgHandling::kKeywordOnly;
    errArg.typeExpr = nullptr;
    tcSignature.argList.parsedArgs.push_back(errArg);
    tcSignature.argTypes.push_back(errorType);

    RefType refType = makeImplicitRefTypeForArg(
        errArg, 0, errorType, /*isMutable*/ true, tcSignature);
    tcSignature.fullArgTypes.push_back(refType);

    // If this is for a lit.fn declaration (as opposed to a function type),
    // add a block argument for this.
    if (fnDecl) {
      Block &body = *cast<FnOp>(fnDecl).getBody();
      (void)body.addArgument(refType, shared.translateLocation(resultArg.loc));
    }

    // The ABI result type is an i1 indicating the error state.
    fullResultType = Builder(shared.getContext()).getI1Type();
    // The result value is always returned through memory.
    rp = TypeConvention::MemoryOnly;
  }

  // Async functions always use in-memory results.
  if (tcSignature.argList.effects.isAsync())
    rp = TypeConvention::MemoryOnly;

  // If it is memory-only, pass it indirectly as the last argument to the
  // function by-reference.
  if (rp == TypeConvention::MemoryOnly) {
    // Synthesize a ByRefResult argument for the result.
    if (!resultArg.name)
      resultArg.name = StringAttr::get(shared.getContext(), "__result__");
    resultArg.convention = ParsedArgument::kConventionByRefResult;
    resultArg.kgenConvention = ArgConvention::ByRefResult;
    resultArg.kwArgHandling = KWArgHandling::kKeywordOnly;
    tcSignature.argList.parsedArgs.push_back(resultArg);
    tcSignature.argTypes.push_back(resultType);

    // Compute the RefType for this new argument with an implicit origin.
    RefType refType = makeImplicitRefTypeForArg(
        resultArg, 0, resultType, /*isMutable*/ true, tcSignature);
    tcSignature.fullArgTypes.push_back(refType);

    // If this is for a lit.fn declaration (as opposed to a function type),
    // add a block argument for this.  We don't register this for name lookup
    // though, we don't want it to conflict with user identifiers, and it is
    // never looked up directly.
    if (fnDecl) {
      Block &body = *cast<FnOp>(fnDecl).getBody();
      auto bbArg =
          body.addArgument(refType, shared.translateLocation(resultArg.loc));

      // Add a decl so this will be found by name lookup within the body.
      shared.getDeclResolver().addFullyResolvedDecl(
          MLValue(bbArg), resultArg.name, resultArg.loc, &declScope);
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
TypeCheckedFnSignature::TypeCheckedFnSignature(TypeCheckedParamList &paramList,
                                               ParsedArgumentList &argList,
                                               const ExprNode *originExpr,
                                               ASTDecl *fnDecl,
                                               SpecialFunctionInfo &fnInfo)
    : paramList(paramList), argList(argList) {
  SharedState &shared = paramList.shared;
  ExprEmitter typeEmitter(paramList.declScope, EC_Type);

  // If this definition is a struct/class member, compute the self type.
  if (fnDecl) {
    if (ASTDecl *parent = fnDecl->tryGetMethodParentDecl()) {
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
  auto checkInitializer = [&]() -> LogicalResult {
    if (!selfType) {
      fnDecl->setErroneous();
      shared.emitError(fnDecl->getLoc(), "'")
          << fnInfo.name << "' must be a method";
      return failure();
    }

    // Initializers without an out argument or a -> Self result may be legacy
    // form.
    if (!argList.resultArg.name && !argList.parsedArgs.empty() &&
        argList.parsedArgs[0].convention == ParsedArgument::kConventionMut &&
        (!argList.resultArg.typeExpr || // Allow "no ->" and "-> None"
         argList.resultArg.typeExpr->kind == ExprNode::kNoneLiteral)) {
      // TODO(25.4): Make this an error.
      shared.emitWarning(argList.parsedArgs[0].loc,
                         "__init__ method with 'mut' convention is deprecated, "
                         "please use 'out' instead");
      argList.resultArg = argList.parsedArgs[0];
      argList.parsedArgs.erase(argList.parsedArgs.begin());
      argList.resultArg.convention = ParsedArgument::kConventionOut;
    }

    // TODO(MOCO-789): Async initializers require a `byref_result` thunk to be
    // emitted. Just forbid them for now.
    if (argList.effects.isAsync()) {
      shared.emitError(fnDecl->getLoc())
          << "TODO: async constructors are not yet supported";
      argList.effects.setAsync(false);
      return failure();
    }

    // @register_passable values are movable by passing the register around, so
    // they can't define a moveinit.
    if (fnInfo.kind == SpecialFunctionKind::kMoveInit &&
        selfType.isRegisterPassable(fnDecl->getLoc(), shared)) {
      shared.emitError(fnDecl->getLoc(),
                       "'@register_passable' types may not have a '")
          << fnInfo.name
          << "' method, they are always movable by copying a register";
      return failure();
    }

    // Trivial types are copyable with memcpy so they can't define copyinit.
    if (fnInfo.kind == SpecialFunctionKind::kCopyInit &&
        selfType.isTrivial(fnDecl->getLoc(), shared)) {
      shared.emitError(fnDecl->getLoc(), "trivial types may not have a '")
          << fnInfo.name << "' method, they are always trivially copyable";
      return failure();
    }

    return success();
  };

  // Check initializers for validity.
  if (fnInfo.isInitializer()) {
    if (failed(checkInitializer())) {
      fnDecl->setErroneous();
      fnInfo = SpecialFunctionInfo();
    }
  }

  // __new__ and __init__ are implicitly static.
  if (fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod)
    cast<FnOp>(fnDecl).setIsStatic(true);

  // Trivial types are copyable with memcpy so they can't define copyinit.
  if (fnInfo.kind == SpecialFunctionKind::kDel &&
      selfType.isTrivial(fnDecl->getLoc(), shared)) {
    fnDecl->setErroneous();
    shared.emitError(fnDecl->getLoc(), "trivial types may not have a '")
        << fnInfo.name << "' method, they are always trivially destroyable";
    fnInfo = SpecialFunctionInfo();
  }

  // True if this is a static method.
  // FIXME: This is completely wrong, @static_method decorator hasn't been
  // applied yet.
  //
  // It isn't clear if this is actually that bad, maybe we should just say that
  // first arguments in methods default to Self it they don't have type.  This
  // could be true for static methods as well.
  bool isStaticMethod = selfType && cast<FnOp>(fnDecl).getIsStatic();

  // Resolve all argument types, generating type check error types for any types
  // that could not be correctly resolved.
  for (size_t i = 0, e = argList.parsedArgs.size(); i != e; ++i)
    typeCheckOneArgument(i, isStaticMethod, fnDecl, *this);

  // Compute the result type.
  typeCheckResult(argList.resultArg, fnInfo, fnDecl, *this);

  // If a capture origin set was specified, emit it. It will be added to the
  // signature type later.
  if (originExpr) {
    // Special rule for `[_]` when specifying the capture origin set: the
    // set is unbound and will be autoparameterized.
    auto setType = OriginSetType::get(shared.getContext());
    if (originExpr->kind == ExprNode::kDiscardLiteral) {
      captureOrigins = UnboundAttr::get(setType);
    } else {
      captureOrigins =
          typeEmitter.emitExprPValue(originExpr, EC_Origin, setType);
    }
  }
}

/// This performs any special checks over the declaration based on its name
/// and whether it is a method.  This happens after decorator processing
/// because that is how defs work in Python.
///
/// If this function detects a problem, it marks the decl as erroneous and
/// resets the SpecialFunctionInfo.
void TypeCheckedFnSignature::verifyFunctionNameBinding(
    ASTDecl &decl, StringAttr name, SpecialFunctionInfo &fnInfo) const {
  FnOp funcOp = cast<FnOp>(decl);

  ArrayRef<ParsedArgument> parsedArgs = argList.parsedArgs;
  ArrayRef<Type> argTypes = this->argTypes;
  auto &shared = paramList.shared;

  // On any semantic error we mark the declaration erroneous - so references to
  // it don't type check, and we clear our special function information.  This
  // reduces cascade errors.
  auto emitErrorLoc = [&](SMLoc loc,
                          const Twine &message = Twine()) -> InflightDiag {
    fnInfo = SpecialFunctionInfo();
    decl.setErroneous();
    return shared.emitError(loc, message);
  };
  auto emitError = [&](const Twine &message = Twine()) -> InflightDiag {
    fnInfo = SpecialFunctionInfo();
    decl.setErroneous();
    return shared.emitError(funcOp.getLoc(), message);
  };

  // If the argument list has a mut result or mut error, ignore it for type
  // checking purposes.
  while (!parsedArgs.empty() && parsedArgs.back().convention ==
                                    ParsedArgument::kConventionByRefResult) {
    parsedArgs = parsedArgs.drop_back();
    argTypes = argTypes.drop_back();
  }

  // If this definition is a struct/class member, compute the self type.
  ASTType selfType;
  constexpr size_t kSelfArgNo = 0;
  if (ASTDecl *parent = decl.getParentDecl();
      parent && isa<StructDeclOp, TraitDeclOp>(*parent)) {
    // The parent decl must be fully resolved in order to resolve any of its
    // members.
    assert(parent->resolvedness == DeclResolvedness::fully);
    selfType = parent->getTypeDeclSelf();
  }

  // Check any special function information.

  // Check that the 'self' argument/result of a method was specified correctly.
  if (selfType && (!funcOp.getIsStatic() ||
                   (fnInfo.flags & SpecialFunctionInfo::kSelfResult))) {
    // Implement this as a lambda so we can early exit with 'return'.
    auto checkSelf = [&](ASTType selfArgType, const ParsedArgument &selfArg) {
      // Don't check broken args, because we don't want redundant diagnostics.
      if (selfArg.isErroneous)
        return;

      // It ok if it exactly matches (typically with a specific convention).
      if (selfType.isEqualCanon(selfArgType))
        return;

      // If an error was already diagnosed with the type, disable follow-ons.
      if (isa<TypeCheckErrorType>(selfArgType)) {
        selfArg.isErroneous = true;
        return;
      }

      // It is ok if the self type has different parameters than the
      // declaration, this is a form of conditional conformance.
      if (selfType.getWithoutParameters(shared).isEqualCanon(
              selfArgType.getWithoutParameters(shared)))
        // TODO: We should check to make sure the parameters are a subtype of
        // the declared parameters.  We don't want Self to say T is Movable, but
        // then have it be implemented with AnyType.
        // Replacing this whole thing with 'where' clauses would be much nicer
        // anyhow.
        return;

      // Otherwise, this is an unrecognized self type. If this is a trait, the
      // explicit self type is very hard to specify in mojo, so we suggest to
      // use 'Self' instead.
      auto diag = emitErrorLoc(selfArg.loc, "'self' argument must have type ");
      if (isa<TraitDeclOp>(*decl.getParentDecl()))
        diag << "'Self' in trait method declaration";
      else
        diag << selfType;
      diag << ", but actually has type " << selfArgType;
      selfArg.isErroneous = true;
      if (selfArg.typeExpr)
        diag << selfArg.typeExpr->getRange();
    };

    if (fnInfo.flags & SpecialFunctionInfo::kSelfResult) {
      // __new__ and __init__ require a Self result type, or a specialization
      // thereof.
      checkSelf(resultType, argList.resultArg);
    } else if (argTypes.empty()) {
      // TODO('def' allows unused arguments): We can/should relax this for
      // 'def' declarations in the future, they should be able to implicit
      // ignore arguments like Python does.
      emitError("self argument must be present in instance method");
    } else {
      // Normal methods require a self argument.
      checkSelf(argTypes[kSelfArgNo], parsedArgs[kSelfArgNo]);
    }
  }

  // Verify the argument count lines up.
  if (fnInfo.kind != SpecialFunctionKind::kNormal) {
    size_t numActualArgs = parsedArgs.size();
    size_t numMin = fnInfo.minNumArguments;
    ssize_t numMax = fnInfo.maxNumArguments;
    if (numMin == size_t(numMax) && numActualArgs != numMin) {
      emitError() << name << " requires " << numMin << " operand"
                  << plural(numMin);
    } else if (numActualArgs < numMin) {
      emitError() << name << " requires at least " << numMin << " operand"
                  << plural(numMin);
    } else if (numMax != -1 && numActualArgs > size_t(numMax)) {
      emitError() << name << " requires at most " << size_t(numMax)
                  << " operand" << plural(numMax);
    }
  }

  // Check other invariants based on method flags.
  if (fnInfo.isInstMethod()) {
    if (!selfType) {
      emitError() << name << " must be a method";
    } else if (funcOp.getIsStatic()) {
      if (!(fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod))
        emitError("special method may not be a static method");
    } else if (fnInfo.requiresOwnedSelfInstMethod() &&
               parsedArgs[kSelfArgNo].convention !=
                   ParsedArgument::kConventionOwned) {
      emitErrorLoc(parsedArgs[kSelfArgNo].loc, "self argument must be 'owned'")
          << FixIt::insertBeforeToken(parsedArgs[kSelfArgNo].loc, "owned ");
    }
  }

  // Get the user-declared result type, which might be a memory-only type.
  ASTType declaredResultType = resultType;

  // If the function is required to return None, verify that.
  if (fnInfo.hasNoneResult() && !declaredResultType.isNoneType())
    emitError() << name << " result type must be elided (or None)";

  // Reject special functions declared as throwing when that is invalid.
  if (argList.effects.isThrows() &&
      fnInfo.flags & SpecialFunctionInfo::kCannotRaise) {
    // Specialize the error if raising is implicit because it was defined as a
    // def.
    if (funcOp.isDef()) {
      emitError() << "cannot define " << name
                  << " as 'def'; 'def' implicitly raises"
                  << FixIt::replaceToken(decl.getLoc(), "fn");
    } else {
      emitError() << name << " cannot be declared as raising an exception";
    }
  }

  // Diagnose common errors and handle other special cases.
  switch (fnInfo.kind) {
  default:
    break;
  case SpecialFunctionKind::kNew:
    emitError("'__new__' is not supported on structs; use '__init__' instead");
    break;
  case SpecialFunctionKind::kMLIRI1:
    if (!declaredResultType.mlirType.isSignlessInteger(1))
      emitError() << name << " result type must be __mlir_type.i1";
    break;
  case SpecialFunctionKind::kCopyInit:
  case SpecialFunctionKind::kMoveInit:
    assert(parsedArgs.size() == 1 && "arg count already checked above");
    if (fnInfo.kind == SpecialFunctionKind::kCopyInit) {
      if (parsedArgs[0].convention != ParsedArgument::kConventionRead)
        emitErrorLoc(parsedArgs[0].loc,
                     "existing value argument must be passed as 'read'");
    } else if (fnInfo.kind == SpecialFunctionKind::kMoveInit) {
      if (parsedArgs[0].convention != ParsedArgument::kConventionOwned)
        emitErrorLoc(parsedArgs[0].loc,
                     "existing value argument must be passed as 'owned'");
    }
    break;
  }

  // If we have a special function kind and didn't have any errors with it,
  // remember which kind it is.
  if (fnInfo.kind != SpecialFunctionKind::kNormal)
    funcOp.setSpecialFnKind(uint8_t(fnInfo.kind));
}

FunctionType TypeCheckedFnSignature::getFunctionType() const {
  return FunctionType::get(fullResultType.mlirType.getContext(), fullArgTypes,
                           {fullResultType.mlirType});
}

/// Form a LIT signature packaging up all the stuff we need to know about this
/// type checked function.
FnTypeGeneratorType TypeCheckedFnSignature::getFnTypeGeneratorType() const {
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
      implicitOriginDecls.size(),
      getOriginsAccessibleByParams(paramList.getParamListAttr(),
                                   paramList.paramDeclAttrs, paramList.shared,
                                   captureOrigins),
      isNestedOriginExclusivityCheckingDisabled);

  /// Silence internal verifier errors when constructing types from the parser.
  /// We don't want to show these to the user.
  auto silenceErrors = [ctx] {
    InFlightDiagnostic diag = mlir::emitError(UnknownLoc::get(ctx));
    diag.abandon();
    return diag;
  };

  FunctionType functionType = getFunctionType();
  return FuncTypeGeneratorType::remapToFuncTypeGenerator(
      paramList.paramDeclAttrs, functionType, argConventions, argList.effects,
      metadata, paramList.getParamListAttr(), silenceErrors);
}
