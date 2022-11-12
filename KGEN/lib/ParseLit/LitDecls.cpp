//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Declaration parsing and name binding logic.
//
//===----------------------------------------------------------------------===//

#include "LitDecls.h"
#include "ASTDecl.h"
#include "IRValues.h"
#include "LitLexer.h"
#include "LitParserBase.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LitSharedState.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// ASTDecl
//===----------------------------------------------------------------------===//

/// If this is an RValue, return it otherwise return null.
RValue ASTDecl::getIfRValue() const {
  // Meta value.
  if (auto attr = dyn_cast_or_null<MAValue>(irValue))
    return MValue(attr);
  // DRValue.
  if (auto value = dyn_cast_or_null<DRValue>(irValue))
    return value;
  return {};
}

/// Given an MLIR op for a struct declaration, return the self type.
ASTType ASTDecl::computeSelfTypeForStruct(LitSharedState &state) {
  auto structOp = cast<LITStructDeclOp>(*this);

  SmallVector<LitSharedState::ParamBinding> parameters;
  for (auto decl : structOp.getParamDecls()) {
    // We're using the parameter from the type declaration scope in the
    // parameter binding list.
    TypedAttr ref = ParamDeclRefAttr::get(decl.getName(), decl.getType());
    parameters.push_back({decl, ref});
  }

  // Methods on structs (but not classes) take the struct implicitly by
  // pointer so they can use and mutate it.
  return state.getASTType(*this, parameters);
}

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

// Declarations (e.g. module, class, function) are parsed in multiple phases
// to increase laziness of the parse as well as make circular references
// possible.
//
// This ensures that the forward references between peer declarations are
// handled correctly as well as circular references, for example in mutually
// recursive functions and code like this:
//
//   def foo():
//     def bar():
//       print(x)
//     x = 42
//     bar()
//   foo()
DeclResolver::DeclResolver(LitSharedState &state) : sharedState(state) {}
DeclResolver::~DeclResolver() {
  // Run the destructors on all the ASTDecl objects to make sure any
  // transitively allocated data is released.
  for (ASTDecl *decl : parsedDeclList)
    decl->~ASTDecl();
}

/// Add a new declaration that needs to be resolved.
ASTDecl &DeclResolver::addDecl(DeclIRValue irValue, Location loc,
                               StringAttr name, ASTDecl *parentDecl,
                               LitLexerCursor cursor, LitLexerCursor endCursor,
                               ssize_t indentation) {
  ASTDecl *decl = sharedState.allocPersistent<ASTDecl>(
      irValue, loc, parentDecl, cursor, endCursor, indentation);
  parsedDeclList.push_back(decl);

  // If this has a parent and a name, insert it into the parents name table so
  // name lookup will resolve it.
  if (!parentDecl || !name)
    return *decl;

  auto [it, inserted] = parentDecl->declsInScope.insert({name, decl});
  if (!inserted) {
    ASTDecl *existing = it->second;
    auto diag =
        sharedState.emitError(decl->getLoc(), "invalid redefinition of ")
        << name;
    diag.attachNote(existing->getLoc()) << "previous definition here";

    // Mark the existing decl and this one as erroneous so uses of either
    // don't create confusing errors.
    decl->hasReferenceError = true;
    existing->hasReferenceError = true;
  }
  return *decl;
}

/// Add a new declaration that needs to be resolved.
ASTDecl &DeclResolver::addDecl(Operation *op, StringAttr name,
                               ASTDecl *parentDecl, LitLexerCursor cursor,
                               LitLexerCursor endCursor, ssize_t indentation) {
  return addDecl(op, op->getLoc(), name, parentDecl, cursor, endCursor,
                 indentation);
}

ASTDecl &DeclResolver::addFullyResolvedDecl(Operation *op, StringAttr name,
                                            ASTType type, ASTDecl *parentDecl) {
  auto &decl =
      addDecl(op, name, parentDecl, LitLexerCursor(), LitLexerCursor(), 0);
  decl.resolvedness = DeclResolvedness::fullyResolved;
  decl.setResolvedType(type);
  return decl;
}

/// Add a declaration that is already fully resolved.
ASTDecl &DeclResolver::addFullyResolvedDecl(DeclIRValue declVal,
                                            StringAttr name, Location loc,
                                            ASTType type, ASTDecl *parentDecl) {
  auto &decl = addDecl(declVal, loc, name, parentDecl, LitLexerCursor(),
                       LitLexerCursor(), 0);
  decl.resolvedness = DeclResolvedness::fullyResolved;
  decl.setResolvedType(type);
  return decl;
}

/// Add a "magic" declaration that has special handling to this scope.  This
/// is used for builtin machinery internal to the language.
ASTDecl &DeclResolver::addMagicDecl(StringRef name, MagicDeclKind kind,
                                    ASTDecl *parentDecl) {
  assert(parentDecl && "top level isn't magic");
  auto &decl = addDecl(MAValue(), parentDecl->getLoc(),
                       StringAttr::get(getContext(), name), parentDecl,
                       LitLexerCursor(), LitLexerCursor(), 0);
  decl.resolvedness = DeclResolvedness::fullyResolved;
  decl.magicKind = kind;
  decl.setResolvedType(sharedState.getASTType(decl, {}));
  return decl;
}

/// Resolve all of the declarations that are visible.
void DeclResolver::resolveAll(SMLoc loc) {
  // We can do this in any order, but choose to use the order they are
  // discovered so diagnostics are mostly top-down.  Resolving declarations
  // may cause more entries to be added to this list.
  for (size_t i = 0; i != parsedDeclList.size(); ++i)
    (void)resolve(*parsedDeclList[i], DeclResolvedness::fullyResolved, loc);
}

/// Resolve the specified declaration to at least the specified level of
/// resolution, performing incremental type checking as appropriate.
LogicalResult DeclResolver::resolve(ASTDecl &decl, DeclResolvedness howResolved,
                                    SMLoc loc) {
  // If decl is already resolved enough, we're done.
  if (decl.resolvedness >= howResolved) {
    // If decl is busted, then return failure.
    return success(!decl.hasReferenceError);
  }

  // If we are currently name binding this operation, we found a cycle, reject
  // it with an error.
  if (!declsCurrentlyProcessing.insert(&decl).second) {
    emitError(sharedState.translateLocation(loc),
              "recursive reference to declaration");
    return failure();
  }

  // If the signature hasn't been parsed, do so.
  if (decl.resolvedness < DeclResolvedness::signatureResolved) {
    // Handle each operation that can be name bound.  We handle this by
    // restoring the lexer to the position where parsing can continue, calling
    // the `resolveSignature` method for the op, and re-saving the new cursor
    // for the next stage of resolution.
    TypeSwitch<ASTDecl &>(decl)
        .Case<LITFuncOp, GeneratorInterfaceOp, LITStructDeclOp, VarDeclOp>(
            [&](auto op) {
              LitLexer lexer(sharedState, decl.getCursor());

              // Resolve the signature: on a parse error, we note that the decl
              // is malformed and should not be referenced to silence downstream
              // errors.
              if (failed(resolveSignature(op, lexer, decl)))
                decl.hasReferenceError = true;
              decl.getCursor() = lexer.getCursor();
            })
        .Case([&](ModuleOp op) { /*Nothing*/ })
        .Default([&](auto attr) {
          emitError(decl.getLoc(),
                    "do not know how to resolve the signature of this decl!");
        });
    decl.resolvedness = DeclResolvedness::signatureResolved;
  }

  // If the declaration hasn't been fully parsed and we need to, do so.
  if (decl.resolvedness < DeclResolvedness::fullyResolved &&
      howResolved == DeclResolvedness::fullyResolved) {
    // Handle each operation that can be name bound.
    TypeSwitch<ASTDecl &>(decl)
        .Case<LITFuncOp, LITStructDeclOp, VarDeclOp>([&](auto op) {
          // Parse the body of the declaration from the correct point.
          LitLexer lexer(sharedState, decl.getCursor());
          if (resolveBody(op, lexer, decl))
            return;

          // If the final parse of the declaration didn't match the initial
          // parse, report an error about unrecognized tokens at end of
          // declaration.
          if (!decl.isMatchingEndCursor(lexer.getCursor()) &&
              !decl.hasReferenceError) {
            if (lexer.getToken().isAny(LitToken::kw_def, LitToken::kw_struct,
                                       LitToken::kw_class, LitToken::kw_var))
              lexer.emitError("definition isn't on its own line at the correct "
                              "indentation");
            else
              lexer.emitError("unknown tokens at the end of a declaration");
          }
        })
        .Case<GeneratorInterfaceOp>([&](auto op) {
          LitLexer lexer(sharedState, decl.getCursor());
          LitToken currToken = lexer.getToken();
          // Allow an empty body
          if (currToken.is(LitToken::Kind::kw_pass) ||
              currToken.is(LitToken::Kind::dot_dot_dot))
            lexer.lexToken();

          if (!decl.isMatchingEndCursor(lexer.getCursor()) &&
              !decl.hasReferenceError)
            lexer.emitError("interfaces have no body: unknown tokens found");
        })
        .Case<ModuleOp>([&](auto op) { /*Nothing*/ })
        .Default([&](auto attr) {
          emitError(decl.getLoc(),
                    "do not know how to resolve the body of this decl!");
        });
    decl.resolvedness = DeclResolvedness::fullyResolved;
  }

  declsCurrentlyProcessing.erase(&decl);
  // If decl is busted, then return failure.
  return success(!decl.hasReferenceError);
}

//===----------------------------------------------------------------------===//
// Function Decl implementation
//===----------------------------------------------------------------------===//

namespace {
/// identifier_opt_type  ::= identifier [":" expression]
/// meta_signature    ::= "[" [meta_param_list] "]"
/// meta_param_list   ::= identifier_opt_type ("," identifier_opt_type)
struct ParsedMetaSignature {
  SmallVector<ParamDeclAttr> inputDecls;
  SmallVector<ASTType> inputASTTypes;
  std::vector<Location> inputLocs;

  ParseResult parseOptionalMetaSignature(LitParserBase &p, ASTDecl &decl) {
    if (!p.consumeIf(LitToken::l_square) || p.consumeIf(LitToken::r_square))
      return success();

    auto parseMetaParameter = [&]() -> ParseResult {
      inputLocs.push_back(p.getTokenLocation());

      StringAttr name;
      if (p.parseIdentifier(name, "expected parameter name")) {
        // TODO: Scan ahead for better recovery.
        return failure();
      }

      FullType paramType;
      if (p.parseToken(LitToken::colon,
                       "meta parameters always require a type") ||
          p.parseType(paramType, decl, None))
        return failure();
      inputDecls.push_back(ParamDeclAttr::get(name, paramType.first));
      inputASTTypes.push_back(paramType.second);
      return success();
    };

    if (p.parseCommaSeparatedList(parseMetaParameter, LitToken::r_square) ||
        p.parseToken(LitToken::r_square, "expected ']' for parameter list"))
      return failure();
    return success();
  }

  /// Add ASTDecl objects for each declared parameter and insert them into the
  /// specified scope for the declaration, so name lookup will find them.
  void addToScope(LitSharedState &sharedState, ASTDecl &decl) {
    auto &declResolver = *sharedState.declResolver;
    // TODO: Use inputTypes.
    for (auto [paramDecl, type, loc] :
         llvm::zip(inputDecls, inputASTTypes, inputLocs)) {
      auto paramRef =
          ParamDeclRefAttr::get(paramDecl.getName(), paramDecl.getType());
      declResolver.addFullyResolvedDecl(MAValue(paramRef), paramDecl.getName(),
                                        loc, type, &decl);
    }
  }
};
} // namespace

namespace {
struct ParsedParam {
  SMLoc loc;
  StringAttr name;
  FullType type;
  ExprNode *initValue = nullptr;

  // TODO: Implement support for variadic parameter markers:
  // Python's parameter grammar embeds checking for `/` and `*` and `**` into
  // the grammar, we can just check for it using ad-hoc logic for simplicity,
  // according to the following rules:
  //   1) Only one /, *, and ** parameter may exist in the parameter list.
  //   2) They are specified in that order.
  //   3) These do not permit default arguments.
  ParseResult parse(LitParserBase &p, ASTDecl &declScope) {
    loc = p.getToken().getLoc();

    if (p.parseIdentifier(name, "expected parameter name"))
      // TODO: Scan ahead for better recovery.
      return failure();

    if (p.consumeIf(LitToken::colon)) {
      if (p.parseType(type, declScope, None))
        return failure();
    }
    if (p.consumeIf(LitToken::equal)) {
      if (p.parseExpression(initValue, None))
        return failure();
    }
    return success();
  };
};
} // namespace

/// Perform type checking for a function signature that has just been parsed
/// but that has not been installed into the specified decl.  This allows
/// magic behavior (like __new__ being static, self getting implicitly
/// declared), checking of method self requirements, inference of default object
/// argument types and enforcement of other invariants.
///
/// This returns failure after emitting an error when a type checking problem
/// is detected.
static ParseResult checkFunctionSignature(ASTDecl &declScope, Operation *op,
                                          ParsedMetaSignature &metaSignature,
                                          SmallVector<ParsedParam> &params,
                                          FullType &resultType,
                                          LitSharedState &shared) {
  auto specialFunctionKind = SpecialFunctionKind::kNormal;
  bool isStatic = false;

  // We either have a function or interface.  Functions are more general and
  // therefore have more checking to perform.
  auto funcOp = dyn_cast<LITFuncOp>(op);
  if (funcOp) {
    specialFunctionKind = funcOp.getSpecialFunctionKind();
    isStatic = funcOp.getIsStatic();
  }

  // If this definition is a struct/class member, return the self type
  // otherwise return a null type.
  ASTType selfType;
  if (auto *parentDecl = declScope.getParentDecl())
    if (isa<LITStructDeclOp>(*parentDecl)) {
      // If this is a method, the signature for the enclosing type must be
      // resolved.
      (void)shared.declResolver->resolve(
          *parentDecl, DeclResolvedness::signatureResolved,
          declScope.getCursor().getToken().getLoc());
      // The self type is stored as the resolved type.
      selfType = parentDecl->getResolvedType();
    }

  // __new__ is implicitly static.
  if (specialFunctionKind == SpecialFunctionKind::kNew) {
    assert(funcOp && "Cannot have special function generators");
    funcOp.setIsStaticAttr(mlir::UnitAttr::get(shared.getContext()));
    isStatic = true;
  }

  // If this is an instance method, enforce that self is declared correctly.
  if (selfType && !isStatic) {
    // Methods on structs (but not classes) always take the struct implicitly
    // by pointer so they can mutate it.
    // TODO: Revise this by adding mutation model.
    FullType selfFullType;
    selfFullType.second = shared.getPointerType(selfType);
    selfFullType.first = shared.getMLIRType(selfFullType.second, op->getLoc());

    // If there are no parameters, install an implicit self parameter.
    if (params.empty()) {
      params.push_back({declScope.getCursor().getToken().getLoc(),
                        StringAttr::get(shared.getContext(), "self"),
                        selfFullType,
                        /*initPtr*/ nullptr});
    }

    // Check that the first method has the right type.
    if (!params[0].type.first)
      params[0].type = selfFullType;
    if (params[0].type.first != selfFullType.first)
      return op->emitError("'self' argument must have type ")
             << selfFullType.second;
  }

  auto checkMethod = [&]() -> ParseResult {
    if (!selfType)
      return op->emitError("special function must be a method");
    return success();
  };

  auto checkInstanceMethod = [&]() -> ParseResult {
    if (checkMethod())
      return failure();
    if (isStatic)
      return op->emitError("special method may not be a static method");
    return success();
  };
  auto checkResultNoneType = [&]() -> ParseResult {
    if (!isa<KGEN::NoneType>(resultType.first))
      return op->emitError("result type must be elided (or None)");
    return success();
  };

  switch (specialFunctionKind) {
  case SpecialFunctionKind::kNormal:
    break;

  case SpecialFunctionKind::kInit:
    // __init__ must be a method and return NoneType.
    if (checkInstanceMethod() || checkResultNoneType())
      return failure();
    if (isa<LITStructDeclOp>(selfType.getDecl()))
      return op->emitError(
          "__init__ is not allowed on structs, use __new__ instead");
    break;

  case SpecialFunctionKind::kNew:
    if (checkMethod())
      return failure();
    // __new__ must return containing type.
    // TODO: We could allow omitting result type.
    if (resultType.first != shared.getMLIRType(selfType, op->getLoc()))
      return op->emitError("result type must be ") << selfType;
    break;
  }

  // If the parameter is missing a type, infer object type.
  // TODO(fn): /require/ types on parameters instead of defaulting to
  // object.
  // TODO(default args): Get the type from the default arg when present.
  for (auto &param : params) {
    if (!param.type.first) {
      param.type.second = shared.getObjectType();
      param.type.first = shared.getMLIRType(param.type.second, param.loc);
    }
  }

  return success();
}

/// funcdef ::=  [decorators] "def" identifier [meta_signature]
///              "(" [value_param_list] ")" ["->" expression] ":" suite
///
/// value_param_list  ::= value_parameter ("," value_parameter)*
/// value_parameter   ::= value_parammarker identifier_opt_type ["="
/// expression] value_parammarker ::= "/" | "*" | "**"
///
LogicalResult DeclResolver::resolveSignature(Operation *defOp, LitLexer &lexer,
                                             ASTDecl &decl) {
  LitParserBase p(lexer);

  ParsedMetaSignature metaSignature;
  SmallVector<ParsedParam> params;
  if (metaSignature.parseOptionalMetaSignature(p, decl) ||
      p.parseToken(LitToken::l_paren, "expected '(' for parameter list"))
    return failure();

  // Add the meta parameters to the symbol table.  We add all of these after
  // generic signature parsing so types used in the signature list resolve to
  // enclosing scopes, and we add them before the value signature list so the
  // types and parameters can resolve to the bound values.
  metaSignature.addToScope(sharedState, decl);

  if (!p.consumeIf(LitToken::r_paren)) {
    if (p.parseCommaSeparatedList(
            [&]() { return params.emplace_back(ParsedParam()).parse(p, decl); },
            LitToken::r_paren) ||
        p.parseToken(LitToken::r_paren, "expected ')' for parameter list"))
      return failure();
  }

  // Parse the result type if present.

  // TODO: This will be one difference between a def and fn: no result type on
  // a def should default to returning a (default initialized) Object, whereas
  // a fn can return void.  We can provide a guaranteed optimization to remove
  // it though.
  FullType resultType;
  if (p.consumeIf(LitToken::minus_greater)) {
    if (p.parseType(resultType, decl, None))
      return failure();
  } else {
    resultType = {KGEN::NoneType::get(getContext()), sharedState.getNoneType()};
  }

  if (p.parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  // Verify that methods and functions like __add__ have the right signature,
  // and adjust them if there are implicit declarations.
  if (checkFunctionSignature(decl, defOp, metaSignature, params, resultType,
                             sharedState)) {
    // If the function wasn't type checked correctly, uses of it may be
    // broken.
    decl.hasReferenceError = true;

    // Set any unspecifies argument types to error type.
    for (auto &param : params) {
      if (!param.type.first) {
        param.type.second = sharedState.getTypeCheckErrorType();
        param.type.first =
            sharedState.getMLIRType(param.type.second, param.loc);
      }
    }
  }

  // The resolvedType for a function is the return type of the function.
  decl.setResolvedType(resultType.second);

  // We have parsed the signature but skipped over the actual types, we use
  // unresolved types for now.
  SmallVector<Location> paramLocs;
  SmallVector<StringAttr> paramNames;
  SmallVector<Type> paramTypes;
  for (auto &param : params) {
    paramLocs.push_back(p.translateLocation(param.loc));
    paramNames.push_back(param.name);
    paramTypes.push_back(param.type.first);

    // TODO: add support for default parameter expressions.
    if (param.initValue)
      p.emitError(param.loc, "TODO: No default values yet");
  }

  // Interfaces are simpler than functions, process them and get out.
  if (auto interfaceOp = dyn_cast<GeneratorInterfaceOp>(defOp)) {
    auto context = getContext();
    assert(interfaceOp);
    interfaceOp.setType(
        FunctionType::get(context, paramTypes, resultType.first));
    interfaceOp.setParamDeclsAttr(
        ParamDeclArrayAttr::get(context, metaSignature.inputDecls));
    // Interface specific.
    return success();
  }

  auto funcOp = dyn_cast<LITFuncOp>(defOp);
  assert(funcOp && "defOp must be a GeneratorInterfaceOp or a LITFuncOp");

  auto builder = decl.getDeclEndBuilder();
  funcOp.setValueParamNamesAttr(builder.getAttr<StringArrayAttr>(paramNames));
  funcOp.setType(builder.getFunctionType(paramTypes, resultType.first));
  funcOp.setParamDeclsAttr(
      builder.getAttr<ParamDeclArrayAttr>(metaSignature.inputDecls));
  funcOp.getBody()->addArguments(paramTypes, paramLocs);

  if (FlatSymbolRefAttr implementsAttr = funcOp.getImplementsAttr()) {
    StringRef interfaceName = implementsAttr.getAttr().getValue();
    if (ASTDecl *interfaceDecl = decl.lookup(implementsAttr.getAttr())) {
      if (!dyn_cast_or_null<GeneratorInterfaceOp>(
              interfaceDecl->getIfOperation()))
        p.emitError(funcOp->getLoc(), "not an interface: ") << interfaceName;

      // FIXME: This needs to type check the signature here, not defer to
      // lowering.  This also needs to resolve the interface.
    } else {
      p.emitError(funcOp->getLoc(),
                  "this function implements an unknown interface: ")
          << interfaceName;
      funcOp.setImplements(llvm::None);
    }
  }

  // Set up the body of the def, creating declarations for the value
  // parameters and adding them to the symbol table.
  for (auto [arg, param] :
       llvm::zip(funcOp.getBody()->getArguments(), params)) {
    // We need to know what parameter convention that argument is passed
    // with, e.g. by-value, by-ref, by-transfer, etc.

    // FIXME: For now, hard code this based on whether it has a pointer.  This
    // will be incorrect when you want to pass a pointer by value etc.
    if (isa<POP::PointerType>(arg.getType())) {
      // Arguments passed by-reference can be directly used.
      addFullyResolvedDecl(LValue(arg), param.name, arg.getLoc(),
                           param.type.second, &decl);
      continue;
    }

    // If this was passed by-value, then create a mutable var.decl that
    // references to the name can load from.
    // TODO: This is the wrong default, reconsider this for 'fn's when we have
    // a notion of immutability.
    auto type = POP::PointerType::get(arg.getType());
    auto varDecl = builder.create<VarDeclOp>(arg.getLoc(), type, param.name);
    addFullyResolvedDecl(varDecl, param.name, param.type.second, &decl);
    builder.create<POP::StoreOp>(arg.getLoc(), arg, varDecl,
                                 /*alignment*/ None);
  }

  return success();
}

ParseResult DeclResolver::resolveBody(LITFuncOp defOp, LitLexer &lexer,
                                      ASTDecl &decl) {
  if (LitParserBase::parseSuite(decl, lexer))
    return failure();

  // Check to see if we have a kgen.return at the end of function.  If not,
  // complain or add one implicitly if we have no results.
  Block *bodyBlock = defOp.getBody();
  if (bodyBlock->empty() || !isa<ReturnOp>(bodyBlock->back())) {
    auto loc = decl.getLoc();
    if (isa<KGEN::NoneType>(defOp.getResultType()) &&
        defOp.getResultParamTypes().empty()) {
      auto b = OpBuilder::atBlockEnd(bodyBlock);

      auto noneAttr = b.getType<NoneAttr>(defOp.getResultType());
      Value noneVal = b.create<ParamConstantOp>(loc, noneAttr);
      b.create<ReturnOp>(loc, ArrayRef<TypedAttr>(), noneVal);
    } else if (!sharedState.errorOccurred) {
      Location endLoc = bodyBlock->empty() ? loc : bodyBlock->back().getLoc();
      emitError(endLoc, "return expected at end of 'def' with results");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Variable Decl implementation
//===----------------------------------------------------------------------===//

/// var_decl_stmt ::= "var" identifier ":" expression ["=" expression]
///                 | "var" identifier "=" expression [TODO]
///
LogicalResult DeclResolver::resolveSignature(VarDeclOp varOp, LitLexer &lexer,
                                             ASTDecl &decl) {
  LitParserBase p(lexer);
  FullType type;
  ExprNode *initValue = nullptr;
  // Parse the type if present.
  // TODO: Make type optional.
  if (p.parseToken(LitToken::colon, "var declaration requires a type") ||
      p.parseType(type, decl, decl.getIndentation()))
    return failure();

  varOp.getResult().setType(POP::PointerType::get(type.first));

  if (p.consumeIf(LitToken::equal)) {
    p.emitError("var initializers not supported yet");
    if (p.parseExpression(initValue, decl.getIndentation()))
      return failure();
  }

  // The resolvedType of a variable declaration is the type of the decl.
  decl.setResolvedType(type.second);
  return success();
}

ParseResult DeclResolver::resolveBody(VarDeclOp op, LitLexer &lexer,
                                      ASTDecl &decl) {
  // Nothing to do for a var decl, we parse everything as part of its
  // signature. We could move to parsing an initializer expression lazily when
  // a type is present if there were a reason to do that (e.g. more laziness
  // desired) in the future.
  return success();
}

//===----------------------------------------------------------------------===//
// Struct Decl implementation
//===----------------------------------------------------------------------===//

/// structdef ::=
///   [decorators] "struct" identifier [meta_signature] ":" suite
///
LogicalResult DeclResolver::resolveSignature(LITStructDeclOp structOp,
                                             LitLexer &lexer, ASTDecl &decl) {
  LitParserBase p(lexer);

  ParsedMetaSignature metaSignature;
  if (metaSignature.parseOptionalMetaSignature(p, decl) ||
      p.parseToken(LitToken::colon, "expected ':' in struct definition"))
    return failure();

  structOp.setParamDeclsAttr(
      ParamDeclArrayAttr::get(getContext(), metaSignature.inputDecls));

  // Add the meta parameters to the struct's symbol table.
  metaSignature.addToScope(sharedState, decl);

  // This is a struct, so we can use 'computeSelfTypeForStruct' to figure out
  // the self type.
  decl.setResolvedType(decl.computeSelfTypeForStruct(sharedState));
  return success();
}

ParseResult DeclResolver::resolveBody(LITStructDeclOp structOp, LitLexer &lexer,
                                      ASTDecl &decl) {
  return LitParserBase::parseSuite(decl, lexer);
}
