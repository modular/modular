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
#include "LitExprs.h"
#include "LitLexer.h"
#include "LitParserBase.h"
#include "LitScope.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// Scope
//===----------------------------------------------------------------------===//

static Location getLocationFrom(Scope::NameEntry entry) {
  if (std::holds_alternative<Scope *>(entry))
    return std::get<Scope *>(entry)->getDecl()->getLoc();
  return std::get<Scope::MetaParameterValue>(entry).loc;
}

static void markErroneous(Scope::NameEntry value) {
  if (std::holds_alternative<Scope *>(value))
    std::get<Scope *>(value)->hasReferenceError = true;
}

/// Add the specified declaration to the current scope, emitting an error on
/// a name collision.
void Scope::addToScope(StringAttr name, MetaParameterValue newValue,
                       LitSharedState &sharedState) {
  auto [it, inserted] = decls.insert({name, newValue});
  if (inserted)
    return;
  Scope::NameEntry &entry = it->second;

  auto diag = emitError(newValue.loc, "invalid redefinition of ") << name;
  diag.attachNote(getLocationFrom(entry)) << "previous definition here";
  sharedState.errorOccurred = true;

  // If the existing entry was a declaration, mark it as erroneous so uses of it
  // don't create confusing errors.
  markErroneous(entry);
}

void Scope::addToScope(Scope *newDeclScope, LitSharedState &sharedState) {
  StringAttr name;
  Operation *newDecl = newDeclScope->getDecl();

  TypeSwitch<Operation *>(newDecl)
      .Case<VarDeclOp, LITFuncOp, LITStructDeclOp>(
          [&](auto op) { name = op.getNameAttr(); })
      .Default([&](auto attr) {
        assert(isa<ModuleOp>(newDecl) && "Unknown declaration kind");
      });

  if (!name) // Don't add for modules.
    return;

  auto [it, inserted] = decls.insert({name, newDeclScope});
  if (inserted)
    return;
  Scope::NameEntry &entry = it->second;

  auto diag = emitError(newDecl->getLoc(), "invalid redefinition of ") << name;
  diag.attachNote(getLocationFrom(entry)) << "previous definition here";
  sharedState.errorOccurred = true;

  // If the existing entry was a declaration, mark it as erroneous so uses of it
  // don't create confusing errors.
  newDeclScope->hasReferenceError = true;
  markErroneous(entry);
}

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

// Declarations (e.g. module, class, function) are parsed in multiple phases to
// increase laziness of the parse as well as make circular references possible.
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
  // Run the destructors on all the scope objects to make sure any transitively
  // allocated data is released.
  for (Scope *scope : parsedDeclList)
    scope->~Scope();
}

/// Add a new declaration that needs to be resolved.
Scope &DeclResolver::addDecl(Operation *decl, Scope *parentScope,
                             LitLexerCursor cursor, LitLexerCursor endCursor,
                             ssize_t indentation) {
  void *rawScopePtr =
      sharedState.persistentAllocator.Allocate(sizeof(Scope), alignof(Scope));
  Scope *scope = new (rawScopePtr)
      Scope(decl, parentScope, cursor, endCursor, indentation);
  parsedDeclList.push_back(scope);

  // If this is a type definition, remember in in a special table so we can look
  // up references from attributes.
  if (isa<LITStructDeclOp>(decl))
    typeSymbolScopes[SymbolTable::getSymbolName(decl)] = scope;

  if (parentScope)
    parentScope->addToScope(scope, sharedState);

  return *scope;
}

Scope &DeclResolver::addFullyResolvedDecl(Operation *decl, Scope *parentScope) {
  auto &scope =
      addDecl(decl, parentScope, LitLexerCursor(), LitLexerCursor(), 0);
  scope.resolvedness = DeclResolvedness::fullyResolved;
  return scope;
}

/// If the specified type is a RefType that resolves to a (possibly
/// parameterized) type, return the scope for the type and the parameters in
/// the reference.  This returns null on error.
std::pair<Scope *, ParamBindArrayAttr>
DeclResolver::getScopeAndParamsFromType(Type type) {
  auto refType = dyn_cast<RefType>(type);
  if (!refType)
    return {};

  auto it = typeSymbolScopes.find(refType.getName().getAttr());
  if (it == typeSymbolScopes.end())
    return {};
  return {it->second, refType.getParamValues()};
}

/// Resolve all of the declarations that are visible.
void DeclResolver::resolveAll(SMLoc loc) {
  // We can do this in any order, but choose to use the order they are
  // discovered so diagnostics are mostly top-down.  Resolving declarations may
  // cause more entries to be added to this list.
  for (size_t i = 0; i != parsedDeclList.size(); ++i)
    (void)resolve(*parsedDeclList[i], DeclResolvedness::fullyResolved, loc);
}

/// Resolve the specified declaration to at least the specified level of
/// resolution, performing incremental type checking as appropriate.
LogicalResult DeclResolver::resolve(Scope &scope, DeclResolvedness howResolved,
                                    SMLoc loc) {
  // If scope is already resolved enough, we're done.
  if (scope.resolvedness >= howResolved) {
    // If decl is busted, then return failure.
    return success(!scope.hasReferenceError);
  }

  Operation *decl = scope.getDecl();

  // If we are currently name binding this operation, we found a cycle, reject
  // it with an error.
  if (!declsCurrentlyProcessing.insert(decl).second) {
    emitError(sharedState.translateLocation(loc),
              "recursive reference to declaration");
    return failure();
  }

  // If the signature hasn't been parsed, do so.
  if (scope.resolvedness < DeclResolvedness::signatureResolved) {
    // Handle each operation that can be name bound.  We handle this by
    // restoring the lexer to the position where parsing can continue, calling
    // the `resolveSignature` method for the op, and re-saving the new cursor
    // for the next stage of resolution.
    TypeSwitch<Operation *>(decl)
        .Case<LITFuncOp, LITStructDeclOp, VarDeclOp>([&](auto op) {
          LitLexer lexer(sharedState, scope.getCursor());

          // Resolve the signature: on a parse error, we note that the decl is
          // malformed and should not be referenced to silence downstream
          // errors.
          if (failed(resolveSignature(op, lexer, scope)))
            scope.hasReferenceError = true;
          scope.getCursor() = lexer.getCursor();
        })
        .Case([&](ModuleOp op) { /*Nothing*/ })
        .Default([&](auto attr) {
          decl->emitError(
              "do not know how to resolve the signature of this decl!");
        });
    scope.resolvedness = DeclResolvedness::signatureResolved;
  }

  // If the declaration hasn't been fully parsed and we need to, do so.
  if (scope.resolvedness < DeclResolvedness::fullyResolved &&
      howResolved == DeclResolvedness::fullyResolved) {
    // Handle each operation that can be name bound.
    TypeSwitch<Operation *>(decl)
        .Case<LITFuncOp, LITStructDeclOp, VarDeclOp>([&](auto op) {
          // Parse the body of the declaration from the correct point.
          LitLexer lexer(sharedState, scope.getCursor());
          if (resolveBody(op, lexer, scope))
            return;

          // If the final parse of the declaration didn't match the initial
          // parse, report an error about unrecognized tokens at end of
          // declaration.
          if (!scope.isMatchingEndCursor(lexer.getCursor()) &&
              !scope.hasReferenceError) {
            if (lexer.getToken().isAny(LitToken::kw_def, LitToken::kw_struct,
                                       LitToken::kw_class, LitToken::kw_var))
              lexer.emitError("definition isn't on its own line at the correct "
                              "indentation");
            else
              lexer.emitError("unknown tokens at the end of a declaration");
          }
        })
        .Case<ModuleOp>([&](auto op) { /*Nothing*/ })
        .Default([&](auto attr) {
          decl->emitError("do not know how to resolve the body of this decl!");
        });
    scope.resolvedness = DeclResolvedness::fullyResolved;
  }

  declsCurrentlyProcessing.erase(decl);
  // If decl is busted, then return failure.
  return success(!scope.hasReferenceError);
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
  std::vector<Location> inputLocs;

  ParseResult parseOptionalMetaSignature(LitParserBase &p, Scope &scope) {
    if (!p.consumeIf(LitToken::l_square) || p.consumeIf(LitToken::r_square))
      return success();

    auto parseMetaParameter = [&]() -> ParseResult {
      inputLocs.push_back(p.getTokenLocation());

      StringAttr name;
      if (p.parseIdentifier(name, "expected parameter name")) {
        // TODO: Scan ahead for better recovery.
        return failure();
      }

      Type paramType;
      if (p.parseToken(LitToken::colon,
                       "meta parameters always require a type") ||
          p.parseType(paramType, scope, None))
        return failure();
      inputDecls.push_back(ParamDeclAttr::get(name, paramType));
      return success();
    };

    if (p.parseCommaSeparatedList(parseMetaParameter) ||
        p.parseToken(LitToken::r_square, "expected ']' for parameter list"))
      return failure();
    return success();
  }

  void addToScope(LitSharedState &sharedState, Scope &scope) {
    for (auto [param, loc] : llvm::zip(inputDecls, inputLocs)) {
      auto value = ParamDeclRefAttr::get(param.getName(), param.getType());
      scope.addToScope(param.getName(), Scope::MetaParameterValue{value, loc},
                       sharedState);
    }
  }
};
} // namespace

namespace {
struct ParsedParam {
  SMLoc loc;
  StringAttr name;
  Type type;
  ExprNode *initValue = nullptr;

  // TODO: Implement support for variadic parameter markers:
  // Python's parameter grammar embeds checking for `/` and `*` and `**` into
  // the grammar, we can just check for it using ad-hoc logic for simplicity,
  // according to the following rules:
  //   1) Only one /, *, and ** parameter may exist in the parameter list.
  //   2) They are specified in that order.
  //   3) These do not permit default arguments.
  ParseResult parse(LitParserBase &p, Scope &scope) {
    loc = p.getToken().getLoc();

    if (p.parseIdentifier(name, "expected parameter name"))
      // TODO: Scan ahead for better recovery.
      return failure();

    if (p.consumeIf(LitToken::colon)) {
      if (p.parseType(type, scope, None))
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

/// Perform additional type checking for a function signature that has just been
/// parsed but that has not been installed into the specified decl.  This allows
/// magic behavior (like __new__ being static, self getting implicitly declared)
/// and enforcement of other invariants.
///
/// This returns failure after emitting an error when a type checking problem is
/// detected.
static ParseResult checkSpecialFunctionSignature(
    Scope &declScope, LITFuncOp defDecl, ParsedMetaSignature &metaSignature,
    SmallVector<ParsedParam> &params, Type &resultType) {

  // This lambda is used by all special functions that are known to be instance
  // methods.
  auto checkInstanceMethod = [&]() -> ParseResult {
    // Get the context of the declaration, rejecting it if it isn't nested in a
    // structure.
    Scope *parent = declScope.getParentScope();
    if (!parent || !isa<LITStructDeclOp>(parent->getDecl()))
      return defDecl.emitError("special function must be a method");
    auto parentStruct = cast<LITStructDeclOp>(parent->getDecl());

    // Figure out the expected type of self.
    if (!parentStruct.getParamDecls().empty())
      return defDecl.emitError(
          "cannot (yet) compute self type in parametric struct");
    ParamBindArrayAttr selfParams =
        ParamBindArrayAttr::get(defDecl.getContext(), {});
    Type selfType = RefType::get(
        FlatSymbolRefAttr::get(parentStruct.getNameAttr()), selfParams);

    // If there are no parameters, install an implicit self parameter.
    if (params.empty()) {
      params.push_back({declScope.getCursor().getToken().getLoc(),
                        StringAttr::get(defDecl.getContext(), "self"), selfType,
                        /*initPtr*/ nullptr});
      return success();
    }

    // Check that the first method has the right type.
    if (!params[0].type)
      params[0].type = selfType;
    if (params[0].type != selfType)
      return defDecl.emitError("'self' argument must have type ") << selfType;

    return success();
  };

  switch (defDecl.getSpecialFunctionKind()) {
  case SpecialFunctionKind::kNormal:
    return success();
  case SpecialFunctionKind::kInit:
    // __init__ must be a method, no other constraints.
    return checkInstanceMethod();
  }
  llvm_unreachable("Unknown special function kind");
}

/// funcdef ::=  [decorators] "def" identifier [meta_signature]
///              "(" [value_param_list] ")" ["->" expression] ":" suite
///
/// value_param_list  ::= value_parameter ("," value_parameter)*
/// value_parameter   ::= value_parammarker identifier_opt_type ["=" expression]
/// value_parammarker ::= "/" | "*" | "**"
///
LogicalResult DeclResolver::resolveSignature(LITFuncOp defDecl, LitLexer &lexer,
                                             Scope &scope) {
  LitParserBase p(lexer);

  ParsedMetaSignature metaSignature;
  SmallVector<ParsedParam> params;
  Type resultType;

  if (metaSignature.parseOptionalMetaSignature(p, scope) ||
      p.parseToken(LitToken::l_paren, "expected '(' for parameter list"))
    return failure();

  // Add the meta parameters to the symbol table.  We add all of these after
  // generic signature parsing so types used in the signature list resolve to
  // enclosing scopes, and we add them before the value signature list so the
  // types and parameters can resolve to the bound values.
  metaSignature.addToScope(sharedState, scope);

  if (!p.consumeIf(LitToken::r_paren)) {
    if (p.parseCommaSeparatedList([&]() {
          return params.emplace_back(ParsedParam()).parse(p, scope);
        }) ||
        p.parseToken(LitToken::r_paren, "expected ')' for parameter list"))
      return failure();
  }

  // Parse the result type if present.

  // TODO: This will be one difference between a def and fn: no result type on
  // a def should default to returning a (default initialized) Object, whereas
  // a fn can return void.  We can provide a guaranteed optimization to remove
  // it though.
  if (p.consumeIf(LitToken::minus_greater)) {
    if (p.parseType(resultType, scope, None))
      return failure();
  }

  if (p.parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  // Verify that functions like __add__ have the right signature, and adjust
  // them if there are implicit declarations.
  if (checkSpecialFunctionSignature(scope, defDecl, metaSignature, params,
                                    resultType)) {
    // If the special function wasn't type checked correctly, then any uses of
    // it may be broken.
    scope.hasReferenceError = true;
  }

  auto builder = scope.getDeclEndBuilder();

  // We have parsed the signature but skipped over the actual types, we use
  // unresolved types for now.
  SmallVector<Location> paramLocs;
  SmallVector<StringAttr> paramNames;
  SmallVector<Type> paramTypes;
  for (auto &param : params) {
    paramLocs.push_back(p.translateLocation(param.loc));
    paramNames.push_back(param.name);

    // If the parameter is missing a type, infer object type.
    // TODO(fn): /require/ types on parameters instead of defaulting to object.
    // TODO: I think there are some other special cases to evaluate, e.g. "self"
    // arguments should be containing type in methods?
    // TODO(default args): Get the type from the default arg when present.
    if (!param.type)
      param.type = builder.getType<ObjectType>();
    paramTypes.push_back(param.type);

    // TODO: add support for default parameter expressions.
    if (param.initValue)
      p.emitError(param.initValue->getLoc(), "TODO: No default values yet");
  }

  SmallVector<Type> resultTypes;
  if (resultType)
    resultTypes.push_back(resultType);

  defDecl.setValueParamNamesAttr(
      StringArrayAttr::get(getContext(), paramNames));
  defDecl.setType(builder.getFunctionType(paramTypes, resultTypes));
  defDecl.setParamDeclsAttr(
      ParamDeclArrayAttr::get(getContext(), metaSignature.inputDecls));
  defDecl.getBody()->addArguments(paramTypes, paramLocs);

  // Set up the body of the def, creating declarations for the value parameters
  // and adding them to the symbol table.
  for (auto [arg, name] : llvm::zip(defDecl.getBody()->getArguments(),
                                    defDecl.getValueParamNames())) {
    // Create a mutable var.decl that references to the name can load from.
    // TODO: This is the wrong default, reconsider this for 'fn's when we have
    // a notion of immutability.
    auto type = POP::PointerType::get(arg.getType());
    auto varDecl = builder.create<VarDeclOp>(arg.getLoc(), type, name);
    addFullyResolvedDecl(varDecl, &scope);
    builder.create<POP::StoreOp>(arg.getLoc(), arg, varDecl,
                                 /*alignment*/ None);
  }

  return success();
}

ParseResult DeclResolver::resolveBody(LITFuncOp defDecl, LitLexer &lexer,
                                      Scope &scope) {
  if (LitParserBase::parseSuite(scope, lexer))
    return failure();

  // Check to see if we have a kgen.return at the end of function.  If not,
  // complain or add one implicitly if we have no results.
  Block *bodyBlock = defDecl.getBody();
  if (bodyBlock->empty() || !isa<ReturnOp>(bodyBlock->back())) {
    if (defDecl.getResultTypes().empty() &&
        defDecl.getResultParamTypes().empty()) {
      // TODO: Generalize lit.func.
      OpBuilder::atBlockEnd(bodyBlock).create<ReturnOp>(
          defDecl->getLoc(), ArrayRef<TypedAttr>(), ArrayRef<Value>());
    } else if (!sharedState.errorOccurred) {
      Location endLoc =
          bodyBlock->empty() ? defDecl.getLoc() : bodyBlock->back().getLoc();
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
LogicalResult DeclResolver::resolveSignature(VarDeclOp varDecl, LitLexer &lexer,
                                             Scope &scope) {
  LitParserBase p(lexer);
  Type type;
  ExprNode *initValue = nullptr;
  // Parse the type if present.
  // TODO: Make type optional.
  if (p.parseToken(LitToken::colon, "var declaration requires a type") ||
      p.parseType(type, scope, scope.getIndentation()))
    return failure();

  varDecl.getResult().setType(POP::PointerType::get(type));

  if (p.consumeIf(LitToken::equal)) {
    p.emitError("var initializers not supported yet");
    if (p.parseExpression(initValue, scope.getIndentation()))
      return failure();
  }
  return success();
}

ParseResult DeclResolver::resolveBody(VarDeclOp op, LitLexer &lexer,
                                      Scope &scope) {
  // Nothing to do for a var decl, we parse everything as part of its signature.
  // We could move to parsing an initializer expression lazily when a type is
  // present if there were a reason to do that (e.g. more laziness desired) in
  // the future.
  return success();
}

//===----------------------------------------------------------------------===//
// Struct Decl implementation
//===----------------------------------------------------------------------===//

/// structdef ::=
///   [decorators] "struct" identifier [meta_signature] ":" suite
///
LogicalResult DeclResolver::resolveSignature(LITStructDeclOp structDecl,
                                             LitLexer &lexer, Scope &scope) {
  LitParserBase p(lexer);

  ParsedMetaSignature metaSignature;
  if (metaSignature.parseOptionalMetaSignature(p, scope) ||
      p.parseToken(LitToken::colon, "expected ':' in struct definition"))
    return failure();

  structDecl.setParamDeclsAttr(
      ParamDeclArrayAttr::get(getContext(), metaSignature.inputDecls));

  // Add the meta parameters to the struct's symbol table.
  metaSignature.addToScope(sharedState, scope);
  return success();
}

ParseResult DeclResolver::resolveBody(LITStructDeclOp op, LitLexer &lexer,
                                      Scope &scope) {
  return LitParserBase::parseSuite(scope, lexer);
}
