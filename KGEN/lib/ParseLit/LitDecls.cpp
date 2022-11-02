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
#include "LitASTDecl.h"
#include "LitExprs.h"
#include "LitLexer.h"
#include "LitParserBase.h"

#include "KGEN/KGENDialect/KGENOps.h"
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
// ASTType
//===----------------------------------------------------------------------===//

ASTType::ASTType(ASTDecl *decl) : decl(decl) {
  assert(decl && "cannot create ASTType with null decl");
  paramValues = ParamBindArrayAttr::get(decl->getContext(), {});
}

ASTType::ASTType(ASTDecl *decl, ParamBindArrayAttr attrs)
    : decl(decl), paramValues(attrs) {}

ParamBindArrayAttr ASTType::getParamValues() const {
  return cast<ParamBindArrayAttr>(paramValues);
}

//===----------------------------------------------------------------------===//
// ASTDecl
//===----------------------------------------------------------------------===//

/// If this is a ParamDecl, return it otherwise return null.
ParamDeclAttr ASTDecl::getParamDecl() const {
  auto attr = dyn_cast_or_null<Attribute>(irDecl);
  return attr ? cast<ParamDeclAttr>(attr) : ParamDeclAttr();
}

/// Given a type declaration, return a RefType for a reference to this with
/// the specified type parameters.  This aborts if the current decl isn't a
/// type.
std::pair<Type, ASTType> ASTDecl::getFullTypeForTypeReference() {
  auto astType = getResolvedType();
  auto structOp = cast<LITStructDeclOp>(*this);
  auto mlirType = RefType::get(FlatSymbolRefAttr::get(structOp.getNameAttr()),
                               astType.getParamValues());

  return {mlirType, astType};
}

/// Given an MLIR op for a struct declaration, return the self type.
ASTType ASTDecl::computeSelfTypeForStruct() {
  auto structOp = cast<LITStructDeclOp>(*this);

  SmallVector<ParamBindAttr> parameters;
  for (auto decl : structOp.getParamDecls()) {
    // We're using the parameter from the type declaration scope in the
    // parameter binding list.
    auto ref = ParamDeclRefAttr::get(decl.getName(), decl.getType());
    parameters.push_back(ParamBindAttr::get(decl.getName(), ref));
  }

  ParamBindArrayAttr selfParams =
      ParamBindArrayAttr::get(structOp.getContext(), parameters);

  // Methods on structs (but not classes) take the struct implicitly by
  // pointer so they can use and mutate it.
  return ASTType(this, selfParams);
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
  // Run the destructors on all the ASTDecl objects to make sure any
  // transitively allocated data is released.
  for (ASTDecl *decl : parsedDeclList)
    decl->~ASTDecl();
}

/// Add a new declaration that needs to be resolved.
ASTDecl &DeclResolver::addDecl(PointerUnion<Operation *, Attribute> irDecl,
                               Location loc, StringAttr name,
                               ASTDecl *parentDecl, LitLexerCursor cursor,
                               LitLexerCursor endCursor, ssize_t indentation) {
  void *rawDeclPtr = sharedState.persistentAllocator.Allocate(sizeof(ASTDecl),
                                                              alignof(ASTDecl));
  ASTDecl *decl = new (rawDeclPtr)
      ASTDecl(irDecl, loc, parentDecl, cursor, endCursor, indentation);
  parsedDeclList.push_back(decl);

  // If this is a type definition, remember in in a special table so we can look
  // up references from attributes.
  if (auto structDecl = dyn_cast<LITStructDeclOp>(*decl))
    typeSymbolDecls[structDecl.getNameAttr()] = decl;

  // If this has a parent and a name, insert it into the parents name table so
  // name lookup will resolve it.
  if (!parentDecl || !name)
    return *decl;

  auto [it, inserted] = parentDecl->declsInScope.insert({name, decl});
  if (!inserted) {
    ASTDecl *existing = it->second;
    auto diag = emitError(decl->getLoc(), "invalid redefinition of ") << name;
    diag.attachNote(existing->getLoc()) << "previous definition here";
    sharedState.errorOccurred = true;

    // Mark the existing decl and this one as erroneous so uses of either
    // don't create confusing errors.
    decl->hasReferenceError = true;
    existing->hasReferenceError = true;
  }
  return *decl;
}

/// Add a new declaration that needs to be resolved.
ASTDecl &DeclResolver::addDecl(Operation *op, ASTDecl *parentDecl,
                               LitLexerCursor cursor, LitLexerCursor endCursor,
                               ssize_t indentation) {
  // Get the name for the entity.
  StringAttr name;
  TypeSwitch<Operation *>(op)
      .Case<VarDeclOp, LITFuncOp, LITStructDeclOp>(
          [&](auto op) { name = op.getNameAttr(); })
      .Case([&](ModuleOp op) {})
      .Default(
          [&](auto attr) { llvm_unreachable("Unknown declaration kind"); });

  return addDecl(op, op->getLoc(), name, parentDecl, cursor, endCursor,
                 indentation);
}

ASTDecl &DeclResolver::addFullyResolvedDecl(Operation *op, ASTType type,
                                            ASTDecl *parentDecl) {
  auto &decl = addDecl(op, parentDecl, LitLexerCursor(), LitLexerCursor(), 0);
  decl.resolvedness = DeclResolvedness::fullyResolved;
  decl.setResolvedType(type);
  return decl;
}

/// Add a declaration that is already fully resolved.
ASTDecl &DeclResolver::addFullyResolvedDecl(ParamDeclAttr attr, Location loc,
                                            ASTType type, ASTDecl *parentDecl) {
  auto &decl = addDecl(attr, loc, attr.getName(), parentDecl, LitLexerCursor(),
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
  auto &decl = addDecl(ParamDeclAttr(), parentDecl->getLoc(),
                       StringAttr::get(getContext(), name), parentDecl,
                       LitLexerCursor(), LitLexerCursor(), 0);
  decl.resolvedness = DeclResolvedness::fullyResolved;
  decl.magicKind = kind;
  return decl;
}

/// If the specified type is a RefType that resolves to a (possibly
/// parameterized) type, return the decl for the type and the parameters in
/// the reference.  This returns null on error.
std::pair<ASTDecl *, ParamBindArrayAttr>
DeclResolver::getDeclAndParamsFromType(Type type) {
  auto refType = dyn_cast<RefType>(type);
  if (!refType)
    return {};

  auto it = typeSymbolDecls.find(refType.getName().getAttr());
  if (it == typeSymbolDecls.end())
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
        .Case<LITFuncOp, LITStructDeclOp, VarDeclOp>([&](auto op) {
          LitLexer lexer(sharedState, decl.getCursor());

          // Resolve the signature: on a parse error, we note that the decl is
          // malformed and should not be referenced to silence downstream
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

    if (p.parseCommaSeparatedList(parseMetaParameter) ||
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
         llvm::zip(inputDecls, inputASTTypes, inputLocs))
      declResolver.addFullyResolvedDecl(paramDecl, loc, type, &decl);
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

/// Perform type checking for a function signature that has just been parsed but
/// that has not been installed into the specified decl.  This allows magic
/// behavior (like __new__ being static, self getting implicitly declared),
/// checking of method self requirements and enforcement of other invariants.
///
/// This returns failure after emitting an error when a type checking problem is
/// detected.
static ParseResult checkFunctionSignature(ASTDecl &declScope, LITFuncOp defDecl,
                                          ParsedMetaSignature &metaSignature,
                                          SmallVector<ParsedParam> &params,
                                          FullType &resultType,
                                          LitSharedState &shared) {

  // If this definition is a struct/class member, return the self type otherwise
  // return a null type.
  FullType selfType;
  if (auto *parentDecl = declScope.getParentDecl())
    if (isa<LITStructDeclOp>(*parentDecl)) {
      // If this is a method, the signature for the enclosing type must be
      // resolved.
      (void)shared.declResolver->resolve(
          *parentDecl, DeclResolvedness::signatureResolved,
          declScope.getCursor().getToken().getLoc());
      selfType = parentDecl->getFullTypeForTypeReference();
    }

  // If this is a method, enforce that self is declared correctly.
  if (selfType.first) {
    // Methods on structs (but not classes) take the struct implicitly by
    // pointer so they can use and mutate it.

    selfType.first = POP::PointerType::get(selfType.first);

    // If there are no parameters, install an implicit self parameter.
    if (params.empty()) {
      params.push_back({declScope.getCursor().getToken().getLoc(),
                        StringAttr::get(defDecl.getContext(), "self"), selfType,
                        /*initPtr*/ nullptr});
    }

    // Check that the first method has the right type.
    if (!params[0].type.first)
      params[0].type = selfType;
    if (params[0].type.first != selfType.first) {
      // TODO(pretty types).
      return defDecl.emitError("'self' argument must have type ")
             << selfType.first;
    }
  }

  // This lambda verifies the decl is a method.
  auto checkInstanceMethod = [&]() -> ParseResult {
    if (!selfType.first)
      return defDecl.emitError("special function must be a method");
    return success();
  };
  auto checkResultNoneType = [&]() -> ParseResult {
    if (!isa<KGEN::NoneType>(resultType.first))
      return defDecl.emitError("result type must be elided (or None)");
    return success();
  };

  switch (defDecl.getSpecialFunctionKind()) {
  case SpecialFunctionKind::kNormal:
    return success();
  case SpecialFunctionKind::kInit:
    // __init__ must be a method and return NoneType.
    if (checkInstanceMethod() || checkResultNoneType())
      return failure();
    return success();
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
LogicalResult DeclResolver::resolveSignature(LITFuncOp defOp, LitLexer &lexer,
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
    if (p.parseCommaSeparatedList([&]() {
          return params.emplace_back(ParsedParam()).parse(p, decl);
        }) ||
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
    resultType = {KGEN::NoneType::get(getContext()),
                  ASTType(sharedState.noneDecl)};
  }
  // The resolvedType for a function is the return type of the function.
  decl.setResolvedType(resultType.second);

  if (p.parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  // Verify that methods and functions like __add__ have the right signature,
  // and adjust them if there are implicit declarations.
  if (checkFunctionSignature(decl, defOp, metaSignature, params, resultType,
                             sharedState)) {
    // If the function wasn't type checked correctly, uses of it may be broken.
    decl.hasReferenceError = true;
  }

  auto builder = decl.getDeclEndBuilder();

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
    if (!param.type.first)
      param.type = sharedState.objectDecl->getFullTypeForTypeReference();
    paramTypes.push_back(param.type.first);

    // TODO: add support for default parameter expressions.
    if (param.initValue)
      p.emitError(param.initValue->getLoc(), "TODO: No default values yet");
  }

  defOp.setValueParamNamesAttr(StringArrayAttr::get(getContext(), paramNames));
  defOp.setType(builder.getFunctionType(paramTypes, resultType.first));
  defOp.setParamDeclsAttr(
      ParamDeclArrayAttr::get(getContext(), metaSignature.inputDecls));
  defOp.getBody()->addArguments(paramTypes, paramLocs);

  // Set up the body of the def, creating declarations for the value parameters
  // and adding them to the symbol table.
  for (auto [arg, param] : llvm::zip(defOp.getBody()->getArguments(), params)) {
    // Create a mutable var.decl that references to the name can load from.
    // TODO: This is the wrong default, reconsider this for 'fn's when we have
    // a notion of immutability.
    auto type = POP::PointerType::get(arg.getType());
    auto varDecl = builder.create<VarDeclOp>(arg.getLoc(), type, param.name);
    addFullyResolvedDecl(varDecl, param.type.second, &decl);
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
      Value noneVal = b.create<NoneValueOp>(loc);
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
  decl.setResolvedType(decl.computeSelfTypeForStruct());
  return success();
}

ParseResult DeclResolver::resolveBody(LITStructDeclOp structOp, LitLexer &lexer,
                                      ASTDecl &decl) {
  return LitParserBase::parseSuite(decl, lexer);
}
