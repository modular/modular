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
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "LitExprNodes.h"
#include "LitParserBase.h"
#include "LitScope.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace M::KGEN::LIT;

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
  for (auto [op, scope] : parsedDecls)
    scope->~Scope();
}

/// Add a new declaration that needs to be resolved.
Scope &DeclResolver::addDecl(Operation *decl, Scope *parentScope,
                             LitLexerCursor cursor) {
  void *rawScopePtr =
      sharedState.persistentAllocator.Allocate(sizeof(Scope), alignof(Scope));
  Scope *scope = new (rawScopePtr) Scope(decl, parentScope, cursor);

  parsedDeclList.push_back(decl);
  parsedDecls[decl] = scope;
  return *scope;
}

/// Resolve all of the declarations that are visible.
void DeclResolver::resolveAll() {
  // We can do this in any order, but choose to use the order they are
  // discovered so diagnostics are mostly top-down.  Resolving declarations may
  // cause more entries to be added to this list.
  for (size_t i = 0; i != parsedDeclList.size(); ++i)
    resolve(*parsedDecls[parsedDeclList[i]]);
}

void DeclResolver::resolve(Scope &scope) {
  // If scope is fully resolved, we're done.
  if (scope.getIsResolved())
    return;

  Operation *decl = scope.getDecl();

  // If we are currently name binding this operation, we found a cycle, reject
  // it with an error.
  if (!declsCurrentlyProcessing.insert(decl).second) {
    assert(0 &&
           "FIXME: Diagnose cyclic reference when it is possible to happen");
  }

  // Handle each operation that can be name bound.
  TypeSwitch<Operation *>(decl)
      .Case<LITFuncOp>([&](auto op) { resolveBody(op, scope); })
      .Case<LITStructDeclOp>([&](auto op) { resolveBody(op, scope); })
      .Case<VarDeclOp>([&](auto op) { resolveSignature(op, scope); })
      .Case<ModuleOp>([&](auto op) { /*Nothing*/ })
      .Default([&](auto attr) {
        decl->emitError("do not know how to perform name binding on this op!");
      });

  declsCurrentlyProcessing.erase(decl);
  scope.setIsResolved();
}

/// Given a cursor location for a type expression that correctly parsed in the
/// first pass, reparse it into an expression and resolve it into a type by
/// performing name lookup and other resolution.  This can produce errors, but
/// always returns a non-null type.
Type DeclResolver::resolveType(LitLexerCursor cursor, Scope &scope,
                               LitParserBase &parser) {
  // FIXME: This is the wrong design, shouldn't be reparsing.

  // Move the cursor to the specified location.
  cursor.restore(parser.getLexer());
  // Re-parse the expression at that location.
  ExprParser exprParser(parser);
  ExprNode *typeExpr = exprParser.parseExpression();
  assert(typeExpr && "We know expr parsing will work");

  auto emitError = [&](const Twine &message) -> Type {
    parser.emitError(typeExpr->getLoc(), message);
    return UnresolvedType::get(parser.getContext());
  };

  // TODO: Make this a recursive walk when we have more interesting types.
  if (auto dre = dyn_cast<DeclRefNode>(typeExpr)) {
    // TODO(types): This is a hack to unblock tests in the interim.
    if (dre->spelling == "index")
      return IndexType::get(parser.getContext());

    // Lookup the identifier.
    Optional<Scope::ScopeValue> lookup = scope.lookup(dre->spelling);
    if (!lookup)
      return emitError("unknown type name '" + dre->spelling + "'");
    if (std::holds_alternative<VarDeclOp>(*lookup))
      return emitError("'" + dre->spelling + "' names a value, not a type");
    auto attr = dyn_cast<SymbolConstantAttr>(
        std::get<Scope::MetaParameterValue>(*lookup).getAttr());
    if (!attr || !isa<MLIRTypeType>(attr.getType()))
      return emitError("'" + dre->spelling + "' names a value, not a type");

    // TODO: Handle type parameters!
    return RefType::get(attr.getSymbol(),
                        ParamBindArrayAttr::get(parser.getContext(), {}));
  }

  return emitError("FIXME: Unsupported type kind!");
}
