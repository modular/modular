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
#include "LitExprNodes.h"
#include "LitLexer.h"
#include "LitParserBase.h"
#include "LitScope.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
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
  if (auto var = dyn_cast<VarDeclOp>(newDecl))
    name = var.getNameAttr();
  else if (auto fn = dyn_cast<LITFuncOp>(newDecl))
    name = fn.getNameAttr();
  else if (auto str = dyn_cast<LITStructDeclOp>(newDecl))
    name = str.getNameAttr();
  else {
    assert(isa<ModuleOp>(newDecl) && "Unknown declaration kind");
    return;
  }

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
  for (auto [op, scope] : parsedDecls)
    scope->~Scope();
}

/// Add a new declaration that needs to be resolved.
Scope &DeclResolver::addDecl(Operation *decl, Scope *parentScope,
                             LitLexerCursor cursor, ssize_t indentation) {
  void *rawScopePtr =
      sharedState.persistentAllocator.Allocate(sizeof(Scope), alignof(Scope));
  Scope *scope =
      new (rawScopePtr) Scope(decl, parentScope, cursor, indentation);
  parsedDeclList.push_back(decl);
  parsedDecls[decl] = scope;

  if (parentScope)
    parentScope->addToScope(scope, sharedState);

  return *scope;
}

Scope &DeclResolver::addFullyResolvedDecl(Operation *decl, Scope *parentScope) {
  auto &scope = addDecl(decl, parentScope, LitLexerCursor(), 0);
  scope.resolvedness = DeclResolvedness::fullyResolved;
  return scope;
}

/// Resolve all of the declarations that are visible.
void DeclResolver::resolveAll(Location loc) {
  // We can do this in any order, but choose to use the order they are
  // discovered so diagnostics are mostly top-down.  Resolving declarations may
  // cause more entries to be added to this list.
  for (size_t i = 0; i != parsedDeclList.size(); ++i)
    (void)resolve(*parsedDecls[parsedDeclList[i]],
                  DeclResolvedness::fullyResolved, loc);
}

/// Resolve the specified declaration to at least the specified level of
/// resolution, performing incremental type checking as appropriate.
LogicalResult DeclResolver::resolve(Scope &scope, DeclResolvedness howResolved,
                                    Location loc) {
  // If scope is already resolved enough, we're done.
  if (scope.resolvedness >= howResolved)
    return success();

  Operation *decl = scope.getDecl();

  // If we are currently name binding this operation, we found a cycle, reject
  // it with an error.
  if (!declsCurrentlyProcessing.insert(decl).second) {
    emitError(loc, "recursive reference to declaration");
    return failure();
  }

  // If the signature hasn't been parsed, do so.
  if (scope.resolvedness < DeclResolvedness::signatureResolved) {
    // Handle each operation that can be name bound.
    TypeSwitch<Operation *>(decl)
        .Case<LITFuncOp, LITStructDeclOp, VarDeclOp>([&](auto op) {
          LitLexer lexer(sharedState, scope.getCursor());
          resolveSignature(op, lexer, scope);
          scope.getCursor() = lexer.getCursor();
        })
        .Case<ModuleOp>([&](auto op) { /*Nothing*/ })
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
          LitLexer lexer(sharedState, scope.getCursor());
          resolveBody(op, lexer, scope);
        })
        .Case<ModuleOp>([&](auto op) { /*Nothing*/ })
        .Default([&](auto attr) {
          decl->emitError("do not know how to resolve the body of this decl!");
        });
    scope.resolvedness = DeclResolvedness::fullyResolved;
  }

  declsCurrentlyProcessing.erase(decl);
  return success();
}
