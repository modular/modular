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
                             LitLexerCursor cursor, ssize_t indentation) {
  void *rawScopePtr =
      sharedState.persistentAllocator.Allocate(sizeof(Scope), alignof(Scope));
  Scope *scope =
      new (rawScopePtr) Scope(decl, parentScope, cursor, indentation);
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
    resolve(*parsedDecls[parsedDeclList[i]], DeclResolvedness::fullyParsed);
}

/// Resolve the specified declaration to at least the specified level of
/// resolution, performing incremental type checking as appropriate.
void DeclResolver::resolve(Scope &scope, DeclResolvedness howResolved) {
  // If scope is already resolved enough, we're done.
  if (scope.resolvedness >= howResolved)
    return;

  Operation *decl = scope.getDecl();

  // If we are currently name binding this operation, we found a cycle, reject
  // it with an error.
  if (!declsCurrentlyProcessing.insert(decl).second) {
    assert(0 &&
           "FIXME: Diagnose cyclic reference when it is possible to happen");
  }

  // If the signature hasn't been parsed, do so.
  if (scope.resolvedness < DeclResolvedness::signatureParsed) {
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
    scope.resolvedness = DeclResolvedness::signatureParsed;
  }

  // If the declaration hasn't been fully parsed and we need to, do so.
  if (scope.resolvedness < DeclResolvedness::fullyParsed &&
      howResolved == DeclResolvedness::fullyParsed) {
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
    scope.resolvedness = DeclResolvedness::fullyParsed;
  }

  declsCurrentlyProcessing.erase(decl);
}
