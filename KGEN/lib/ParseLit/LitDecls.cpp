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
#include "LitScope.h"
#include "LitSharedState.h"

using namespace M;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

DeclResolver::DeclResolver(SharedParserState &state)
    : sharedParserState(state) {}
DeclResolver::~DeclResolver() {}

/// Add a new declaration that needs to be resolved.
void DeclResolver::addDecl(LLCL::RCRef<Scope> declScope) {
  Operation *op = declScope->getDecl();
  parsedDeclList.push_back(op);
  parsedDecls[op] = std::move(declScope);
}

/// Resolve all of the declarations that are visible.
void DeclResolver::resolveAll() {
  // We can do this in any order, but choose to use the order they are
  // discovered so diagnostics are mostly top-down.  Resolving declarations may
  // cause more entries to be added to this list.
  for (size_t i = 0; i != parsedDeclList.size(); ++i)
    resolve(*parsedDecls[parsedDeclList[i]]);
}
