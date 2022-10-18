//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Declaration parsing and name binding logic.
//
//===----------------------------------------------------------------------===//

#ifndef LITDECLS_H
#define LITDECLS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace M::KGEN {
class LITFuncOp;
class LITStructDeclOp;
class VarDeclOp;
} // namespace M::KGEN

namespace M::KGEN::LIT {

class LitLexerCursor;
class LitParserBase;
class LitSharedState;
class Scope;

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

class DeclResolver {
public:
  DeclResolver(LitSharedState &state);
  ~DeclResolver();

  /// Resolve all of the declarations that are visible, processing the entire
  /// translation unit.
  void resolveAll();

  /// Add a new declaration that needs to be resolved.
  Scope &addDecl(Operation *decl, Scope *parentScope, LitLexerCursor cursor);

  /// Given a cursor location for a type expression that correctly parsed in the
  /// first pass, reparse it into an expression and resolve it into a type by
  /// performing name lookup and other resolution.  This can produce errors, but
  /// always returns a non-null type.
  Type resolveType(LitLexerCursor cursor, Scope &scope, LitParserBase &parser);

private:
  void resolve(Scope &scope);

  void resolveBody(LITFuncOp op, Scope &scope);
  void resolveBody(LITStructDeclOp op, Scope &scope);
  void resolveSignature(VarDeclOp op, Scope &scope);

private:
  /// This is shared state across the whole parser.
  LitSharedState &sharedState;

  /// This is a mapping of every declaration (module, func, struct, etc) that
  /// we have parsed, along with the metadata for it maintained in `Scope`.
  DenseMap<Operation *, Scope *> parsedDecls;

  /// This array holds all of the parsed declarations in a deterministic order.
  std::vector<Operation *> parsedDeclList;

  /// Name binding is an recursive process in the general case.  This keeps
  /// track of the declarations currently being name bound so we can diagnose
  /// cyclic dependencies.
  DenseSet<Operation *> declsCurrentlyProcessing;
};

} // namespace M::KGEN::LIT

#endif // LITDECLS_H
