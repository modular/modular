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

#include "LitSharedState.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace M::KGEN {
class LITFuncOp;
class LITStructDeclOp;
class VarDeclOp;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class LitLexer;
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

  MLIRContext *getContext() const { return sharedState.context; }

  /// Resolve all of the declarations that are visible, processing the entire
  /// translation unit.
  void resolveAll(llvm::SMLoc loc);

  /// Add a new declaration that needs to be resolved.
  Scope &addDecl(Operation *decl, Scope *parentScope, LitLexerCursor cursor,
                 ssize_t indentation);

  /// Add a declaration that is already fully resolved.
  Scope &addFullyResolvedDecl(Operation *decl, Scope *parentScope);

  /// Return the scope for the specified declaration that is already entered
  /// into the resolver.
  Scope &getScopeForDecl(Operation *decl) {
    auto *scope = getScopeForDeclIfPresent(decl);
    assert(scope && "not a declaration???");
    return *scope;
  }

  /// Return the scope for the specified declaration if it is in the resolver.
  Scope *getScopeForDeclIfPresent(Operation *decl) {
    auto it = parsedDecls.find(decl);
    return it != parsedDecls.end() ? it->second : nullptr;
  }

  /// Resolve the specified declaration to at least the specified level of
  /// resolution, performing incremental type checking as appropriate.
  LogicalResult resolve(Scope &scope, DeclResolvedness howResolved,
                        llvm::SMLoc loc);

private:
  /// The resolveSignature methods are invoked on an operation to parse and type
  /// check the signature for the operation.  On parse failure, these should
  /// return a failure, which will cause the driver to mark the decl as invalid
  /// for further references.
  LogicalResult resolveSignature(LITFuncOp op, LitLexer &lexer, Scope &scope);
  LogicalResult resolveSignature(LITStructDeclOp op, LitLexer &lexer,
                                 Scope &scope);
  LogicalResult resolveSignature(VarDeclOp op, LitLexer &lexer, Scope &scope);

  void resolveBody(LITFuncOp op, LitLexer &lexer, Scope &scope);
  void resolveBody(LITStructDeclOp op, LitLexer &lexer, Scope &scope);
  void resolveBody(VarDeclOp op, LitLexer &lexer, Scope &scope);

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

  DeclResolver(const DeclResolver &) = delete;
  DeclResolver &operator=(const DeclResolver &) = delete;
};

} // namespace M::KGEN::LIT

#endif // LITDECLS_H
