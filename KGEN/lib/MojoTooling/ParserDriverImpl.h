//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef PARSERDRIVERIMPL_H
#define PARSERDRIVERIMPL_H

#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoTooling/ParserDriver.h"

namespace M {

/// Resolves an unparsed decl enough for the language server to operate. Fully
/// body-resolves all decls descended from the root, validates their doc
/// strings, and leaves everything else unparsed.
void resolveForLSP(KGEN::LIT::DeclResolver &resolver, KGEN::LIT::ASTDecl &decl);

/// Signature-resolves all parsed decls that are not yet resolved, skipping
/// lazy named imports and body-resolving non-FnOp bytecode decls. This is the
/// second stage of LSP resolution: after resolveForLSP covers the direct
/// children of a container, this step covers transitive dependencies that were
/// pulled in during resolution.
void resolveSignaturesForLSP(KGEN::LIT::DeclResolver &resolver);

/// This class represents the internal implementation of the parser driver.
struct MojoParserContext::Impl {
  Impl(llvm::SourceMgr &sourceMgr, KGEN::LIT::ParserConfig &config);

  //===--------------------------------------------------------------------===//
  // General State
  //===--------------------------------------------------------------------===//

  /// The shared state for the parser.
  KGEN::LIT::SharedState sharedState;

  /// The top level decl for everything being parsed.
  KGEN::LIT::ASTDecl *topLevelDecl = nullptr;

  /// The main module we are parsing into.
  mlir::OwningOpRef<ModuleOp> module;

  //===--------------------------------------------------------------------===//
  // REPL State
  //===--------------------------------------------------------------------===//

  /// The location mapper used for REPL expressions.
  MojoParserContext::REPLLocMapper replLocMapper;

  /// The decls of each REPL module that have been successfully parsed, mapped
  /// to the previous REPL module that they replaced.
  llvm::MapVector<KGEN::LIT::ASTDecl *, KGEN::LIT::ASTDecl *>
      prevReplModuleDecls;
  SmallVector<KGEN::LIT::ASTDecl *> replModuleDecls;
};
} // namespace M

#endif // PARSERDRIVERIMPL_H
