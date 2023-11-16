//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef PARSERDRIVERIMPL_H
#define PARSERDRIVERIMPL_H

#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoTooling/ParserDriver.h"

namespace M {
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
