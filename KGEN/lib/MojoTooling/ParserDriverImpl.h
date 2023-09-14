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

  /// The decls of each REPL module that have been successfully parsed.
  SmallVector<KGEN::LIT::ASTDecl *> replModuleDecls;

  /// The detached IR created for invalid REPL modules.
  /// TODO: We should restructure the parser to make it clean to drop parsed
  /// modules in the case of failure, in which case we could remove this.
  SmallVector<OwningOpRef<Operation *>> detachedREPLModules;
};
} // namespace M

#endif // PARSERDRIVERIMPL_H
