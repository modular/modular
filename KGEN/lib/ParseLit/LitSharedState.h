//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the base class for Lit file parsers that is common between
// expression and statement parsing.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_SHARED_STATE_H
#define LIT_SHARED_STATE_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN::LIT {
class DeclResolver;

//===----------------------------------------------------------------------===//
// SharedParserState
//===----------------------------------------------------------------------===//

/// This is state shared across multiple different instances of LitParserBase
/// which are always shared across them.
class SharedParserState {
public:
  SharedParserState(llvm::SourceMgr &sourceMgr, MLIRContext *context);

  llvm::SourceMgr &sourceMgr;
  MLIRContext *const context;
  std::unique_ptr<DeclResolver> declResolver;

  /// This is set to true if an error occurred at any point processing the file.
  bool errorOccurred = false;

  /// We allow a single ExprParser instance to be live at a time, this gives
  /// efficiently accessible scratch space for it to scribble into.
  llvm::BumpPtrAllocator exprAllocator;
  bool hasExprParser = false;
};

} // namespace M::KGEN::LIT

#endif // LIT_SHARED_STATE_H
