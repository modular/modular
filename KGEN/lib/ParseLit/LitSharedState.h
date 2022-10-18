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
#include "mlir/IR/BuiltinAttributes.h"

namespace llvm {
class SourceMgr;
}

namespace M::KGEN::LIT {
class DeclResolver;

/// This is state shared across multiple different instances of LitParser
/// which are always shared across them.
class LitSharedState {
public:
  LitSharedState(llvm::SourceMgr &sourceMgr, MLIRContext *context);

  llvm::SourceMgr &sourceMgr;
  MLIRContext *const context;
  std::unique_ptr<DeclResolver> declResolver;

  const mlir::StringAttr bufferNameIdentifier;

  /// This is set to true if an error occurred at any point processing the file.
  bool errorOccurred = false;

  /// This is used for memory that lives as long as the global parser does.
  llvm::BumpPtrAllocator persistentAllocator;

  /// We allow a single ExprParser instance to be live at a time, this gives
  /// efficiently accessible scratch space for it to scribble into.
  llvm::BumpPtrAllocator exprAllocator;
  bool hasExprParser = false;
};

} // namespace M::KGEN::LIT

#endif // LIT_SHARED_STATE_H
