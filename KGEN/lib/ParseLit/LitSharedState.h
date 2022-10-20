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
  ~LitSharedState();

  llvm::SourceMgr &sourceMgr;
  MLIRContext *const context;
  std::unique_ptr<DeclResolver> declResolver;

  const mlir::StringAttr bufferNameIdentifier;

  /// This is set to true if an error occurred at any point processing the file.
  bool errorOccurred = false;

  /// This is used for memory that lives as long as the global parser does.
  llvm::BumpPtrAllocator persistentAllocator;

  /// Inflate a light weight SMLoc into an MLIR Location object for addition
  /// into the IR.
  Location translateLocation(llvm::SMLoc loc);
};

/// This enum indicates how much parsing and type checking has been done on
/// this declaration.
enum class DeclResolvedness : int8_t {
  /// This declaration hasn't been parsed outside of its identifier being
  /// processed.  We don't know anything about its arguments, generic
  /// signature, etc.
  unparsed,

  /// This declaration has had its signature parsed and type checked, so we know
  /// what parameters and metaparameters it might take, but its body hasn't been
  /// processed.
  signatureResolved,

  /// This declaration has been fully type checked, including its body.  Any
  /// declarations within the body may not be fully resolved though.
  fullyResolved
};

} // namespace M::KGEN::LIT

#endif // LIT_SHARED_STATE_H
