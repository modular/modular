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

namespace M::KGEN {
class ParamDeclAttr;
}

namespace M::KGEN::LIT {
class DeclResolver;
class ASTDecl;
class ASTType;
class MValue;

inline const char *plural(size_t value) { return value == 1 ? "" : "s"; }

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

  MLIRContext *getContext() const { return context; }

  /// Given the symbol for a lit declaration, return the ASTDecl that
  /// corresponds to it.  This doesn't allow null symbols, so it always
  /// succeeds.
  ASTDecl &getDeclForSymbol(SymbolRefAttr symbol);

  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTType getTypeCheckErrorType() const;

  /// This is the decl for the builtin 'kgen.none' type.
  ASTType getNoneType() const;

  /// This is the decl for the builtin lit.object struct type.
  ASTType getObjectType() const;

  /// This is set to true if an error occurred at any point processing the file.
  bool errorOccurred = false;

  /// Emit an error through the parser's logic.
  InFlightDiagnostic emitError(Location loc, const Twine &twine);

  /// Emit an error through the parser's logic.
  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &twine);

  /// Inflate a lightweight SMLoc into an MLIR Location object for addition
  /// into the IR.
  Location translateLocation(llvm::SMLoc loc) const;

  /// Allocate an expression node into the persistent bump pointer allocator.
  template <typename T, typename... Args>
  T *allocPersistent(Args &&...args) {
    void *node = persistentAllocator.Allocate(sizeof(T), llvm::Align::Of<T>());
    return new (node) T(std::forward<Args>(args)...);
  }

  /// memcpy the specified ArrayRef into the persistent allocator and return a
  /// pointer to the new data.  This cannot be used with things that have
  /// non-trivial copyctors/dtors because the expression allocator does run
  /// destructors.
  template <typename T>
  ArrayRef<T> getPersistentCopy(ArrayRef<T> elements) {
    if (elements.empty())
      return elements;

    size_t dataSize = sizeof(T) * elements.size();
    T *result = static_cast<T *>(
        persistentAllocator.Allocate(dataSize, llvm::Align::Of<T>()));
    memcpy(result, elements.data(), dataSize);
    return ArrayRef<T>(result, elements.size());
  }

  /// Add magic things to the builtins decl when parsing starts.
  void addBuiltinTypes(ASTDecl &builtinsDecl);

private:
  /// This is used for memory that lives as long as the global parser does.
  llvm::BumpPtrAllocator persistentAllocator;

  class Impl;
  std::unique_ptr<Impl> impl;
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
