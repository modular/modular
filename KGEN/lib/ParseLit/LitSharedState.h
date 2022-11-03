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
class ParamBindAttr;
class ParamBindArrayAttr;
}

namespace M::KGEN::LIT {
class DeclResolver;
class ASTDecl;
class ASTType;

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

  /// Get a uniqued and pointer sized reference to an ASTType.
  ASTType getASTType(ASTDecl *decl, ArrayRef<ParamBindAttr> params);

  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTDecl *typeCheckErrorTypeDecl = nullptr;
  ASTType getTypeCheckErrorType() const;

  /// This is the "type" type, which can bind to any lit type.
  ASTDecl *typeTypeDecl = nullptr;
  ASTType getTypeType() const;

  /// This is the decl for the builtin 'index' type.
  ASTDecl *indexDecl = nullptr;
  ASTType getIndexType() const;

  /// This is the decl for the builtin 'kgen.none' type.
  ASTDecl *noneDecl = nullptr;
  ASTType getNoneType() const;

  /// This is the decl for the builtin POP::PointerType type.
  ASTDecl *pointerDecl = nullptr;
  // FIXME: This isn't correctly parameterized.
  ASTType getPointerType() const;

  /// This is the decl for the builtin signature type.
  ASTDecl *signatureDecl = nullptr;
  // FIXME: This isn't correctly parameterized; we need variadics.
  ASTType getSignatureType() const;

  /// This is the decl for the builtin lit.object type.
  ASTDecl *objectDecl = nullptr;
  ASTType getObjectType() const;

  /// This is set to true if an error occurred at any point processing the file.
  bool errorOccurred = false;

  /// Inflate a lightweight SMLoc into an MLIR Location object for addition
  /// into the IR.
  Location translateLocation(llvm::SMLoc loc);

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

private:
  /// This is used for memory that lives as long as the global parser does.
  llvm::BumpPtrAllocator persistentAllocator;

  class Impl;
  std::unique_ptr<Impl> impl;

  ArrayRef<ParamBindAttr> getUniquedParams(ArrayRef<ParamBindAttr> params);
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

/// This keeps track of specific kinds of "magic" declarations that do not have
/// a standard AST representation.
enum class MagicDeclKind {
  // This is not a magic declaration, process it as normal.
  kNormal,
  // This type is produced when an error is detected to simplify clients.
  kTypeCheckErrorType,
  // This is the 'type' type.
  kTypeType,
  // This is the __builtin.mlirtype.builtin.index type.
  kIndexType,
  // This is the __builtin.mlirtype.lit.none type.
  kNoneType,
  // This is a POP::PointerType type.
  kPointerType,
  // This is a KGEN Signature for a callable function.
  kSignatureType,
};

} // namespace M::KGEN::LIT

#endif // LIT_SHARED_STATE_H
