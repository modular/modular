//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// AST representation for a declaration.
//
//===----------------------------------------------------------------------===//

#ifndef AST_TYPE_H
#define AST_TYPE_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace M::KGEN {
class ParamBindAttr;
class ParamBindArrayAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class ASTDecl;

/// This is the underlying storage for an ASTType and shouldn't be interacted
/// with directly.  Use ASTType instead.
class ASTTypeStorage {
private:
  friend class LitSharedState;
  friend class ASTType;
  ASTTypeStorage(ASTDecl &decl, ArrayRef<ParamBindAttr> paramValues)
      : decl(decl), paramValues(paramValues) {}
  ASTTypeStorage(const ASTTypeStorage &) = delete;
  const ASTTypeStorage &operator=(const ASTTypeStorage &) = delete;

  // Note that this is bump pointer allocated and its destructor is never run.
  ASTDecl &decl;
  ArrayRef<ParamBindAttr> paramValues;
};

/// This type represents an AST type for a value or declaration, which is a
/// pointer to the DeclAST that defines it as well as any bound parameters.
///
/// Instances of this type are always created by LitSharedState to ensure that
/// their parameter arrays are uniqued and this can be copied around with ease.
/// This type is typically used via ASTType.
///
/// The bound parameters may themselves refer to other parameters in the
/// enclosing scope, e.g. in the case of `SomeType[size*42]`.
///
/// This is a pointer-sized reference to a uniqued ASTType, maintained in the
/// persistent allocator for the current parse.
class ASTType {
public:
  ASTType() : pointer(nullptr) {}

  ASTDecl *getDecl() const {
    assert(pointer && "Cannot dereference null ASTType");
    return &pointer->decl;
  }
  ArrayRef<ParamBindAttr> getParamValues() const {
    assert(pointer && "Cannot dereference null ASTType");
    return pointer->paramValues;
  }

  operator bool() const { return pointer != nullptr; }
  bool operator!() const { return pointer == nullptr; }

  /// Return the MLIR type that corresponds to this AST type.  This assumes the
  /// ASTType is correctly formed.
  Type getMLIRType(MLIRContext *context);

  /// Convert this type to a human readable string representation so it can be
  /// printed out for diagnostics.
  std::string getAsString() const;

  /// Print to standard error with newline after it, for use in a debugger.
  void dump() const;

  void *getAsVoidPointer() const { return pointer; }
  static ASTType getFromVoidPointer(void *ptr) {
    return ASTType(static_cast<ASTTypeStorage *>(ptr));
  }

private:
  friend class LitSharedState;
  ASTType(ASTTypeStorage *pointer) : pointer(pointer) {}
  ASTTypeStorage *pointer;
};

mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, ASTType type);

using FullType = std::pair<Type, ASTType>;

} // namespace M::KGEN::LIT

namespace llvm {
template <>
struct PointerLikeTypeTraits<M::KGEN::LIT::ASTType> {
public:
  using ASTType = M::KGEN::LIT::ASTType;
  static inline void *getAsVoidPointer(ASTType value) {
    return const_cast<void *>(value.getAsVoidPointer());
  }
  static inline ASTType getFromVoidPointer(void *pointer) {
    return ASTType::getFromVoidPointer(pointer);
  }
  enum {
    NumLowBitsAvailable = PointerLikeTypeTraits<void *>::NumLowBitsAvailable
  };
};

} // namespace llvm

#endif // AST_TYPE_H
