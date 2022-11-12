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
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace M::KGEN {
class ParamDeclAttr;
class ParamBindAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class ASTDecl;
class MValue;
enum class MagicDeclKind : uint8_t;

/// This is the underlying storage for an ASTType and shouldn't be interacted
/// with directly.  Use ASTType instead.
class ASTTypeStorage {
private:
  friend class LitSharedState;
  friend class ASTType;
  ASTTypeStorage(ASTDecl &decl,
                 ArrayRef<std::pair<ParamDeclAttr, MValue>> paramValues);
  ASTTypeStorage(const ASTTypeStorage &) = delete;
  const ASTTypeStorage &operator=(const ASTTypeStorage &) = delete;

  // Note that this is bump pointer allocated and its destructor is never run.
  ASTDecl &decl;
  ArrayRef<std::pair<ParamDeclAttr, MValue>> paramValues;

  /// This is a cached MLIR type that is filled in the first time an ASTType is
  /// converted to an MLIR type.  On error converting the type, the error is
  /// diagnosed and this is filled in with an error type.
  Type mlirType;
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

  // Accessors for the type.
  ASTDecl &getDecl() const {
    assert(pointer && "Cannot dereference null ASTType");
    return pointer->decl;
  }

  using ParamBinding = std::pair<ParamDeclAttr, MValue>;
  ArrayRef<ParamBinding> getParamValues() const;

  /// ASTType is nullable.
  bool isNull() const { return pointer == nullptr; }
  explicit operator bool() const { return pointer != nullptr; }
  bool operator!() const { return pointer == nullptr; }

  /// Return true if this type is the specified 'magic' type.
  bool isMagicType(MagicDeclKind kind) const;

  /// Return true if this ASTType is canonically equal (equal ignoring sugar) to
  /// the specified other type.
  bool isEqualCanon(ASTType other) const;

  /// If this is a bound builtin lit Pointer type, return the element type,
  /// otherwise return null.
  MValue getPointerElementType() const;

  /// This is used for types that are known on valid LValues, which must always
  /// have pointer type.  This is just an asserting form of
  /// getPointerElementType.
  MValue getLValueElementType() const;

  /// Convert this type to a human readable string representation so it can be
  /// printed out for diagnostics.  This may also be inserted into raw_ostream
  /// and diagnostics.
  std::string getAsString() const;

  /// Print to standard error with newline after it, for use in a debugger.
  void dump() const;

  /// ASTType can be put into a PointerUnion, these are implementation details.
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
raw_ostream &operator<<(raw_ostream &os, ASTType type);

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
