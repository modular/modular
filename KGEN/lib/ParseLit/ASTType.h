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
class LitSharedState;
class MValue;

/// This type represents an AST type for a value or declaration, which is an
/// MLIRType.
///
class ASTType {
public:
  ASTType() {}
  ASTType(Type type) : type(type) {}

  Type getMLIRType() const { return type; }

  /// If this is a user declared type, return the declaration that this came
  /// from.  If this is a raw MLIR type, return null.
  ASTDecl *getDecl(LitSharedState &shared) const;

  /// ASTType is nullable.
  bool isNull() const { return !type; }
  explicit operator bool() const { return !!type; }
  bool operator!() const { return !type; }

  /// Return true if this ASTType is canonically equal (equal ignoring sugar) to
  /// the specified other type.
  bool isEqualCanon(ASTType other) const;

  /// Convert this type to a human readable string representation so it can be
  /// printed out for diagnostics.  This may also be inserted into raw_ostream
  /// and diagnostics.
  std::string getAsString() const;

  /// Print to standard error with newline after it, for use in a debugger.
  void dump() const;

  /// ASTType can be put into a PointerUnion, these are implementation details.
  void *getAsVoidPointer() const {
    return const_cast<void *>(type.getAsOpaquePointer());
  }
  static ASTType getFromVoidPointer(void *ptr) {
    return ASTType(Type::getFromOpaquePointer(ptr));
  }

private:
  Type type;
};

mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, ASTType type);
raw_ostream &operator<<(raw_ostream &os, ASTType type);

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
