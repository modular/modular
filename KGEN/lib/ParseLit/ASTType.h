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
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace M::KGEN {
class ParamBindAttr;
class ParamBindArrayAttr;
}

namespace M::KGEN::LIT {
class ASTDecl;
class LitSharedState;

/// This is a simple wrapper around an MLIR Type that provides helpful utilities
/// for working with our types, provides pretty printing in diagnostics, and
///
class ASTType {
public:
  /// The MLIR version of the type is conveniently accessible.
  Type mlirType;

  ASTType() {}

  // Implicitly convert to and from MLIR Type.
  ASTType(Type mlirType) : mlirType(mlirType) {}
  operator Type() const { return mlirType; }

  /// ASTType is nullable.
  bool isNull() const { return !mlirType; }
  explicit operator bool() const { return !!mlirType; }
  bool operator!() const { return !mlirType; }

  /// If this is a user declared type, return the declaration that this came
  /// from.  If this is a raw MLIR type, return null.
  ASTDecl *getDecl(LitSharedState &shared) const;

  /// If this is a parametric user defined type, return all parameter bindings
  /// on this reference to the type.  Note that this is potentially a partial
  /// binding set - incomplete bindings (missing bindings) are valid.
  ParamBindArrayAttr getParamBindings() const;

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
    return const_cast<void *>(mlirType.getAsOpaquePointer());
  }
  static ASTType getFromVoidPointer(void *ptr) {
    return ASTType(Type::getFromOpaquePointer(ptr));
  }
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
