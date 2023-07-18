//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// AST representation for a declaration.
//
//===----------------------------------------------------------------------===//

#ifndef ASTTYPE_H
#define ASTTYPE_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace M::KGEN {
class ParamBindAttr;
class ParamBindArrayAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class ASTDecl;
class CValue;
class SharedState;
class InflightDiag;
template <typename ValueType>
struct ASTExprAnd;

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

  // Initialize an ASTType from a parameter expression of metatype type.
  ASTType(TypedAttr typeParamExpr);

  operator Type() const { return mlirType; }

  /// ASTType is nullable.
  bool isNull() const { return !mlirType; }
  explicit operator bool() const { return !!mlirType; }
  bool operator!() const { return !mlirType; }

  /// If this is a user declared type, return the declaration that this came
  /// from.  If this is a raw MLIR type, return null.
  ASTDecl *getDecl(SharedState &shared) const;

  /// If this is a parametric user defined type, return all parameter bindings
  /// on this reference to the type.  Note that this is potentially a partial
  /// binding set - incomplete bindings (missing bindings) are valid.
  ParamBindArrayAttr getParamBindings() const;

  /// Return true if this ASTType is canonically equal (equal ignoring sugar) to
  /// the specified other type.
  bool isEqualCanon(ASTType other) const;

  /// Return true if this is a None type.
  bool isNoneType() const;
  /// Return true if this is a TypeCheckError type.
  bool isTypeCheckErrorType() const;

  /// Return true if this type is a register-passable type that can be passed
  /// around and copied in SSA values instead of having to live in memory.
  ///
  /// The location specifies the location of the reference in case the use is
  /// invalid in this location.
  bool isRegisterPassable(llvm::SMLoc loc, SharedState &shared) const;

  /// Return the StructDeclOp::RegisterPassable enum for this type.
  uint8_t getRegisterPassability(llvm::SMLoc loc, SharedState &shared) const;

  /// Return true if this type is a 'trivial' type, that is one that can be
  /// passed around by copying the bits, and whose destructor is a noop.
  bool isTrivial(llvm::SMLoc loc, SharedState &shared) const;

  /// Return true if this type needs to be destroyed.  This is false for trivial
  /// types like Int.  Note: this resolves the body of a struct type.
  bool hasDestructor(llvm::SMLoc loc, SharedState &shared) const;

  /// Return true if this type is copyable, either because it is trivial or has
  /// a copy constructor. Note: this resolves the body of a struct type.
  bool isCopyable(llvm::SMLoc loc, SharedState &shared) const;

  /// Return true if this type is movable from its own type, either because it
  /// is trivial or has a move constructor from self. Note: this resolves the
  /// body of a struct type.
  bool isMovable(llvm::SMLoc loc, SharedState &shared) const;

  /// Return whether this type is movable, either because it is trivial, a
  /// register passable type, or has a move constructor that works with the
  /// specified input value.  Note: this resolves the body of a struct type.
  bool isMovableFrom(ASTExprAnd<CValue> value, SharedState &shared) const;

  /// Given a POP::PointerType, return the element as an ASTType.  This aborts
  /// if the current type isn't a pointer.
  ASTType getPointerElementType() const;

  /// Given a VariadicType, return the element as an ASTType.  This aborts if
  /// the current type isn't a VariadicType.
  ASTType getVariadicElementType() const;

  /// Returns the user-defined result type, looking through implicit memory
  /// results and stripping off the variant from error throwing results if
  /// needed.
  ASTType getSignatureUserResultType() const;

  /// Convert this type to a human readable string representation so it can be
  /// printed out for diagnostics.  This may also be inserted into raw_ostream
  /// and diagnostics.
  /// TODO(16040): Remove demangleParams flag when symbol names are name-erased.
  std::string getAsString(bool forDiag = false,
                          bool demangleParams = false) const;

  /// Print to standard error with newline after it, for use in a debugger.
  void dump() const;

  /// ASTType can be put into a PointerUnion, these are implementation details.
  void *getAsVoidPointer() const {
    return const_cast<void *>(mlirType.getAsOpaquePointer());
  }
  static ASTType getFromVoidPointer(void *ptr) {
    return ASTType(Type::getFromOpaquePointer(ptr));
  }

  /// Print the ASTType. If `forDiag` is set, prettier printing is used to
  /// print the type. If `demangleParams` is set, parameter names will be
  /// demangled, if necessary.
  /// TODO(16040): Remove demangleParams flag when symbol names are name-erased.
  void print(raw_ostream &os, bool forDiag = false,
             bool demangleParams = false) const;
};

void addToDiagnostic(ASTType type, InflightDiag &diag);
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

#endif // ASTTYPE_H
