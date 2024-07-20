//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// AST representation for a declaration.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_ASTTYPE_H
#define KGEN_MOJOPARSER_ASTTYPE_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace M {
class InflightDiag;

namespace KGEN {
class ParamDeclAttr;
} // namespace KGEN

namespace KGEN::LIT {
class ASTDecl;
class CValue;
class SharedState;
template <typename ValueType>
struct ASTExprAnd;
enum class TypeConvention : uint32_t;
class RefType;
class RefPackType;

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

  /// Get the metatype of the type.
  Type getMetaType() const;

  /// If this is a user declared type, return the declaration that this came
  /// from.  If this is a raw MLIR type, return null.
  ASTDecl *getDecl(SharedState &shared) const;

  /// If this is a parametric user defined type, return all parameter bindings
  /// on this reference to the type.  Note that this is potentially a partial
  /// binding set - incomplete bindings (missing bindings) are valid.
  ArrayRef<TypedAttr> getParamBindings() const;

  /// Return this type with any parameter bindings removed.
  ASTType getWithoutParameters(SharedState &shared) const;

  /// Get the default values for the unbound parameters of the type.
  ArrayRef<TypedAttr> getDefaultPosParams() const;

  /// Return true if this ASTType is canonically equal (equal ignoring sugar) to
  /// the specified other type.
  bool isEqualCanon(ASTType other) const;

  /// Return true if this is the same as another ASTType are the same, or if
  /// they match when UnknownAttr parameters in the 'this' type are treated as
  /// the same as the corresponding parameter in the second type.
  ///    Foo[1] != Foo[2]   but  Bar[?, 1] == Bar[7, 1]
  bool isEqualAllowingUnknownAttr(ASTType other, SharedState &shared) const;

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
  TypeConvention getRegisterPassability(llvm::SMLoc loc,
                                        SharedState &shared) const;

  /// Return true if this type is @register_passable or if it is a generic type
  /// that could bind to a concrete @register_passable type.
  bool mightBeRegisterPassable(llvm::SMLoc loc, SharedState &shared) const;

  /// Return the nonmaterializable decorator target for the type, or null if
  /// there is none.
  ASTType getNonmaterializableTarget(SharedState &shared) const;

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

  /// Given a reference, return the element as an ASTType.  This aborts
  /// if the current type isn't a reference.
  ASTType getReferenceElementType() const;

  /// Given a VariadicType, return the element as an ASTType.  This aborts if
  /// the current type isn't a VariadicType.
  ASTType getVariadicElementType() const;

  /// Return the RefPackType that corresponds to the VariadicPack instance.
  RefPackType getVariadicPackInfo() const;

  /// Given a variadic keyword dictionary type, return the dictionary's value
  /// type as an ASTType.
  ASTType getKwargsDictValueType() const;

  /// Given a variadic keyword dictionary reference type, return the
  /// dictionary's value type as an ASTType.
  ASTType getKwargsDictRefValueType() const;

  /// Returns the user-defined result type, looking through implicit memory
  /// results and stripping off the variant from error throwing results if
  /// needed.
  ASTType getSignatureUserResultType() const;

  /// If this type is parameterized, and if any of the parameters refer to a
  /// ParamIndexRefAttr, replace it with an UnboundAttr so parameter inference
  /// will infer it.
  ///
  /// This makes parameter inference sensitive to what to propagate vs infer.
  /// For example, if expectedType is known to be 'SIMD[uint8, 1]', then we can
  /// infer which constructor to use when the input is an IntLiteral.
  ///
  /// On the other hand, if expectedType is something like 'SIMD[?, 1]' and the
  /// argument is an Int8, then we need the implicit conversion to infer the
  /// base element.  Our solution to this is to rip and replace parameters that
  /// contain unbound parameters, replacing them with UnboundAttr so inference
  /// can find them.
  ASTType getWithUnknownParametersReplaced(SharedState &shared) const;

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

  /// Print the specified parameter like we would in AST type printing.
  static void printParam(raw_ostream &os, TypedAttr param, bool forDiag,
                         bool demangleParams);
  /// Get the specified parameter as a string.
  static std::string getParamAsString(TypedAttr param, bool forDiag,
                                      bool demangleParams);

  /// Create and return a reference type with 'this' as the underlying element
  /// type an implicit lifetime reference with the specified arg name.
  RefType getRefForArgument(const Twine &argName, bool isMut);
};
raw_ostream &operator<<(raw_ostream &os, ASTType type);

} // namespace KGEN::LIT

void addToDiagnostic(KGEN::LIT::ASTType type, InflightDiag &diag);
void addToDiagnostic(TypedAttr paramValue, InflightDiag &diag);

} // namespace M

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

/// Cast from an (const) ASTType to a MLIR type.
template <typename T>
struct CastInfo<T, M::KGEN::LIT::ASTType>
    : public NullableValueCastFailed<T>,
      public DefaultDoCastIfPossible<T, M::KGEN::LIT::ASTType,
                                     CastInfo<T, M::KGEN::LIT::ASTType>> {
  // Provide isPossible here because here we have the const-stripping from
  // ConstStrippingCast.
  static bool isPossible(M::KGEN::LIT::ASTType type) {
    return type && T::classof(type.mlirType);
  }
  static T doCast(M::KGEN::LIT::ASTType type) { return cast<T>(type.mlirType); }
};
template <typename T>
struct CastInfo<T, const M::KGEN::LIT::ASTType>
    : public ConstStrippingForwardingCast<T, const M::KGEN::LIT::ASTType,
                                          CastInfo<T, M::KGEN::LIT::ASTType>> {
};
} // namespace llvm

#endif // KGEN_MOJOPARSER_ASTTYPE_H
