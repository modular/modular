//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOTOOLING_ASTDECLREF_H
#define KGEN_MOJOTOOLING_ASTDECLREF_H

#include "KGEN/MojoParser/ASTType.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace M {
namespace KGEN::LIT {
class ASTDecl;
class SharedState;
} // namespace KGEN::LIT

class DeclView;
class MojoASTTypeRef;
enum class DeclViewKind;

//===----------------------------------------------------------------------===//
// MojoASTDeclRef
//===----------------------------------------------------------------------===//

/// This class provides a view into a Mojo AST declaration.
class MojoASTDeclRef {
public:
  MojoASTDeclRef(KGEN::LIT::ASTDecl *decl = nullptr) : decl(decl) {}

  /// Returns the operation corresponding to this decl if there is one, nullptr
  /// otherwise. The returned operation should only be used for introspection,
  /// it should not be modified in any way.
  Operation *getIfOperation() const;

  /// Returns if the AST declaration is valid.
  operator bool() const { return decl != nullptr; }

  /// Access the underlying AST decl.
  KGEN::LIT::ASTDecl &operator*() { return *decl; }
  KGEN::LIT::ASTDecl *operator->() { return decl; }
  const KGEN::LIT::ASTDecl &operator*() const { return *decl; }
  const KGEN::LIT::ASTDecl *operator->() const { return decl; }

  /// Returns the type corresponding to this declaration. If not available, this
  /// returns an invalid `MojoASTTypeRef`.
  MojoASTTypeRef getType() const;

  /// Get the name of this declaration if available.
  std::optional<StringRef> getName() const;

  /// Get the deprecation warning on this declaration if available.
  std::optional<StringRef> getDeprecationWarning() const;

  /// Get the location of the start token of this decl. It might not be the
  /// identifier.
  llvm::SMLoc getLoc() const;

  /// Get the parent MojoASTDeclRef of this decl.
  MojoASTDeclRef getParentDecl() const;

  /// Get a DeclView that can be used for more easily inspecting the metadata of
  /// this decl. It supports aliases, modules, functions, structs, arguments,
  /// struct fields and variables.
  std::unique_ptr<DeclView> getView() const;

  /// Return the approximate kind of view this decl would create, if any. This
  /// isn't guaranteed to be the exact decl kind, but it can be used to provide
  /// a fast indicator of the type of view this decl represents.
  std::optional<DeclViewKind> getApproximateViewKind() const;

  //===--------------------------------------------------------------------===//
  // Children
  //===--------------------------------------------------------------------===//

  /// This class represents an individual child entry. It contains the name of
  /// the child declaration, and the group of declarations that share the same
  /// name.
  class ChildEntry {
  public:
    /// Return the name of this entry.
    StringRef getName() const { return name; }

    /// Return the declarations within this entry.
    auto getDecls() const {
      return llvm::map_range(rawEntries, [](KGEN::LIT::ASTDecl *entry) {
        return MojoASTDeclRef(entry);
      });
    }

  private:
    friend MojoASTDeclRef;

    /// Constructs a new child entry.
    ChildEntry(StringRef name, ArrayRef<KGEN::LIT::ASTDecl *> rawEntries)
        : name(name), rawEntries(rawEntries) {}

    /// The name of this entry.
    StringRef name;

    /// The raw entry array.
    ArrayRef<KGEN::LIT::ASTDecl *> rawEntries;
  };

  /// This class defines an iterator over the children of a declaration.
  class ChildIterator
      : public llvm::indexed_accessor_iterator<ChildIterator,
                                               KGEN::LIT::ASTDecl *, ChildEntry,
                                               ChildEntry, ChildEntry> {
  public:
    /// Accesses the entry at the current position.
    ChildEntry operator*() const;

  private:
    friend MojoASTDeclRef;

    /// Constructs a new iterator.
    ChildIterator(MojoASTDeclRef decl, size_t index);
  };

  /// Return the children of this declaration.
  llvm::iterator_range<ChildIterator> getChildren() const;

private:
  using ApproximateDeclViewKind = std::optional<DeclViewKind>;
  using DeclViewInstance = std::unique_ptr<DeclView>;

  /// The documentation for this method is in the corresponding cpp file.
  template <typename ResultType, typename DeclViewT, typename... DeclArgs>
  ResultType createDeclView(DeclArgs &&...declArgs) const;

  /// The documentation for this method is in the corresponding cpp file.
  template <typename ResultType>
  ResultType getViewImpl() const;

  /// Allow MojoParserContext to access the internal impl.
  friend class MojoParserContext;

  /// The ASTDecl being referenced.
  KGEN::LIT::ASTDecl *decl;
};

//===----------------------------------------------------------------------===//
// MojoASTTypeRef
//===----------------------------------------------------------------------===//

/// This class provides a view into an Mojo AST type.
class MojoASTTypeRef {
public:
  MojoASTTypeRef() : MojoASTTypeRef(nullptr) {}
  MojoASTTypeRef(KGEN::LIT::ASTType type) : type(type) {}
  MojoASTTypeRef(Type type) : type(type) {}
  MojoASTTypeRef(const void *impl) : type(Type::getFromOpaquePointer(impl)) {}

  /// Returns if the AST declaration is valid.
  operator bool() const { return bool(type); }

  /// Returns a readable string representation of this type.
  std::string getAsString() const;

  /// If the current type is a reference, return the type of the pointee. This
  /// aborts if the current type isn't a reference.
  MojoASTTypeRef getReferenceElementType() const;

  /// Return the MLIR type associated with this
  Type getMLIRType() const;

private:
  // Return the decl that defined this type.
  MojoASTDeclRef getDecl(KGEN::LIT::SharedState &sharedState);

  /// Allow MojoParserContext to access the internal implementation.
  friend class MojoParserContext;

  /// The internal implementation of the AST type.
  KGEN::LIT::ASTType type;
};

} // namespace M

namespace llvm {
/// Cast from an MojoASTTypeRef to a mojo type.
template <typename T>
struct CastInfo<T, M::MojoASTTypeRef>
    : public NullableValueCastFailed<T>,
      public DefaultDoCastIfPossible<T, M::MojoASTTypeRef,
                                     CastInfo<T, M::MojoASTTypeRef>> {
  // Provide isPossible here because here we have the const-stripping from
  // ConstStrippingCast.
  static bool isPossible(M::MojoASTTypeRef astType) {
    if (!astType)
      return false;
    return T::classof(astType.getMLIRType());
  }

  static T doCast(M::MojoASTTypeRef astType) {
    return cast<T>(astType.getMLIRType());
  }
};

template <typename T>
struct CastInfo<T, const M::MojoASTTypeRef>
    : public ConstStrippingForwardingCast<T, const M::MojoASTTypeRef,
                                          CastInfo<T, M::MojoASTTypeRef>> {};

} // namespace llvm

#endif // KGEN_MOJOTOOLING_ASTDECLREF_H
