//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_ASTDECLREF_H
#define KGEN_MOJOPARSER_ASTDECLREF_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace M {
namespace KGEN::LIT {
class SharedState;
} // namespace KGEN::LIT

class DeclView;
class MojoASTTypeRef;

//===----------------------------------------------------------------------===//
// MojoASTDeclRef
//===----------------------------------------------------------------------===//

/// This class provides a view into a Mojo AST declaration.
class MojoASTDeclRef {
public:
  MojoASTDeclRef() : MojoASTDeclRef(nullptr) {}

  MojoASTDeclRef(void *impl) : impl(impl) {}

  /// Returns the operation corresponding to this decl if there is one, nullptr
  /// otherwise. The returned operation should only be used for introspection,
  /// it should not be modified in any way.
  Operation *getIfOperation() const;

  /// Returns if the AST declaration is valid.
  operator bool() const { return impl != nullptr; }

  /// Returns the underlying pointer to the implementation backed by this class.
  /// It can be used as a unique ID for this declaration.
  void *getAsVoidPointer() const { return impl; }

  /// Returns the type corresponding to this declaration. If not availble, this
  /// returns an invalid `MojoASTTypeRef`.
  MojoASTTypeRef getType() const;

  /// Get the mangled name of this declaration if available.
  std::optional<StringAttr> getMangledName() const;

  /// Get the name of this declaration if available.
  std::optional<StringRef> getName() const;

  /// Get the location of the start token of this decl. It might not be the
  /// identifier.
  llvm::SMLoc getLoc() const;

  /// Get a DeclView that can be used for more easily inspecting the metadata of
  /// this decl.
  std::unique_ptr<DeclView> getView() const;

private:
  /// Allow MojoParserContext to access the internal implementation.
  friend class MojoParserContext;

  /// The internal implementation of the AST declaration.
  void *impl;
};

//===----------------------------------------------------------------------===//
// MojoASTTypeRef
//===----------------------------------------------------------------------===//

/// This class provides a view into an Mojo AST type.
class MojoASTTypeRef {
public:
  MojoASTTypeRef() : MojoASTTypeRef(nullptr) {}
  MojoASTTypeRef(void *impl) : impl(impl){};
  MojoASTTypeRef(const mlir::Type &type);

  /// Returns if the AST declaration is valid.
  operator bool() const { return impl != nullptr; }

  /// Returns a readable string representation of this type.
  std::string getAsString() const;

  /// If the current type is a pointer, return the type of the pointee. This
  /// aborts if the current type isn't a pointer.
  MojoASTTypeRef getPointerElementType() const;

  /// Return the MLIR type associated with this
  Type getMLIRType() const;

  /// Returns the underlying pointer to the implementation backed by this class.
  /// It can be used as a unique ID for this declaration.
  void *getAsVoidPointer() const { return impl; }

private:
  // Return the decl that defined this type.
  MojoASTDeclRef getDecl(KGEN::LIT::SharedState &sharedState);

  /// Allow MojoParserContext to access the internal implementation.
  friend class MojoParserContext;

  /// The internal implementation of the AST type.
  void *impl;
};

} // namespace M

#endif // KGEN_MOJOPARSER_ASTDECLREF_H
