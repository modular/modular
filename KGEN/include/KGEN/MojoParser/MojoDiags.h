//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares support for function-call related machinery.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_MOJODIAGS_H
#define KGEN_MOJOPARSER_MOJODIAGS_H

#include "KGEN/MojoParser/ASTType.h"
#include "Support/Compiler/Diags.h"

namespace M::KGEN::LIT {

/// This is a wrapper around MojoInflightDiag that adds some Mojo-specific
/// functionality.
class MojoInflightDiag : public InflightDiag {
public:
  MojoInflightDiag(InflightDiag &&diag,
                   ArrayRef<std::pair<Location, ASTType>> emittedTypes)
      : InflightDiag(std::move(diag)), emittedTypes(emittedTypes) {}
  ~MojoInflightDiag();

  // Override emission of ASTType to track the types we've emitted so our dtor
  // can emit notes about them.
  MojoInflightDiag &operator<<(ASTType type) & {
    addType(type);
    return *this;
  }
  MojoInflightDiag operator<<(ASTType type) && {
    addType(type);
    return std::move(*this);
  }

  // These are all wrappers for the underlying functionality that preserves the
  // Self type.
  MojoInflightDiag(MojoInflightDiag &&other) = default;
  MojoInflightDiag &operator=(MojoInflightDiag &&other) = default;

  MojoInflightDiag attachNote(Location loc) && {
    auto types = emittedTypes;
    return {std::move(*this).InflightDiag::attachNote(loc), types};
  }
  MojoInflightDiag attachNote(llvm::SMLoc loc) &&;
  MojoInflightDiag &attachNote(Location loc) & {
    InflightDiag::attachNote(loc);
    return *this;
  }
  MojoInflightDiag &attachNote(llvm::SMLoc loc) &;
  template <typename Arg>
  MojoInflightDiag &operator<<(Arg &&value) & {
    addToDiagnostic(std::forward<Arg>(value), *this);
    return *this;
  }
  template <typename Arg>
  MojoInflightDiag operator<<(Arg &&value) && {
    addToDiagnostic(std::forward<Arg>(value), *this);
    return std::move(*this);
  }

  void addType(ASTType type);

private:
  SmallVector<std::pair<Location, ASTType>, 2> emittedTypes;
};

// A wrapper around Diags that adds Mojo-specific functionality.
class MojoDiags : public Diags {
public:
  using Diags::Diags;
  MojoInflightDiag emitError(Location loc, const Twine &message) {
    return MojoInflightDiag(Diags::emitError(loc, message), {});
  }
  MojoInflightDiag emitWarning(Location loc, const Twine &message) {
    return MojoInflightDiag(Diags::emitWarning(loc, message), {});
  }
  MojoInflightDiag emitError(llvm::SMLoc loc, const Twine &message);
  MojoInflightDiag emitWarning(llvm::SMLoc loc, const Twine &message);
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_MOJODIAGS_H
