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
  struct EmittedParamInfo {
    Location loc;
    TypedAttr value;
    ASTDecl *ctxDecl;
  };

  MojoInflightDiag(InflightDiag &&diag,
                   ArrayRef<EmittedParamInfo> emittedParams)
      : InflightDiag(std::move(diag)), emittedParams(emittedParams) {}
  ~MojoInflightDiag();

  // These are all wrappers for the underlying functionality that preserves the
  // Self type.
  MojoInflightDiag(MojoInflightDiag &&other) = default;
  MojoInflightDiag &operator=(MojoInflightDiag &&other) = default;

  MojoInflightDiag attachNote(Location loc) && {
    auto params = emittedParams;
    return {std::move(*this).InflightDiag::attachNote(loc), params};
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

  void addEmittedParam(TypedAttr param, std::optional<Location> loc,
                       ASTDecl *ctxDecl);

  ArrayRef<EmittedParamInfo> getEmittedParams() const { return emittedParams; }

private:
  SmallVector<EmittedParamInfo, 2> emittedParams;
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

namespace M {
using KGEN::LIT::MojoInflightDiag;
void addToDiagnostic(KGEN::LIT::ASTType type, InflightDiag &diag);
void addToDiagnostic(TypedAttr paramValue, InflightDiag &diag);
void addToDiagnostic(MojoInflightDiag &&otherDiag, InflightDiag &diag);
} // namespace M

#endif // KGEN_MOJOPARSER_MOJODIAGS_H
