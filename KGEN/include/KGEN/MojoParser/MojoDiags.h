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

#include "KGEN/Diagnostics/DiagnosticIDs.h"
#include "KGEN/Diagnostics/DiagnosticRegistry.h"
#include "KGEN/MojoParser/ASTType.h"
#include "Support/Compiler/Diags.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

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

  // Move operations preserve the MojoInflightDiag return type.
  MojoInflightDiag(MojoInflightDiag &&other) = default;
  MojoInflightDiag &operator=(MojoInflightDiag &&other) = default;

  /// Returns the associated SharedState if active, null if abandoned.
  SharedState *getSharedIfActive() const {
    auto *diags = getDiags();
    return !diags ? nullptr : static_cast<SharedState *>(diags->extraContext);
  }

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

/// Extends Diags to return MojoInflightDiag and support DiagID-based emission.
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

  /// Emit an error/warning from a DiagID, dispatching arguments via
  /// addToDiagnostic so MLIR types and ASTTypes format correctly.
  template <typename LocT, typename... Args>
  MojoInflightDiag emitError(LocT loc, Diag::DiagID id, Args &&...args);
  template <typename LocT, typename... Args>
  MojoInflightDiag emitWarning(LocT loc, Diag::DiagID id, Args &&...args);
};

} // namespace M::KGEN::LIT

namespace M {
using KGEN::LIT::MojoInflightDiag;
void addToDiagnostic(KGEN::LIT::ASTType type, InflightDiag &diag);
void addToDiagnostic(TypedAttr paramValue, InflightDiag &diag);
void addToDiagnostic(MojoInflightDiag &&otherDiag, InflightDiag &diag);
} // namespace M

namespace M::KGEN::LIT {

namespace detail {

// Dispatch args[n] to addToDiagnostic. ADL via MojoInflightDiag finds all
// M::addToDiagnostic overloads (ASTType, TypedAttr, Twine, etc.).
inline void applyNthArgToDiag(size_t, MojoInflightDiag &) {}

template <typename Arg0, typename... Rest>
void applyNthArgToDiag(size_t n, MojoInflightDiag &diag, Arg0 &&arg0,
                       Rest &&...rest) {
  if (n == 0)
    addToDiagnostic(std::forward<Arg0>(arg0), diag);
  else
    applyNthArgToDiag(n - 1, diag, std::forward<Rest>(rest)...);
}

// Parse a DiagID message template, emitting text segments and dispatching
// {N} placeholders to addToDiagnostic(args[N]).
template <typename... Args>
void applyDiagTemplate(MojoInflightDiag &diag, llvm::StringRef tmpl,
                       Args &&...args) {
  while (!tmpl.empty()) {
    size_t bracePos = tmpl.find('{');
    if (bracePos == llvm::StringRef::npos) {
      addToDiagnostic(llvm::Twine(tmpl), diag);
      break;
    }
    if (bracePos > 0)
      addToDiagnostic(llvm::Twine(tmpl.take_front(bracePos)), diag);
    tmpl = tmpl.drop_front(bracePos + 1); // consume '{'
    size_t closePos = tmpl.find('}');
    if (closePos == llvm::StringRef::npos) {
      addToDiagnostic(llvm::Twine("{") + tmpl, diag);
      break;
    }
    llvm::StringRef idxStr = tmpl.take_front(closePos);
    tmpl = tmpl.drop_front(closePos + 1); // consume '}'
    size_t idx;
    if (idxStr.getAsInteger(10, idx)) {
      // Malformed index: emit as literal text.
      addToDiagnostic(llvm::Twine("{") + idxStr + "}", diag);
      continue;
    }
    applyNthArgToDiag(idx, diag, std::forward<Args>(args)...);
  }
}

} // namespace detail

template <typename LocT, typename... Args>
inline MojoInflightDiag MojoDiags::emitError(LocT loc, Diag::DiagID id,
                                             Args &&...args) {
  const Diag::DiagnosticInfo *info = Diag::DiagnosticRegistry::get().lookup(id);
  assert(info && "DiagID not registered in DiagnosticRegistry");
  auto diag = emitError(loc, llvm::Twine());
  detail::applyDiagTemplate(diag, info->messageTemplate,
                            std::forward<Args>(args)...);
  return diag;
}

template <typename LocT, typename... Args>
inline MojoInflightDiag MojoDiags::emitWarning(LocT loc, Diag::DiagID id,
                                               Args &&...args) {
  const Diag::DiagnosticInfo *info = Diag::DiagnosticRegistry::get().lookup(id);
  assert(info && "DiagID not registered in DiagnosticRegistry");
  auto diag = emitWarning(loc, llvm::Twine());
  detail::applyDiagTemplate(diag, info->messageTemplate,
                            std::forward<Args>(args)...);
  return diag;
}

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_MOJODIAGS_H
