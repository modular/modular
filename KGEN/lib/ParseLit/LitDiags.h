//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the preferred diagnostics machinery used by the frontend.
// It is intentionally similar to the MLIR diagnostics machinery, but disallows
// printing MLIR types (forcing use of ASTTypes) and supports additional
// features like source ranges.
//
//===----------------------------------------------------------------------===//

#ifndef LITDIAGS_H
#define LITDIAGS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace M::KGEN::LIT {
using llvm::SMLoc;
using llvm::SourceMgr;

class LitDiagnostic;

class LitDiags {
public:
  LitDiags(SourceMgr &sourceMgr, MLIRContext *context);

  llvm::SourceMgr &sourceMgr;
  MLIRContext *const context;

  /// Return the identifier for the main buffer in the SourceMgr.
  StringAttr getBufferNameIdentifier() const;

  bool isErrorEmitted() const { return errorEmitted; }

  /// Emit an error through the parser's logic.
  LitDiagnostic emitError(Location loc, const Twine &message);

  /// Emit an error through the parser's logic.
  LitDiagnostic emitError(llvm::SMLoc loc, const Twine &message);

  /// Encode the specified source location information into a Location object
  /// for attachment to the IR or error reporting.  This always returns a
  /// FileLineColLoc.
  Location translateLocation(llvm::SMLoc loc) const;

private:
  /// This is the StringAttr for the main buffer identifier.  It is type erased
  /// to void* to reduce header polution.
  const void *const bufferNameIdentifier;

  /// This is set to true if an error occurred at any point processing the
  /// file.
  bool errorEmitted = false;
};

/// This class represents a diagnostic that is built up by the parser and
/// emitted when destroyed.  This allows clients to incrementally build up the
/// message, attach notes, ranges and fixit hints all with a simple interface.
///
/// Each diagnostic is made up of a primary message and is optionally followed
/// by any number of note messages.
class LitDiagnostic {
public:
  LitDiagnostic(Location loc, LitDiags &diags);
  ~LitDiagnostic();
  LitDiagnostic(LitDiagnostic &&other);

  /// Abandon emission of this message, this will make it be a noop when its
  /// destructor runs.
  void abandon() { diags = nullptr; }

  // LitDiagnostic always converts to failure.  This allows certain patterns to
  // be more ergonomic.
  operator LogicalResult() const { return failure(); }
  operator ParseResult() const { return failure(); }

  /// Add a note to this diagnostic at the specified location, and change the
  /// emission point to start filling it in.
  LitDiagnostic attachNote(Location loc) &&;
  LitDiagnostic &attachNote(Location loc) &;

  // Insertion operations for various things that contribute to the current
  // messages's text.  These are implemented with appendText methods.
  template <typename Arg>
  LitDiagnostic &operator<<(Arg &&value) & {
    appendText(std::forward<Arg>(value), *this);
    return *this;
  }
  template <typename Arg>
  LitDiagnostic operator<<(Arg value) && {
    appendText(std::forward<Arg>(value), *this);
    return std::move(*this);
  }

  /// This method can be used by appendText methods to add things to the
  /// diagnostic.
  void addText(const Twine &text);

private:
  /// Each message in a diagnostic must have a location and text, and may
  /// have any number of highlighted ranges and fixit hints.
  struct Message;

  // We store the primary diagnostic and any notes in this vector.  The primary
  // diagnostic is always first, and always present.  This uses a std::vector
  // so Message can be defined out of line, and to make the move operation
  // efficient.
  std::vector<Message> messages;

  /// This is the diagnostic object to emit to, or null if abandoned.
  LitDiags *diags;
};

// Allow inserting string-like things.
void appendText(const Twine &text, LitDiagnostic &diag);
void appendText(char text, LitDiagnostic &diag);
void appendText(size_t number, LitDiagnostic &diag);
void appendText(StringAttr text, LitDiagnostic &diag);
void appendText(Attribute attr, LitDiagnostic &diag);

} // namespace M::KGEN::LIT

#endif // LITDIAGS_H
