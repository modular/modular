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
#include <functional>
#include <string>

namespace llvm {
class SourceMgr;
class SMFixIt;
} // namespace llvm

namespace M::KGEN::LIT {
using llvm::SMFixIt;
using llvm::SMLoc;
using llvm::SourceMgr;
class LitDiagnostic;
class LitSourceRange;
class LitFixIt;

class LitDiags {
public:
  LitDiags(SourceMgr &sourceMgr, MLIRContext *context, bool useMLIRDiagnostics);
  ~LitDiags();

  llvm::SourceMgr &sourceMgr;
  MLIRContext *const context;

  /// Return the identifier for the main buffer in the SourceMgr.
  StringAttr getBufferNameIdentifier() const;

  bool isErrorEmitted() const { return errorEmitted; }

  /// Emit an error.
  LitDiagnostic emitError(Location loc, const Twine &message);
  LitDiagnostic emitError(llvm::SMLoc loc, const Twine &message);

  /// Emit a warning.
  LitDiagnostic emitWarning(Location loc, const Twine &message);
  LitDiagnostic emitWarning(llvm::SMLoc loc, const Twine &message);

  /// Encode the specified source location information into a Location object
  /// for attachment to the IR or error reporting.  This always returns a
  /// FileLineColLoc.
  Location translateLocation(llvm::SMLoc loc) const;

  /// This is a helper object that allows turning Location objects into SMLoc's.
  class SourceMgrLocationMapper;
  std::unique_ptr<SourceMgrLocationMapper> sourceMgrMapper;

  /// Specify a function used to adjust the end-point of a token given a pointer
  /// to the start of the token.
  void setTokenEndPointAdjustmentFn(std::function<void(SMLoc &)> fn) {
    tokenEndPointAdjustmentFn = std::move(fn);
  }

  std::function<void(SMLoc &)> tokenEndPointAdjustmentFn;

  /// This is the StringAttr for the main buffer identifier.  It is type erased
  /// to void* to reduce header polution.
  const void *const bufferNameIdentifier;

  /// This is true if we should use MLIR for diagnostics (e.g. to enable
  /// -verify-diagnostics and other MLIR testing features), but we prefer
  /// llvm::SourceMgr for better QoI: it supports source ranges and FixIt hints.
  const bool useMLIRDiagnostics;

private:
  friend class LitDiagnostic;
  LitDiags(const LitDiags &) = delete;

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
  LitDiagnostic(Location loc, LitDiags &diags, bool isWarning);
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
  // messages.  These are implemented with addToDiagnostic methods.
  template <typename Arg>
  LitDiagnostic &operator<<(Arg &&value) & {
    addToDiagnostic(std::forward<Arg>(value), *this);
    return *this;
  }
  template <typename Arg>
  LitDiagnostic operator<<(Arg value) && {
    addToDiagnostic(std::forward<Arg>(value), *this);
    return std::move(*this);
  }

  /// This method can be used by addToDiagnostic impls to add things to the
  /// diagnostic.
  void addText(const Twine &text);
  void addSourceRange(LitSourceRange range);
  void addFixIt(LitFixIt fixIt);

private:
  void emitMLIRDiagnostic();
  void emitSourceMgrDiagnostic();

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

  /// True if this is a warning, false if this is an error.
  bool isWarning;
};

/// Represents a range in source code.  The default use-case for this class is
/// to represent source ranges in terms of lit token positions beginnings, where
/// the start/end of the range indicate the beginning of the tokens included in
/// the range.  This makes it easier to construct and work with.
///
/// For example, in the expression `yoda + 492`, a range with the start/end both
/// pointing to the 'y', would indicate the full identifier "yoda".  Similarly,
/// a range pointing to the 'y' and the '4' would cover the entire span from the
/// start of y through the end of 2.
///
/// There are some narrow cases where you may want to diagnose within a token,
/// e.g. complaining about a format character in a string literal.  In those
/// cases, you may use a 'byte-level' string, which uses a half-open range and
/// is not extended to include the end of the token.
class LitSourceRange {
public:
  /// Build a null range.
  LitSourceRange() = default;

  /// Build a normal token-start range.
  LitSourceRange(SMLoc start, SMLoc end);

  /// Build a byte-level range.
  static LitSourceRange getByteLevel(SMLoc start, SMLoc end);

  SMLoc getStart() const;
  SMLoc getEnd() const;
  bool isValid() const { return start != nullptr; }
  bool isByteLevel() const { return byteLevel; }

private:
  const char *start = nullptr, *end = nullptr;
  bool byteLevel = false;
};

/// A FixIt hint is a source rewrite that some IDEs can apply automatically when
/// errors occur.  Generation of FixIt hints is great for QoI.  Error recovery
/// in the parser must always follow the logic that would have happened if the
/// FixIt was applied so the user doesn't get a different downstream error after
/// applyign the FixIt hint.
class LitFixIt {
public:
  LitFixIt(LitSourceRange range, const Twine &replacement);

  /// This is the source range to remove.
  LitSourceRange range;
  /// This is what to replace it with.
  std::string replacement;
};

// These methods enable adding common types to the current diagnostic.
void addToDiagnostic(const Twine &text, LitDiagnostic &diag);
void addToDiagnostic(char text, LitDiagnostic &diag);
void addToDiagnostic(size_t number, LitDiagnostic &diag);
void addToDiagnostic(StringAttr attr, LitDiagnostic &diag);

/// This adds a source range highlight.
void addToDiagnostic(LitSourceRange range, LitDiagnostic &diag);
/// This adds a fixit hint.
void addToDiagnostic(LitFixIt fixIt, LitDiagnostic &diag);

} // namespace M::KGEN::LIT

#endif // LITDIAGS_H
