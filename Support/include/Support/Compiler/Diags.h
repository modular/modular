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

#ifndef SUPPORT_COMPILER_DIAGS_H
#define SUPPORT_COMPILER_DIAGS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include <functional>
#include <string>
#include <vector>

namespace llvm {
class SourceMgr;
class SMFixIt;
} // namespace llvm

namespace M {
using llvm::SMFixIt;
using llvm::SMLoc;
using llvm::SourceMgr;
class InflightDiag;
class SourceRange;
class FixIt;

class Diags {
public:
  Diags(SourceMgr &sourceMgr, MLIRContext *context, bool useMLIRDiagnostics,
        int maxNotesPerDiagnostic);
  ~Diags();

  llvm::SourceMgr &sourceMgr;
  MLIRContext *const context;

  /// Return the identifier for the main buffer in the SourceMgr.
  StringAttr getBufferNameIdentifier() const;

  bool isErrorEmitted() const { return errorEmitted; }

  bool isDiagnosticEmitted() const { return diagnosticEmitted; }

  /// Clear out the current diagnostic state.
  void clear() { errorEmitted = diagnosticEmitted = false; }

  /// Emit an error.
  InflightDiag emitError(Location loc, const Twine &message);
  InflightDiag emitError(llvm::SMLoc loc, const Twine &message);

  /// Emit a warning.
  InflightDiag emitWarning(Location loc, const Twine &message);
  InflightDiag emitWarning(llvm::SMLoc loc, const Twine &message);

  /// Encode the specified source location information into a Location object
  /// for attachment to the IR or error reporting.  This always returns a
  /// FileLineColLoc.
  Location translateLocation(llvm::SMLoc loc) const;

  /// Decode the specific MLIR location information into an SMLoc for use with
  /// the SourceMgr. This returns an invalid SMLoc if the location is not
  /// understood.
  SMLoc convertLocToSMLoc(LocationAttr loc) const;

  /// Convert the given source range to an SMRange.
  llvm::SMRange convertToSMRange(SourceRange range) const;

  /// This is a helper object that allows turning Location objects into SMLoc's.
  class SourceMgrLocationMapper;
  std::unique_ptr<SourceMgrLocationMapper> sourceMgrMapper;

  /// Specify a function used to adjust the end-point of a token given a pointer
  /// to the start of the token.
  void setTokenEndPointAdjustmentFn(std::function<void(SMLoc &)> fn) {
    tokenEndPointAdjustmentFn = std::move(fn);
  }

  std::function<void(SMLoc &)> tokenEndPointAdjustmentFn;

  /// This is true if we should use MLIR for diagnostics (e.g. to enable
  /// -verify-diagnostics and other MLIR testing features), but we prefer
  /// llvm::SourceMgr for better QoI: it supports source ranges and FixIt hints.
  const bool useMLIRDiagnostics;

private:
  friend class InflightDiag;
  Diags(const Diags &) = delete;

  /// This is the StringAttr for the main buffer identifier. It is type erased
  /// to void* to reduce header polution. This field is lazy initialized to
  /// handle the case where the main buffer is added after the Diags object is
  /// constructed.
  mutable std::optional<const void *> bufferNameIdentifier;

  /// This is set to true if an error occurred at any point processing the
  /// file.
  bool errorEmitted = false;

  /// This is set to true if any diagnostic occurred at any point processing the
  /// file.
  bool diagnosticEmitted = false;

  /// Configuration for how many notes to print for a diagnostic.
  int maxNotesPerDiagnostic;

  /// This is a StringAttr for an unknown buffer name. It is type erased to
  /// void* to reduce header polution.
  const void *unknownBufferNameIdentifier;
};

/// This class represents a diagnostic that is built up by the parser and
/// emitted when destroyed.  This allows clients to incrementally build up the
/// message, attach notes, ranges and fixit hints all with a simple interface.
///
/// Each diagnostic is made up of a primary message and is optionally followed
/// by any number of note messages.
class InflightDiag {
public:
  InflightDiag(Location loc, Diags &diags, bool isWarning);
  ~InflightDiag();
  InflightDiag(InflightDiag &&other);
  InflightDiag &operator=(InflightDiag &&other);

  // This class in non-copyable.
  InflightDiag(const InflightDiag &) = delete;
  InflightDiag &operator=(const InflightDiag &) = delete;

  /// Abandon emission of this message, this will make it be a noop when its
  /// destructor runs.
  void abandon() { diags = nullptr; }

  // InflightDiag always converts to failure.  This allows certain patterns to
  // be more ergonomic.
  operator LogicalResult() const { return failure(); }
  operator ParseResult() const { return failure(); }

  /// Add a note to this diagnostic at the specified location, and change the
  /// emission point to start filling it in.
  InflightDiag attachNote(Location loc) &&;
  InflightDiag &attachNote(Location loc) &;
  InflightDiag attachNote(SMLoc loc) &&;
  InflightDiag &attachNote(SMLoc loc) &;

  // Insertion operations for various things that contribute to the current
  // messages.  These are implemented with addToDiagnostic methods.
  template <typename Arg>
  InflightDiag &operator<<(Arg &&value) & {
    addToDiagnostic(std::forward<Arg>(value), *this);
    return *this;
  }
  template <typename Arg>
  InflightDiag operator<<(Arg value) && {
    addToDiagnostic(std::forward<Arg>(value), *this);
    return std::move(*this);
  }

  /// These methods can be used by addToDiagnostic impls to add things to the
  /// diagnostic.
  void addText(const Twine &text);
  void addSourceRange(SourceRange range);
  void addFixIt(FixIt fixIt);
  void addDiag(InflightDiag &&otherDiag);

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
  Diags *diags;

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
class SourceRange {
public:
  /// Build a null range.
  SourceRange() = default;

  /// Build a normal token-start range.
  SourceRange(SMLoc start, SMLoc end);
  SourceRange(llvm::SMRange range);

  /// Build a byte-level range.
  static SourceRange getByteLevel(SMLoc start, SMLoc end);

  SMLoc getStart() const;
  SMLoc getEnd() const;
  bool isValid() const { return start != nullptr; }
  bool isByteLevel() const { return byteLevel; }

  /// Return an SMRange that corresponds to this source range.
  llvm::SMRange getSMRange() const;

private:
  const char *start = nullptr, *end = nullptr;
  bool byteLevel = false;
};

/// A FixIt hint is a source rewrite that some IDEs can apply automatically when
/// errors occur.  Generation of FixIt hints is great for QoI.  Error recovery
/// in the parser must always follow the logic that would have happened if the
/// FixIt was applied so the user doesn't get a different downstream error after
/// applying the FixIt hint.
class FixIt {
public:
  FixIt(SourceRange range, const Twine &replacement);

  /// This constructor creates a fixit that removes the specified token.
  static FixIt remove(SMLoc loc);

  /// This constructor creates a fixit that removes the specified token range.
  static FixIt remove(SourceRange range);

  /// This constructor creates a fixit that replaces the one token at the
  /// specified location with some text.
  static FixIt replaceToken(SMLoc loc, const Twine &text);

  /// This constructor creates a fixit that inserts some text before the token
  /// at the specified location, without replacing the token.
  static FixIt insertBeforeToken(SMLoc loc, const Twine &text);

  /// This constructor creates a fixit that inserts some text after the token
  /// at the specified location.
  static FixIt insertAfterToken(SMLoc loc, const Twine &text, Diags &diags);

  /// This is the source range to remove.
  SourceRange range;
  /// This is what to replace it with.
  std::string replacement;
};

// These methods enable adding common types to the current diagnostic.
void addToDiagnostic(const Twine &text, InflightDiag &diag);
void addToDiagnostic(char text, InflightDiag &diag);
void addToDiagnostic(size_t number, InflightDiag &diag);
void addToDiagnostic(StringAttr attr, InflightDiag &diag);

/// This adds a source range highlight.
void addToDiagnostic(SourceRange range, InflightDiag &diag);
/// This adds a fixit hint.
void addToDiagnostic(FixIt fixIt, InflightDiag &diag);
/// This concatenates another diagnostic.
void addToDiagnostic(InflightDiag &&otherDiag, InflightDiag &diag);

} // namespace M

#endif // SUPPORT_COMPILER_DIAGS_H
