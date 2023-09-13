//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities for processing and formatting Mojo doc strings
// into various formats.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_DOCSTRING_H
#define KGEN_MOJOPARSER_DOCSTRING_H

#include "KGEN/MojoParser/SharedState.h"
#include "Support/ADT/SmartVariant.h"

namespace M::KGEN::LIT {
class DocStringAttr;

//===----------------------------------------------------------------------===//
// DocString
//===----------------------------------------------------------------------===//

/// This class represents a processed Mojo doc string.
class DocString {
public:
  /// Construct a new DocString from a given raw doc-string.
  DocString(DocStringAttr rawDocStringAttr);

  /// Return the summary of the doc string.
  StringRef getSummary() const { return summary; }

  /// Return the fully body description of the doc string.
  ArrayRef<StringRef> getDescription() const { return descriptionLines; }

  /// Return the beginning location of the doc string, or nullptr if the doc
  /// string is not attached to a location.
  FileLineColLoc getLoc() const { return loc; }

  //===----------------------------------------------------------------------===//
  // Section names

  /// Within a doc string, the "Constraints" section describes invariants that
  /// must be true for the struct or function.
  static const char *kSectionConstraints;

  /// Within a doc string, the "Parameters" section lists descriptions of each
  /// parameter.
  static const char *kSectionParameters;

  /// Within a doc string, the "Args" section lists descriptions of each
  /// function argument.
  static const char *kSectionArgs;

  /// Within a doc string, the "Returns" section describes the results of a
  /// function.
  static const char *kSectionReturns;

private:
  /// The short summary of the doc string.
  std::string summary;

  /// The lines comprising the description.
  SmallVector<StringRef> descriptionLines;

  /// The beginning location of the doc string.
  FileLineColLoc loc;
};

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

/// Validate the doc string for the given decl, emitting warnings for any
/// invalid format issues.
void validateDocString(SharedState &sharedState, ASTDecl &decl);

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_DOCSTRING_H
