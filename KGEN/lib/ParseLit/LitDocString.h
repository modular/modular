//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities for processing and formatting Lit doc strings
// into various formats.
//
//===----------------------------------------------------------------------===//

#ifndef LITDOCSTRING_H
#define LITDOCSTRING_H

#include "Lexer.h"
#include "LitSharedState.h"
#include "Support/ADT/SmartVariant.h"

namespace M::KGEN::LIT {
//===----------------------------------------------------------------------===//
// LitDocString
//===----------------------------------------------------------------------===//

/// This class represents a processed Lit doc string.
class LitDocString {
public:
  /// Construct a new LitDocString from a given raw doc-string.
  LitDocString(StringRef rawDocString);

  /// Return the summary of the doc string.
  StringRef getSummary() const { return summary; }

  /// Return the fully body description of the doc string.
  ArrayRef<StringRef> getDescription() const { return descriptionLines; }

private:
  /// The short summary of the doc string.
  std::string summary;

  /// The lines comprising the description.
  SmallVector<StringRef> descriptionLines;
};

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

/// Generate markdown documentation for the given decl.
void generateLitMarkdownDoc(ASTDecl &decl, raw_ostream &os);

/// Validate the doc string for the given decl, emitting warnings for any
/// invalid format issues.
void validateLitDocString(LitSharedState &sharedState, ASTDecl &decl);

} // namespace M::KGEN::LIT

#endif // LITDOCSTRING_H
