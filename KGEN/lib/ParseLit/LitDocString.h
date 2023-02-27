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

#include "LitLexer.h"
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
  StringRef getDescription() const { return description; }

private:
  /// The short summary of the doc string.
  std::string summary;
  /// The description of the doc string if it's a multi-line doc, empty for
  /// single-line doc.
  std::string description;
};

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

/// Generate markdown documentation for the given decl.
void generateLitMarkdownDoc(ASTDecl &decl, raw_ostream &os);

} // namespace M::KGEN::LIT

#endif // LITDOCSTRING_H
