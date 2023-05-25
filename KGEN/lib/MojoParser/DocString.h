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

#ifndef DOCSTRING_H
#define DOCSTRING_H

#include "Lexer.h"
#include "SharedState.h"
#include "Support/ADT/SmartVariant.h"

namespace M::KGEN::LIT {
//===----------------------------------------------------------------------===//
// DocString
//===----------------------------------------------------------------------===//

/// This class represents a processed Mojo doc string.
class DocString {
public:
  /// Construct a new DocString from a given raw doc-string.
  DocString(StringRef rawDocString);

  /// Return the summary of the doc string.
  StringRef getSummary() const { return summary; }

  /// Return the fully body description of the doc string.
  ArrayRef<StringRef> getDescription() const { return descriptionLines; }

  /// Return the beginning location of the doc string.
  SMLoc getLoc() const { return loc; }

private:
  /// The short summary of the doc string.
  std::string summary;

  /// The lines comprising the description.
  SmallVector<StringRef> descriptionLines;

  /// The beginning location of the doc string.
  SMLoc loc;
};

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

/// Generate a JSON representation of the documentation for the given decl, and
/// write it to the given output stream. The output of the generation is defined
/// in the following format:
///
/// Module:
/// {
///   "kind": "module",
///   "name": "...",
///   "summary": "...",
///   "description": "...",
///   "aliases": [ ... ],
///   "functions": [ ... ],
///   "structs": [ ... ]
/// }
///
/// Struct:
/// {
///   "kind": "struct",
///   "name": "...",
///   "summary": "...",
///   "description": "...",
///   "parameters": [
///     {
///       "name": "bar",
///       "type": "Int",
///       "description": "...",
///     }
///   ],
///   "aliases": [ ... ],
///   "functions": [ ... ],
///   "structs": [ ... ]
/// }
///
/// Function:
/// {
///   "kind": "function",
///   "name": "baz",
///   "overloads": [
///     {
///       "signature": "baz() -> Int",
///       "summary": "...",
///       "description": "...",
///       "args": [
///         {
///           "name": "foo",
///           "type": "Int",
///           "description": "...",
///         }
///       ]
///       "parameters": [
///         {
///           "name": "bar",
///           "type": "Int",
///           "description": "...",
///         }
///       ],
///       "returns": "...",
///       "constraints": "..."
///     }
///   ]
/// }
///
/// Alias:
///  {
///    "kind": "alias",
///    "name": "...",
///    "value": "...",
///    "summary": "...",
///    "description": "..."
///  }
///
void generateMojoDocJSON(ASTDecl &decl, raw_ostream &os);

/// Validate the doc string for the given decl, emitting warnings for any
/// invalid format issues.
void validateDocString(SharedState &sharedState, ASTDecl &decl);

} // namespace M::KGEN::LIT

#endif // DOCSTRING_H
