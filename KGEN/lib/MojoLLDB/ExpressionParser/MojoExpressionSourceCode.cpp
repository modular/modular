//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoExpressionSourceCode.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace M::KGEN::Mojo;

/// Return true if the given line matches any of the given prefixes.
template <typename Prefixes>
static bool matchesAnyPrefix(StringRef line, const Prefixes &prefixes) {
  return llvm::any_of(
      prefixes, [&](StringRef prefix) { return line.starts_with(prefix); });
}

static bool isFunctionOrStructDeclaration(StringRef code) {
  static constexpr auto kPrefixes = {"fn ",
                                     "def ",
                                     "struct ",
                                     "@adaptive",
                                     "@always_inline",
                                     "@export",
                                     "@register_passable"};
  return matchesAnyPrefix(code, kPrefixes);
}

static bool isIndented(StringRef code) {
  static constexpr auto kPrefixes = {" ", "\t"};
  return matchesAnyPrefix(code, kPrefixes);
}

static bool isSimpleImport(StringRef code) {
  // `import` is a reserved keyword.
  return code.starts_with("import ");
}

static bool isFromImport(StringRef code) {
  // `from` is a reserved keyword.
  return code.starts_with("from ");
}

static bool isAlias(StringRef code) { return code.starts_with("alias "); }

static bool isOpenParenthesis(char c) { return c == '(' || c == '['; }

static bool isCloseParenthesis(char c) { return c == ')' || c == ']'; }

/// Parse the beginning of `unparsedCode` as a simple `import *` statement. If
/// the parsing fails, false is returned. `unparsedCode` is modified to point to
/// the next statement is the parsing was successful, in which case true is
/// returned.
static bool tryHandleSimpleImport(StringRef &unparsedCode,
                                  llvm::raw_string_ostream &topLevelOS) {
  if (!isSimpleImport(unparsedCode))
    return false;
  // It seems that mojo doesn't support simple imports yet.
  auto [line, rest] = unparsedCode.split("\n");
  topLevelOS << line << "\n";
  unparsedCode = rest;
  return true;
}

/// Parse the beginning of `unparsedCode` as a `from * import` statement, a
/// `fn`, a `def` or a `struct` top level statement. If the parsing fails, false
/// is returned. `unparsedCode` is modified to point to the next statement is
/// the parsing was successful, in which case true is returned.
static bool
tryHandleFromImportAliasFunctionOrStruct(StringRef &unparsedCode,
                                         llvm::raw_string_ostream &topLevelOS) {
  bool isFunctionOrStruct = isFunctionOrStructDeclaration(unparsedCode);
  if (!isFunctionOrStruct && !isFromImport(unparsedCode) &&
      !isAlias(unparsedCode))
    return false;

  // These statements can have a hierarchy of () or [], so we need to parse
  // until we have visited all of them.

  // If we are in a function or struct, we also need to find a : outside of any
  // parenthesis.
  bool requiresOuterColon = isFunctionOrStruct;

  // The following block will find the top declaration and not the body of the
  // entity we are parsing. For example, if we have the function
  //
  //   fn foo() -> Int:
  //     return 12
  //
  // then this block find the `fn foo() -> Int:\n`, even if it's split across
  // many lines. The body will be handled later.
  {
    // This is an iterator of the unparsed code.
    size_t pos = 0;
    // This counts how many unmatched ( or [ we have found so far.
    size_t openings = 0;
    for (size_t end = unparsedCode.size(); pos < end; ++pos) {
      if (unparsedCode[pos] == '\n' && openings == 0 && !requiresOuterColon)
        break;

      if (isOpenParenthesis(unparsedCode[pos]))
        ++openings;
      else if (isCloseParenthesis(unparsedCode[pos]))
        --openings;
      else if (unparsedCode[pos] == ':' && openings == 0)
        requiresOuterColon = false;
    }
    topLevelOS << unparsedCode.substr(0, pos + 1);
    unparsedCode = unparsedCode.substr(pos + 1);
  }

  if (isFunctionOrStruct) {
    // We now absorb all indented code included empty lines, which make the body
    // of the entity we are parsing. This doesn't apply to aliases, for example.
    while (!unparsedCode.empty()) {
      auto [line, rest] = unparsedCode.split("\n");
      if (!line.empty() && !isIndented(line))
        break;
      unparsedCode = rest;
      topLevelOS << line << "\n";
    }
  }
  return true;
}

MojoExpressionSourceCode::MojoExpressionSourceCode(StringRef exprText) {
  llvm::raw_string_ostream topLevelOS(topLevelCode), mainBodyOS(mainBodyCode);

  StringRef unparsedCode = exprText;

  /// The following code will consume chunks of code assigning them to either
  /// the top-level or the main body sections.
  while (!unparsedCode.empty()) {
    // Note: We are not yet handling multiline expressions with \.
    if (!tryHandleFromImportAliasFunctionOrStruct(unparsedCode, topLevelOS) &&
        !tryHandleSimpleImport(unparsedCode, topLevelOS)) {
      // Any other case is just main body code.
      auto [line, rest] = unparsedCode.split("\n");
      mainBodyOS << line << "\n";
      unparsedCode = rest;
    }
  }

  auto ensureEOLTerminated = [](auto &str) {
    if (!str.empty() && str.back() != '\n')
      str += '\n';
  };

  ensureEOLTerminated(topLevelCode);
  ensureEOLTerminated(mainBodyCode);
}
