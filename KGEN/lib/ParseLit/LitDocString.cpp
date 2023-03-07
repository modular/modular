//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LitDocString.h"
#include "ASTDecl.h"
#include "KGEN/LITDialect/LITOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Return the indentation level of the first line of the string.
static size_t getIndentationLevel(StringRef str) {
  return str.size() - str.ltrim().size();
}

//===----------------------------------------------------------------------===//
// LitDocString
//===----------------------------------------------------------------------===//

LitDocString::LitDocString(StringRef rawDocString) {
  // This function processes a doc-string, following a similar structure as
  // defined by PEP 257 for how multi-line doc strings should be formatted.
  // https://peps.python.org/pep-0257/#multi-line-docstrings

  // Split the doc string into lines.
  SmallVector<StringRef> lineStorage;
  rawDocString.split(lineStorage, '\n');
  MutableArrayRef<StringRef> lines(lineStorage);
  if (lines.empty())
    return;

  // Determine the minimum indentation (first line doesn't count).
  size_t indent = std::numeric_limits<size_t>::max();
  for (auto &line : lines.drop_front())
    if (size_t lineIndent = getIndentationLevel(line))
      indent = std::min(indent, lineIndent);

  // Remove the necessary indentation from all but the first line, which has all
  // leading whitespace removed.
  lines[0] = lines[0].trim();
  if (indent) {
    for (size_t i = 1; i < lines.size(); ++i)
      if (!lines[i].empty())
        lines[i] = lines[i].drop_front(indent).rtrim();
  }

  // Strip off trailing and leading blank lines.
  while (!lines.empty() && lines.back().empty())
    lines = lines.drop_back(1);
  while (!lines.empty() && lines.front().empty())
    lines = lines.drop_front(1);

  // If the docstring is empty, there is nothing to do.
  if (lines.empty())
    return;

  // Join the lines back together.
  size_t line = 0, lineE = lines.size();

  // We treat up to the first blank line as the summary, and join together on a
  // single line.
  llvm::raw_string_ostream summaryOS(summary);
  summaryOS << lines[line++];
  while (line < lineE && !lines[line].empty())
    summaryOS << " " << lines[line++];

  // Skip any blank lines.
  while (line < lineE && lines[line].empty())
    ++line;

  // The remaining lines are the description.
  llvm::raw_string_ostream descOS(description);
  while (line < lineE)
    descOS << lines[line++] << "\n";
}

//===----------------------------------------------------------------------===//
// Markdown Generation
//===----------------------------------------------------------------------===//

namespace {
class MarkdownGenerator {
public:
  MarkdownGenerator(raw_ostream &os) : os(os) {}

  void generate(ASTDecl &decl) {
    TypeSwitch<ASTDecl &>(decl).Case<FileModuleOp, FuncOp, StructDeclOp>(
        [&](auto op) { generateLitMarkdownDocFor(decl, op); });
  }

private:
  //===----------------------------------------------------------------------===//
  // Utils

  void generateLitMarkdownForChildren(ASTDecl &decl) {
    SmallVector<std::pair<StringAttr, TinyPtrVector<ASTDecl *>>> children(
        decl.getDeclsInScope().begin(), decl.getDeclsInScope().end());
    llvm::sort(children, [](auto &lhs, auto &rhs) {
      return lhs.first.getValue() < rhs.first.getValue();
    });

    llvm::SaveAndRestore<size_t> saveDepth(depth, depth + 1);
    for (auto &child : children) {
      for (auto &childDecl : child.second) {
        // Skip declarations that were imported from other scopes.
        // TODO: We should note that these are imported/aliases.
        if (childDecl->getParentDecl() != &decl)
          continue;

        generate(*childDecl);
      }
    }
  }

  /// Extract a LitDocString from a given decl, or None if there is no doc
  /// string.
  std::optional<LitDocString> getLitDocString(ASTDecl &decl) {
    StringRef docStr = decl.getDocString();
    if (docStr.empty())
      return std::nullopt;
    return LitDocString(docStr);
  }

  /// Return if the given name should be hidden from the markdown output.
  bool shouldHideName(StringRef name) {
    // Non-underscore names are never hidden.
    if (!name.startswith("_"))
      return false;

    // Keep special language names, which have leading and trailing underscores,
    // even though they start with `_`.
    return !(name.startswith("__") && name.endswith("__"));
  }

  void generateMarkdownHeader(function_ref<void()> nameFn) {
    for (size_t i = 0; i < depth; ++i)
      os.write('#');
    os << " ";
    nameFn();
    os << "\n\n";
  }
  void generateMarkdownHeader(const Twine &name) {
    generateMarkdownHeader([&] { os << name; });
  }

  /// Generate a table from the given form:
  ///
  /// Header:
  ///   Element1: ...
  ///   Element2: ...
  ///     ...
  ///   ElementN: ...
  ///
  void generateTwoColumnMarkdownTable(
      StringRef header, ArrayRef<StringRef> lines, size_t &line, size_t lineE,
      function_ref<void(StringRef)> processEntryName) {
    os << "| " << header << " | |\n";
    os << "| :---- | --- |\n";
    for (++line; line < lineE && !lines[line].empty();) {
      // Extract the argument name and description.
      auto [argName, argDesc] = lines[line].split(':');
      argName = argName.trim();
      argDesc = argDesc.trim();

      os << "| ";
      processEntryName(argName);
      os << " | " << argDesc;

      // Merge in additional description lines that have a larger
      // indentation.
      size_t indent = getIndentationLevel(lines[line]);
      while (++line < lineE && getIndentationLevel(lines[line]) > indent)
        os << " " << lines[line].trim();
      os << " |\n";
    }
    os << "\n\n";
  }
  void generateTwoColumnMarkdownTable(StringRef header,
                                      ArrayRef<StringRef> lines, size_t &line,
                                      size_t lineE) {
    generateTwoColumnMarkdownTable(
        header, lines, line, lineE,
        [&](StringRef entryName) { os << entryName; });
  }

  /// Generate a table from the given form:
  ///
  /// Header:
  ///   Element1...
  ///   Element2...
  ///     ...
  ///   ElementN...
  ///
  void generateSingleColumnMarkdownTable(const Twine &header,
                                         ArrayRef<StringRef> lines,
                                         size_t &line, size_t lineE) {
    os << "| " << header << " |\n";
    os << "| :---- |\n";
    for (++line; line < lineE && !lines[line].empty();) {
      // Extract the argument name and description.
      os << "| " << lines[line].trim();

      // Merge in additional description lines that have a larger
      // indentation.
      size_t indent = getIndentationLevel(lines[line]);
      while (++line < lineE && getIndentationLevel(lines[line]) > indent)
        os << " " << lines[line].trim();
      os << " |\n";
    }
    os << "\n\n";
  }
  /// Generate a table from the given form:
  ///
  /// Header:
  ///   Element1...
  void generateSingleEntrySingleColumnMarkdownTable(const Twine &header,
                                                    ArrayRef<StringRef> lines,
                                                    size_t &line,
                                                    size_t lineE) {
    os << "| " << header << " |\n";
    os << "| :---- |\n";

    // Extract the argument name and description.
    os << "| " << lines[++line].trim();

    // Merge in additional description lines that have equal or larger
    // indentation.
    size_t indent = getIndentationLevel(lines[line]);
    while (++line < lineE && getIndentationLevel(lines[line]) >= indent)
      os << " " << lines[line].trim();

    os << " |\n\n\n";
  }

  //===----------------------------------------------------------------------===//
  // Parameters

  /// Generate a markdown table for a parameter section of a doc-string, using
  /// the provided mapping for types and verification of parameters.
  void
  generateParameterTable(const llvm::StringMap<std::string> &paramNameToType,
                         ArrayRef<StringRef> lines, size_t &line,
                         size_t lineE) {
    auto processParam = [&](StringRef paramName) {
      os << paramName;

      // TODO: Emit errors when we encounter unknown parameters.
      auto it = paramNameToType.find(paramName);
      if (it != paramNameToType.end())
        os << " (" << it->second << ")";
    };
    generateTwoColumnMarkdownTable("Parameters", lines, line, lineE,
                                   processParam);
  }

  //===----------------------------------------------------------------------===//
  // Types

  std::string generateTypeString(Type type) {
    return ASTType(type).getAsString();
  }

  //===----------------------------------------------------------------------===//
  // Function Markdown Generation

  /// Generate markdown documentation for the given function.
  void generateLitMarkdownDocFor(ASTDecl &decl, FuncOp funcOp) {
    // Strip off the mangled suffix from the base function name.
    StringRef name =
        funcOp.getName().take_until([](char c) { return c == '('; });
    if (shouldHideName(name))
      return;

    bool isNonStaticMethod =
        isa<StructDeclOp>(funcOp->getParentOp()) && !funcOp.getIsStatic();
    auto argTypes = funcOp.getArgumentTypes();
    auto argNames = funcOp.getValueParamNames();

    // If this is a non-static method, drop the self argument.
    if (isNonStaticMethod) {
      argTypes = argTypes.drop_front();
      argNames = argNames.drop_front();
    }

    // Grab the types of the arguments to the function.
    SmallVector<std::string> argTypeNames;
    for (Type argType : argTypes)
      argTypeNames.push_back(generateTypeString(argType));
    llvm::StringMap<StringRef> argNameToType;
    for (auto [index, value] : llvm::enumerate(argNames)) {
      if (index < argTypeNames.size())
        argNameToType[value] = argTypeNames[index];
    }

    // Grab the types of the parameters to the function.
    llvm::StringMap<std::string> paramNameToType;
    for (auto [index, value] : llvm::enumerate(funcOp.getInputParams()))
      paramNameToType[value.getName()] = generateTypeString(value.getType());

    // Grab the result type, if it's non-none.
    std::optional<std::string> resultTypeName;
    Type resultType = funcOp.getResultType();
    if (!resultType.isa<LIT::NoneType>())
      resultTypeName = generateTypeString(resultType);

    generateMarkdownHeader([&] {
      // Strip off the mangled suffix from the base function name.
      os << "`` " << name.split('(').first;

      os << "(";
      interleaveComma(argTypeNames, os);
      os << ")";

      if (resultTypeName)
        os << " -> " << *resultTypeName;

      os << " ``:";
    });

    if (std::optional<LitDocString> docStr = getLitDocString(decl)) {
      os << docStr->getSummary() << "\n\n";
      processFunctionDocDescription(docStr->getDescription(), paramNameToType,
                                    argNameToType, resultTypeName);
    }

    // Recursively generate documentation for the module's children.
    generateLitMarkdownForChildren(decl);
  }

  void processFunctionDocDescription(
      StringRef description,
      const llvm::StringMap<std::string> &paramNameToType,
      const llvm::StringMap<StringRef> &argNameToType,
      const std::optional<std::string> &returnType) {
    // Split the description into lines, so we can process the different
    // sections.
    SmallVector<StringRef, 4> lines;
    description.split(lines, '\n');

    // Process the lines of the description, looking for markers.
    size_t line = 0, lineE = lines.size();
    for (; line < lineE; ++line) {
      if (lines[line] == "Args:") {
        auto processArg = [&](StringRef argName) {
          os << argName;

          // TODO: Emit errors when we encounter unknown arguments.
          auto it = argNameToType.find(argName);
          if (it != argNameToType.end())
            os << " (" << it->second << ")";
        };
        generateTwoColumnMarkdownTable("Args", lines, line, lineE, processArg);
        continue;
      }
      if (lines[line] == "Parameters:") {
        generateParameterTable(paramNameToType, lines, line, lineE);
        continue;
      }
      if (lines[line] == "Returns:") {
        // TODO: Validate the return type. Check that the function has one when
        // the section is specified, and vice versa.
        generateSingleEntrySingleColumnMarkdownTable(
            "Returns (" + returnType.value_or("") + ")", lines, line, lineE);
        continue;
      }
      if (lines[line] == "Constraints:") {
        generateSingleColumnMarkdownTable("Constraints", lines, line, lineE);
        continue;
      }

      os << lines[line] << "\n";
    }
  }

  //===----------------------------------------------------------------------===//
  // Struct Markdown Generation

  void generateLitMarkdownDocFor(ASTDecl &decl, StructDeclOp structOp) {
    StringRef name = structOp.getName();
    if (shouldHideName(name))
      return;
    generateMarkdownHeader(name);

    if (std::optional<LitDocString> docStr = getLitDocString(decl)) {
      os << docStr->getSummary() << "\n\n";

      // Grab the types of the parameters to the struct.
      llvm::StringMap<std::string> paramNameToType;
      for (auto [index, value] : llvm::enumerate(structOp.getInputParams()))
        paramNameToType[value.getName()] = generateTypeString(value.getType());

      processStructDocDescription(docStr->getDescription(), paramNameToType);
    }

    // Recursively generate documentation for the module's children.
    generateLitMarkdownForChildren(decl);
  }

  void processStructDocDescription(
      StringRef description,
      const llvm::StringMap<std::string> &paramNameToType) {
    // Split the description into lines, so we can process the different
    // sections.
    SmallVector<StringRef, 4> lines;
    description.split(lines, '\n');

    // Process the lines of the description, looking for markers.
    size_t line = 0, lineE = lines.size();
    for (; line < lineE; ++line) {
      if (lines[line] == "Parameters:") {
        generateParameterTable(paramNameToType, lines, line, lineE);
        continue;
      }

      os << lines[line] << "\n";
    }
  }

  //===----------------------------------------------------------------------===//
  // Module Markdown Generation

  void generateLitMarkdownDocFor(ASTDecl &decl, FileModuleOp moduleOp) {
    StringRef name = moduleOp.getName();
    name.consume_front("$");
    generateMarkdownHeader("Module: " + name);

    if (std::optional<LitDocString> docStr = getLitDocString(decl))
      os << docStr->getSummary() << "\n\n" << docStr->getDescription() << "\n";

    // Recursively generate documentation for the module's children.
    generateLitMarkdownForChildren(decl);
  }

  //===----------------------------------------------------------------------===//
  // Fields

  /// The output stream.
  raw_ostream &os;

  /// The current depth of the markdown header.
  size_t depth = 1;
};
} // namespace

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

void M::KGEN::LIT::generateLitMarkdownDoc(ASTDecl &decl, raw_ostream &os) {
  MarkdownGenerator generator(os);
  generator.generate(decl);
}
