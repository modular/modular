//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "DocString.h"
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

/// Extract a DocString from a given decl, or None if there is no doc string.
static std::optional<DocString> getDocString(ASTDecl &decl) {
  StringRef docStr = decl.getDocString();
  if (docStr.empty())
    return std::nullopt;
  return DocString(docStr);
}

//===----------------------------------------------------------------------===//
// DocString
//===----------------------------------------------------------------------===//

DocString::DocString(StringRef rawDocString) {
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
  for (StringRef line : lines.drop_front()) {
    if (line.empty())
      continue;
    indent = std::min(indent, getIndentationLevel(line));
    if (indent == 0)
      break;
  }

  // Remove the necessary indentation from all but the first line, which has all
  // leading whitespace removed.
  lines[0] = lines[0].ltrim();
  if (indent) {
    for (size_t i = 1; i < lines.size(); ++i)
      if (!lines[i].empty())
        lines[i] = lines[i].drop_front(indent);
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

  // The remaining lines are the description, or sections there within.
  descriptionLines.append(lines.begin() + line, lines.end());
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

    // Skip declarations that were imported from other scopes.
    // TODO: We should note that these are imported/aliases.
    auto filterChildren = [&](TinyPtrVector<ASTDecl *> &children) {
      return llvm::make_filter_range(children, [&](ASTDecl *child) {
        return child->getParentDecl() == &decl;
      });
    };

    llvm::SaveAndRestore<size_t> saveDepth(depth, depth + 1);
    for (auto &child : children) {
      if (shouldHideName(child.first))
        continue;
      auto filteredChildren = filterChildren(child.second);
      if (filteredChildren.empty())
        continue;

      // If the children are functions, generate all of the overloads under a
      // single header.
      if (isa<FuncOp>(**filteredChildren.begin())) {
        generateLitMarkdownDocForFunctions(child.first, filteredChildren);
        continue;
      }

      for (auto &childDecl : filteredChildren)
        generate(*childDecl);
    }
  }

  /// Extract a DocString from a given decl, or None if there is no doc
  /// string.
  std::optional<DocString> getDocString(ASTDecl &decl) {
    StringRef docStr = decl.getDocString();
    if (docStr.empty())
      return std::nullopt;
    return DocString(docStr);
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

  /// Generate a list section from the given form:
  ///
  /// Header:
  ///   Element1: ...
  ///   Element2: ...
  ///     ...
  ///   ElementN: ...
  ///
  void
  generateMarkdownListSection(StringRef header, ArrayRef<StringRef> lines,
                              size_t &line, size_t lineE,
                              function_ref<void(StringRef)> processEntryName) {
    os << "**" << header << ":**\n\n";
    for (++line; line < lineE && !lines[line].empty();) {
      // Extract the argument name and description.
      auto [argName, argDesc] = lines[line].split(':');
      argName = argName.trim();
      argDesc = argDesc.trim();

      os << "- ";
      processEntryName(argName);
      os << ": " << argDesc;

      // Merge in additional description lines that have a larger
      // indentation.
      size_t indent = getIndentationLevel(lines[line]);
      while (++line < lineE && getIndentationLevel(lines[line]) > indent)
        os << " " << lines[line].trim();
      os << "\n";
    }
    os << "\n\n";
  }
  void generateMarkdownListSection(StringRef header, ArrayRef<StringRef> lines,
                                   size_t &line, size_t lineE) {
    generateMarkdownListSection(header, lines, line, lineE,
                                [&](StringRef entryName) { os << entryName; });
  }

  /// Generate a paragraph section from the given form:
  ///
  /// Header:
  ///   Element1...
  ///   Element2...
  ///     ...
  ///   ElementN...
  ///
  void generateParagraphMarkdownSection(const Twine &header,
                                        ArrayRef<StringRef> lines, size_t &line,
                                        size_t lineE) {
    os << "**" << header << "**:\n\n";
    for (++line; line < lineE && !lines[line].empty();) {
      os << lines[line].trim();

      // Merge in additional description lines that have a larger
      // indentation.
      size_t indent = getIndentationLevel(lines[line]);
      while (++line < lineE && getIndentationLevel(lines[line]) > indent)
        os << " " << lines[line].trim();
      os << "\n";
    }
    os << "\n\n";
  }
  /// Generate a list section from the given form:
  ///
  /// Header:
  ///   Element1...
  void generateSingleEntryMarkdownListSection(StringRef header,
                                              ArrayRef<StringRef> lines,
                                              size_t &line, size_t lineE) {
    os << "**" << header << ":**\n\n";

    // Emit the description.
    os << lines[++line].trim();

    // Merge in additional description lines that have equal or larger
    // indentation.
    size_t indent = getIndentationLevel(lines[line]);
    while (++line < lineE && getIndentationLevel(lines[line]) >= indent)
      os << " " << lines[line].trim();
    os << "\n\n\n";
  }

  //===----------------------------------------------------------------------===//
  // Parameters

  /// Generate a markdown table for a parameter section of a doc-string, using
  /// the provided mapping for types and verification of parameters.
  void generateParameterTable(const llvm::StringMap<std::string> &paramToDetail,
                              ArrayRef<StringRef> lines, size_t &line,
                              size_t lineE) {
    auto processParam = [&](StringRef paramName) {
      auto it = paramToDetail.find(paramName);
      if (it != paramToDetail.end())
        os << "`` " << it->second << " ``";
    };
    generateMarkdownListSection("Parameters", lines, line, lineE, processParam);
  }

  //===----------------------------------------------------------------------===//
  // Types

  /// Generate a documentation string for the given type, with an optional
  /// value convention, parent struct "Self" type, and variable name.
  std::string generateTypeString(
      Type type, std::optional<ASTType> selfType = std::nullopt,
      std::optional<ValueInputConvention> convention = std::nullopt,
      StringRef variableName = "") {
    std::string typeName;
    llvm::raw_string_ostream os(typeName);
    ASTType astType(type);

    // Handle variadic types.
    if (isa<VariadicType>(type)) {
      astType = astType.getVariadicElementType();
      os << "*";
    }

    // Process the variable name if present.
    if (!variableName.empty())
      os << variableName << ": ";

    // Process the convention if present.
    StringRef typeSuffix;
    if (convention) {
      switch (*convention) {
      case ValueInputConvention::ByRef:
      case ValueInputConvention::ByRefResult:
      case ValueInputConvention::InitSelf:
        // TODO: This is probably wrong for ByRefResult?
        astType = astType.getPointerElementType();
        typeSuffix = "&";
        break;
      case ValueInputConvention::OwnedInMem:
      case ValueInputConvention::BorrowedInMem:
        // TODO: Produce "owned" marker in docs.
        astType = astType.getPointerElementType();
        break;
      case ValueInputConvention::OwnedInReg:
      case ValueInputConvention::BorrowedInReg:
        break;
      }
    }

    // If this type is the same as the self type, use the "Self" keyword.
    if (selfType && astType.isEqualCanon(*selfType))
      os << "Self";
    else
      os << astType.getAsString();

    // Append the type suffix.
    os << typeSuffix;
    return os.str();
  }

  //===----------------------------------------------------------------------===//
  // Function Markdown Generation

  /// Generate markdown documentation for the given function.
  void generateLitMarkdownDocFor(ASTDecl &decl, FuncOp funcOp) {
    // Strip off the mangled suffix from the base function name.
    StringRef name =
        funcOp.getName().take_until([](char c) { return c == '('; });
    return generateLitMarkdownDocForFunctions(name, ArrayRef<ASTDecl *>(&decl));
  }
  template <typename DeclRangeT>
  void generateLitMarkdownDocForFunctions(StringRef name,
                                          const DeclRangeT &decls) {
    generateMarkdownHeader(name);
    for (auto *decl : decls)
      generateMarkdownForOverload(*decl, cast<FuncOp>(*decl), name);
  }

  /// Generate a markdown sub-section for the overload described by the given
  /// function.
  void generateMarkdownForOverload(ASTDecl &decl, FuncOp funcOp,
                                   StringRef name) {
    SignatureType signature = funcOp.getSignature();

    auto argTypes = funcOp.getArgumentTypes();
    auto argNames = funcOp.getValueParamNames();
    auto argConventions = signature.getValueInputConventions();
    Type resultType = funcOp.getResultType();
    std::optional<ValueInputConvention> resultConvention;

    // Check for a by-ref result type, which gets modeled as the first argument
    // (as it needs to be passed through memory).
    if (!argConventions.empty() &&
        argConventions.front() == ValueInputConvention::ByRefResult) {
      resultType = argTypes.front();
      argTypes = argTypes.drop_front();
      argNames = argNames.drop_front();
      argConventions = argConventions.drop_front();
      resultConvention = ValueInputConvention::ByRefResult;
    }

    // If this is a method, grab the expected "Self" type.
    std::optional<ASTType> selfType;
    if (isa<StructDeclOp>(funcOp->getParentOp()))
      selfType = decl.getParentDecl()->getSelfType();

    // Grab the types of the arguments to the function.
    SmallVector<std::string> argTypeNames;
    for (auto [index, argType] : llvm::enumerate(argTypes)) {
      argTypeNames.push_back(generateTypeString(
          argType, selfType, argConventions[index], argNames[index]));
    }
    llvm::StringMap<StringRef> argNameToDetail;
    for (auto [index, value] : llvm::enumerate(argNames)) {
      if (index < argTypeNames.size())
        argNameToDetail[value] = argTypeNames[index];
    }

    // Grab the types of the parameters to the function.
    llvm::StringMap<std::string> paramToDetail;
    for (auto [index, value] : llvm::enumerate(funcOp.getInputParams())) {
      paramToDetail[value.getName()] =
          generateTypeString(value.getType(), selfType,
                             /*convention=*/std::nullopt, value.getName());
    }

    // Grab the result type, if it's non-none.
    std::optional<std::string> resultTypeName;
    if (!resultType.isa<LIT::NoneType>()) {
      resultTypeName =
          generateTypeString(resultType, selfType, resultConvention);
    }

    generateFunctionSignature(name, argTypeNames, resultTypeName);

    if (std::optional<DocString> docStr = getDocString(decl)) {
      os << docStr->getSummary() << "\n\n";
      processFunctionDocDescription(docStr->getDescription(), paramToDetail,
                                    argNameToDetail, resultTypeName);
      os << "\n";
    }

    // Recursively generate documentation for the module's children.
    generateLitMarkdownForChildren(decl);
  }

  /// Generate markdown for the signature of a function, given its components.
  void
  generateFunctionSignature(StringRef name, ArrayRef<std::string> argTypeNames,
                            const std::optional<std::string> &resultTypeName) {
    // Strip off the mangled suffix from the base function name.
    os << "> `` " << name.split('(').first;

    os << "(";
    interleaveComma(argTypeNames, os);
    os << ")";

    if (resultTypeName)
      os << " -> " << *resultTypeName;

    os << " ``\n\n";
  }

  void processFunctionDocDescription(
      ArrayRef<StringRef> description,
      const llvm::StringMap<std::string> &paramToDetail,
      const llvm::StringMap<StringRef> &argNameToDetail,
      const std::optional<std::string> &returnType) {
    // Process the lines of the description, looking for markers.
    for (size_t line = 0, lineE = description.size(); line < lineE; ++line) {
      if (description[line] == "Args:") {
        auto processArg = [&](StringRef argName) {
          auto it = argNameToDetail.find(argName);
          if (it != argNameToDetail.end())
            os << "`` " << it->second << " ``";
        };
        generateMarkdownListSection("Args", description, line, lineE,
                                    processArg);
        continue;
      }
      if (description[line] == "Parameters:") {
        generateParameterTable(paramToDetail, description, line, lineE);
        continue;
      }
      if (description[line] == "Returns:") {
        if (returnType) {
          generateSingleEntryMarkdownListSection("Returns", description, line,
                                                 lineE);
        }
        continue;
      }
      if (description[line] == "Constraints:") {
        generateParagraphMarkdownSection("Constraints", description, line,
                                         lineE);
        continue;
      }

      os << description[line] << "\n";
    }
  }

  //===----------------------------------------------------------------------===//
  // Struct Markdown Generation

  void generateLitMarkdownDocFor(ASTDecl &decl, StructDeclOp structOp) {
    generateMarkdownHeader(structOp.getName());

    if (std::optional<DocString> docStr = getDocString(decl)) {
      os << docStr->getSummary() << "\n\n";

      // Grab the types of the parameters to the struct.
      llvm::StringMap<std::string> paramToDetail;
      for (auto [index, value] : llvm::enumerate(structOp.getInputParams())) {
        paramToDetail[value.getName()] =
            generateTypeString(value.getType(), /*selfType=*/std::nullopt,
                               /*convention=*/std::nullopt, value.getName());
      }

      processStructDocDescription(docStr->getDescription(), paramToDetail);
    }

    // Recursively generate documentation for the module's children.
    generateLitMarkdownForChildren(decl);
  }

  void processStructDocDescription(
      ArrayRef<StringRef> description,
      const llvm::StringMap<std::string> &paramToDetail) {
    // Process the lines of the description, looking for markers.
    for (size_t line = 0, lineE = description.size(); line < lineE; ++line) {
      if (description[line] == "Parameters:") {
        generateParameterTable(paramToDetail, description, line, lineE);
        continue;
      }

      os << description[line] << "\n";
    }
  }

  //===----------------------------------------------------------------------===//
  // Module Markdown Generation

  void generateLitMarkdownDocFor(ASTDecl &decl, FileModuleOp moduleOp) {
    StringRef name = moduleOp.getName();
    name.consume_front("$");
    generateMarkdownHeader("Module: " + name);

    // If the module has a doc string, emit it.
    if (std::optional<DocString> docStr = getDocString(decl)) {
      os << docStr->getSummary() << "\n\n";
      for (StringRef descLine : docStr->getDescription())
        os << descLine << "\n";
    }

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

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

namespace {
class DocStringValidator {
public:
  DocStringValidator(SharedState &sharedState) : sharedState(sharedState) {}

  void validate(ASTDecl &decl) {
    std::optional<DocString> docStr = getDocString(decl);
    if (docStr && !decl.hasReferenceError) {
      TypeSwitch<ASTDecl &>(decl).Case<FuncOp, StructDeclOp>(
          [&](auto op) { validateDecl(decl, op, *docStr); });
    }
  }

private:
  //===----------------------------------------------------------------------===//
  // Utils

  /// Process a document section of the given form:
  ///
  /// Header:
  ///   Element1: ...
  ///   Element2: ...
  ///     ...
  ///   ElementN: ...
  ///
  void process2ColumnDocSection(
      ArrayRef<StringRef> &lines,
      function_ref<void(StringRef, SMLoc)> processEntryName) {
    size_t sectionIndent = getIndentationLevel(lines[0]);
    lines = lines.drop_front();
    while (!lines.empty()) {
      size_t lineIndent = getIndentationLevel(lines[0]);
      if (lineIndent <= sectionIndent)
        break;
      StringRef entryName = lines[0].split(':').first.trim();

      // Skip additional description lines that have a larger indentation.
      StringRef lastDocLine;
      do {
        lastDocLine = lines[0];
        lines = lines.drop_front();
      } while (!lines.empty() && getIndentationLevel(lines[0]) > lineIndent);
      processEntryName(entryName, SMLoc::getFromPointer(lastDocLine.end()));
    }
  }

  //===----------------------------------------------------------------------===//
  // Arguments and Parameters

  /// Process a parameter or argument section.
  void processParamOrArgs(SMLoc loc, StringRef tag,
                          llvm::MapVector<StringRef, SMLoc> &elements,
                          ArrayRef<StringRef> &lines) {
    StringRef sectionLine = lines[0];
    bool emittedUnexpectedOrderWarning = false;
    ptrdiff_t nextEltIndex = 0;
    SmallVector<SMLoc> elementDocEndLocs(elements.size());
    process2ColumnDocSection(
        lines, [&](StringRef paramName, SMLoc docEndLoc) {
          SMLoc paramLoc = SMLoc::getFromPointer(paramName.data());
          size_t currentEltIndex = nextEltIndex++;

          auto it = elements.find(paramName);
          if (it == elements.end()) {
            sharedState.emitWarning(paramLoc)
                << "unknown " << tag << " '" << paramName << "' in doc string";
            return;
          }

          // If we have already seen this element, emit a warning.
          if (std::exchange(it->second, paramLoc).isValid()) {
            sharedState.emitWarning(paramLoc) << "duplicate " << tag << " '"
                                              << paramName << "' in doc string";
            return;
          }

          // Ensure the elements are in the same order as the decl.
          if (!emittedUnexpectedOrderWarning) {
            size_t expectedEltIndex = it - elements.begin();
            if (currentEltIndex != expectedEltIndex) {
              sharedState.emitWarning(paramLoc)
                  << "'" << paramName << "' is defined at index "
                  << expectedEltIndex
                  << ", but specified in doc string at index "
                  << currentEltIndex;
              emittedUnexpectedOrderWarning = true;
            }
          }

          // Record the location of the end of the doc string for this element.
          elementDocEndLocs[it - elements.begin()] = docEndLoc;
        });

    // Emit warnings for any elements that were not documented.
    StringRef indentStr =
        sectionLine.take_front(loc.getPointer() - sectionLine.data());
    SMLoc sectionEndLoc = SMLoc::getFromPointer(sectionLine.end());
    for (auto [i, it] : llvm::enumerate(elements)) {
      auto &[element, seenLoc] = it;
      if (seenLoc.isValid())
        continue;
      LitDiagnostic diag = sharedState.emitWarning(loc)
                           << tag << " '" << element << "' is not documented";

      // Attach a fixit to add the element to the doc string.
      SMLoc prevEndLoc = (i == 0) ? sectionEndLoc : elementDocEndLocs[i - 1];
      diag.addFixIt(
          LitFixIt(LitSourceRange::getByteLevel(prevEndLoc, prevEndLoc),
                   "\n" + indentStr + std::string(4, ' ') + element + ":"));
      elementDocEndLocs[i] = prevEndLoc;
    }
  }
  void processArguments(SMLoc loc, llvm::MapVector<StringRef, SMLoc> &elements,
                        ArrayRef<StringRef> &lines) {
    processParamOrArgs(loc, "argument", elements, lines);
  }
  void processParameters(SMLoc loc, llvm::MapVector<StringRef, SMLoc> &elements,
                         ArrayRef<StringRef> &lines) {
    processParamOrArgs(loc, "parameter", elements, lines);
  }

  /// Process the sections within the given doc string description.
  void processDocSections(ArrayRef<StringRef> &lines,
                          DenseMap<StringRef, SMLoc> &sections,
                          function_ref<void(StringRef, SMLoc)> processSection) {
    for (; !lines.empty(); lines = lines.drop_front()) {
      // Sections end with `:`.
      StringRef section = lines[0];
      if (!section.consume_back(":"))
        continue;

      // Check if this is a known section.
      auto sectionIt = sections.find(section);
      if (sectionIt == sections.end()) {
        // Check to see if this is a known section that is just overindented.
        section = section.ltrim();
        sectionIt = sections.find(section);
        if (sectionIt == sections.end())
          continue;
        sharedState.emitWarning(SMLoc::getFromPointer(section.data()))
            << "section tag '" << section << "' is overindented";
      }
      SMLoc lineLoc = SMLoc::getFromPointer(section.data());
      SMLoc &sectionLoc = sectionIt->second;

      // If we have already seen this section, emit a warning.
      if (sectionLoc.isValid()) {
        auto diag = sharedState.emitWarning(
            lineLoc, "duplicate '" + section + "' section found in doc string");
        diag.attachNote(sectionLoc) << "see previous definition here";
        continue;
      }
      sectionLoc = lineLoc;

      // Process the section.
      processSection(section, lineLoc);
      if (lines.empty())
        break;
    }
  }

  //===----------------------------------------------------------------------===//
  // Functions

  /// Generate markdown documentation for the given function.
  void validateDecl(ASTDecl &decl, FuncOp funcOp, DocString &docStr) {
    SignatureType signature = funcOp.getSignature();
    auto argNames = funcOp.getValueParamNames();
    bool hasResultType = !funcOp.getResultType().isa<LIT::NoneType>();
    if (!hasResultType && signature.hasMemoryOnlyResult()) {
      argNames = argNames.drop_front();
      hasResultType = true;
    }

    // If this is a method, drop the self argument. We don't expect this to be
    // explicitly documented.
    if (isa<StructDeclOp>(funcOp->getParentOp()) && !funcOp.getIsStatic())
      argNames = argNames.drop_front();

    // Grab the types of the arguments to the function.
    llvm::MapVector<StringRef, SMLoc> seenArguments;
    for (StringAttr argName : argNames)
      seenArguments.insert({argName, SMLoc()});

    // Grab the parameters to the function.
    llvm::MapVector<StringRef, SMLoc> seenParameters;
    for (auto [index, value] : llvm::enumerate(funcOp.getInputParams()))
      seenParameters.insert({value.getName(), SMLoc()});

    // Process the sections of the doc string.
    DenseMap<StringRef, SMLoc> sections = {
        {"Args", SMLoc()},
        {"Parameters", SMLoc()},
        {"Returns", SMLoc()},
    };
    ArrayRef<StringRef> description = docStr.getDescription();
    auto processFn = [&](StringRef section, SMLoc loc) mutable {
      if (section == "Args") {
        processArguments(loc, seenArguments, description);
      } else if (section == "Parameters") {
        processParameters(loc, seenParameters, description);
      } else if (section == "Returns") {
        if (!hasResultType) {
          sharedState.emitWarning(loc, "unexpected 'Returns' in doc string for "
                                       "function with no results");
        }
      }
    };
    processDocSections(description, sections, processFn);
  }

  //===----------------------------------------------------------------------===//
  // Structs

  void validateDecl(ASTDecl &decl, StructDeclOp structOp, DocString &docStr) {
    // Grab the parameters to the struct.
    llvm::MapVector<StringRef, SMLoc> seenParameters;
    for (auto [index, value] : llvm::enumerate(structOp.getInputParams()))
      seenParameters.insert({value.getName(), SMLoc()});

    // Process the sections of the doc string.
    DenseMap<StringRef, SMLoc> sections = {
        {"Parameters", SMLoc()},
    };
    ArrayRef<StringRef> description = docStr.getDescription();
    auto processFn = [&](StringRef section, SMLoc loc) mutable {
      if (section == "Parameters")
        processParameters(loc, seenParameters, description);
    };
    processDocSections(description, sections, processFn);
  }

  /// Reference to the main shared state.
  SharedState &sharedState;
};
} // namespace

void M::KGEN::LIT::validateDocString(SharedState &sharedState, ASTDecl &decl) {
  DocStringValidator validator(sharedState);
  validator.validate(decl);
}
