//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/DocString.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Return the indentation level of the first line of the string.
static size_t getIndentationLevel(StringRef str) {
  return str.size() - str.ltrim().size();
}

/// Return if one of the given decorators is the doc_private decorator.
static bool hasDocPrivateDecorator(ArrayRef<TypedAttr> decorators) {
  return hasDecorator(decorators,
                      "stdlib::builtin::_documentation::doc_private");
}

/// Given a function name such as "__init__(module::Struct=&)", returns whether
/// it is a "special function," also known as a "dunder method"
/// (double-underscore method).
static bool isPublicSpecialFunction(StringRef name) {
  // If this is a normal function, treat it as a dunder if it's a __.
  if (SpecialFunctionInfo::getKind(name) == SpecialFunctionKind::kNormal)
    return name.starts_with("__");

  // Otherwise, for non-normal special functions, ignore mlir methods.
  return !name.starts_with("__mlir");
}

/// Return if a decl should be hidden given its name.
bool LIT::shouldHideDeclInDocGen(ASTDecl &decl, StringRef name) {
  // Hide private names (non special names starting with _).
  if (name.starts_with("_") && !isPublicSpecialFunction(name))
    return true;

  // Otherwise, check to see if this was marked explicitly to be hidden.
  return TypeSwitch<ASTDecl &, bool>(decl)
      .Case<FuncOp, GlobalVarDeclOp, StructDeclOp>(
          [&](auto op) { return hasDocPrivateDecorator(op.getDecorators()); })
      .Default(false);
}

/// A struct requires a doc string if it's defined at the top level of a module,
/// unless its name begins with an underscore.
static bool requiresDocString(ASTDeclInterface op) {
  return !op.getDeclName().strref().starts_with("_") &&
         isa<FileModuleOp>(op->getParentOp());
}

/// If a function matches all of the following conditions, it requires a doc
/// string:
/// 1. It's a "public" function, meaning its name does not start with an
///    underscore, unless it's a special function such as `__init__`.
/// 2. It's defined at the top level of a module, or as a (not-synthesized)
///    method on a struct that itself requires a doc string.
static bool requiresDocString(LIT::FuncOp op) {
  StringRef name = *op.getSourceName();
  if (name.starts_with("_") && !isPublicSpecialFunction(name))
    return false;

  // Don't require doc strings for explicitly annotated methods.
  if (hasDocPrivateDecorator(op.getDecorators()))
    return false;

  Operation *parent = op->getParentOp();
  if (isa<FileModuleOp>(parent))
    return true;

  if (auto parentStruct = dyn_cast<StructDeclOp>(parent)) {
    // We rely on an assumption here that only synthesized methods have the same
    // location as their parent struct.
    return requiresDocString(parentStruct) &&
           op.getLoc() != parentStruct.getLoc();
  }
  if (auto parentTrait = dyn_cast<TraitDeclOp>(parent)) {
    // We rely on an assumption here that only synthesized methods have the same
    // location as their parent struct.
    return requiresDocString(parentTrait) &&
           op.getLoc() != parentTrait.getLoc();
  }
  return false;
}

/// If a struct field matches all of the following conditions, it requires a doc
/// string:
/// 1. It's a "public" field, meaning its name does not start with an
///    underscore.
/// 2. Its parent struct requires a doc string.
static bool requiresDocString(StructFieldOp op) {
  return !op.getName().starts_with("_") &&
         requiresDocString(cast<StructDeclOp>(op->getParentOp()));
}

/// Return if the operation is nested in a private module or package.
static bool isOpInPrivateModule(Operation *declOp) {
  if (!declOp)
    return false;
  for (Operation *op = declOp->getParentOp(); op; op = op->getParentOp()) {
    if (auto fileOp = dyn_cast<FileModuleOp>(op)) {
      StringRef name = fileOp.getSymName();
      if (name.starts_with("_") && name != "__init__")
        return true;
    } else if (auto packageOp = dyn_cast<PackageOp>(op)) {
      if (packageOp.getSymName().starts_with("_"))
        return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// DocString
//===----------------------------------------------------------------------===//

DocString::DocString(DocStringAttr rawDocStringAttr)
    : loc(rawDocStringAttr.getLocation()) {
  // This function processes a doc-string, following a similar structure as
  // defined by PEP 257 for how multi-line doc strings should be formatted.
  // https://peps.python.org/pep-0257/#multi-line-docstrings
  StringRef rawDocString = rawDocStringAttr.getString();

  // Split the doc string into lines.
  SmallVector<StringRef> lineStorage;
  rawDocString.split(lineStorage, '\n');
  MutableArrayRef<StringRef> lines(lineStorage);
  if (lines.empty())
    return;

  // Determine the minimum indentation (first line doesn't count).
  indent = std::numeric_limits<size_t>::max();
  for (StringRef &line : lines.drop_front()) {
    // Trim out any carriage returns.
    line = line.trim("\r");

    if (line.empty())
      continue;
    indent = std::min(indent, getIndentationLevel(line));
    if (indent == 0)
      break;
  }

  // Remove the necessary indentation from all but the first line, which has all
  // leading whitespace removed.
  lines[0] = lines[0].ltrim().rtrim("\r");
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

std::string DocString::formatDescription(ArrayRef<StringRef> descriptionLines) {
  std::string result;
  llvm::raw_string_ostream os(result);

  // If the iterator is currently in a code block, this is the indent of the
  // code block.
  std::optional<unsigned> codeBlockIndent;
  for (const StringRef &line : descriptionLines) {
    StringRef strippedLine = line.trim();

    // Functor that determines if a line should be emitted.
    auto shouldEmitLine = [&] {
      // Check if we're in a code block currently.
      if (codeBlockIndent) {
        // Check for the end of the code block.
        if (strippedLine == "```" &&
            (line.size() - strippedLine.size()) == *codeBlockIndent) {
          codeBlockIndent.reset();
          return true;
        }

        // If the line is a REPL magic, strip it out.
        if (line.starts_with("%"))
          return false;
        // Otherwise, just emit the line.
        return true;
      }

      // Check for a new code block.
      if (strippedLine == "```mojo")
        codeBlockIndent = line.size() - strippedLine.size();
      return true;
    };
    if (shouldEmitLine()) {
      os << line;
      if (&line != &descriptionLines.back())
        os << "\n";
    }
  }
  return result;
}

SmallVector<DocString::CodeBlock> DocString::getCodeBlocks() const {
  SmallVector<CodeBlock> codeBlocks;

  // Process the description looking for code blocks.
  std::optional<CodeBlock> curCodeBlock;
  for (unsigned i = 0, e = descriptionLines.size(); i < e; ++i) {
    StringRef line = descriptionLines[i];
    StringRef strippedLine = line.trim();

    // Check if we're in a code block currently.
    if (curCodeBlock) {
      // If this isn't the end, add the line to the current block.
      if (strippedLine != "```" ||
          (line.size() - strippedLine.size()) != curCodeBlock->indentLevel)
        continue;
      // Just ignore empty code blocks.
      if (curCodeBlock->lineRange.first != i) {
        curCodeBlock->lineRange.second = i - 1;
        codeBlocks.emplace_back(std::move(*curCodeBlock));
      }
      curCodeBlock.reset();
      continue;
    }

    // Check for a new code block.
    if (strippedLine == "```mojo") {
      curCodeBlock.emplace(
          CodeBlock(*this, line.size() - strippedLine.size(), i + 1));
    }
  }
  return codeBlocks;
}

//===----------------------------------------------------------------------===//
// DocString::CodeBlock
//===----------------------------------------------------------------------===//

StringRef DocString::CodeBlock::getRawCode() const {
  const char *startPos =
      docString->descriptionLines[lineRange.first].data() - docString->indent;
  const char *lastPos = docString->descriptionLines[lineRange.second].end();
  return StringRef(startPos, lastPos - startPos);
}

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

/// Return the names of the arguments to the given function.
static SmallVector<StringAttr> getFunctionArgumentNames(LIT::FuncOp funcOp) {
  // In general, each function argument must be documented, but exceptions are
  // pruned from the list below.
  LITSignatureType sig = funcOp.getSignature();

  // The compiler can insert an implicit `__result__` argument, which stores
  // memory-only results, at the end of an argument list.  Because these
  // arguments are hidden artifacts of the compiler, they don't need to be
  // documented.
  size_t end =
      sig.getNumArguments() - sig.hasMemoryOnlyResult() - sig.isThrows();
  // Methods take `self` as an explicit first argument, for which
  // documentation isn't required.
  bool hasSelf = isa<StructDeclOp, TraitDeclOp>(funcOp->getParentOp()) &&
                 !funcOp.getIsStatic();

  SmallVector<StringAttr> argNames;
  for (size_t idx = hasSelf; idx < end; ++idx)
    argNames.emplace_back(sig.getArgName(idx));

  return argNames;
}

/// Return the names of the parameters to the given function.
static SmallVector<StringAttr> getFunctionParameterNames(LIT::FuncOp funcOp) {
  SmallVector<StringAttr> result;
  for (PogMetadataAttr pogAttr :
       funcOp.getSignature().getParamListAttrs().getPogs())
    if (pogAttr.getPassingKind() != PassingKind::Implicit)
      result.emplace_back(pogAttr.getName());
  return result;
}

/// Return if the given function is expecting results.
static bool doesFunctionHaveResults(LIT::FuncOp funcOp) {
  return !ASTType(funcOp.getUserResultType()).isNoneType();
}

namespace {
/// Used to specify the level of validation to perform for doc strings. The idea
/// is that some doc strings, such as ones added to non-public functions, act as
/// code comments, and do not require the same degree of strict validation as
/// doc strings for publicly exported symbols.
enum class ValidationKind {
  /// Perform basic checks, such as that when arguments are documented, they use
  /// the correct names.
  Normal,
  /// Perform strict checks: doc strings are required, and those doc strings
  /// must include relevant sections, such as a "returns" section for functions
  /// that have results.
  Strict,
};

/// Return whether this character is valid as the first character in a doc
/// string summary, section body, or argument description.
static bool isValidFirstCharacter(char c) { return !llvm::isLower(c); }

class DocStringValidator {
public:
  DocStringValidator(SharedState &sharedState, ASTDecl &decl)
      : sharedState(sharedState),
        diagnoseMissingDocStrings(
            sharedState.shouldDiagnoseMissingDocStrings()),
        errorOnInvalidDocStrings(sharedState.shouldErrorOnInvalidDocStrings()),
        docStr(decl.getDocString()
                   ? std::optional<DocString>(decl.getDocString())
                   : std::nullopt) {
    // If the doc string isn't valid, there's nothing to do.
    if (!docStr || !docStr->getLoc())
      return;
    rawDocStr = decl.getDocString().getString();

    // Otherwise, try to resolve a file location for the doc string.
    FileLineColLoc docLocAttr = docStr->getLoc();
    llvm::SourceMgr &sourceMgr = sharedState.getSourceMgr();

    // Check for an already opened file.
    int docBufferId = 0;
    for (int i = 0, e = sourceMgr.getNumBuffers(); i < e; ++i) {
      int bufferId = sourceMgr.getMainFileID() + i;
      auto *buffer = sourceMgr.getMemoryBuffer(bufferId);
      if (buffer->getBufferIdentifier() == docLocAttr.getFilename()) {
        docBufferId = bufferId;
        break;
      }
    }

    // Otherwise, try to pull in the file.
    if (!docBufferId) {
      std::string unused;
      docBufferId = sourceMgr.AddIncludeFile(docLocAttr.getFilename().str(),
                                             SMLoc(), unused);
      if (!docBufferId)
        return;
    }
    docStartLoc = sourceMgr.FindLocForLineAndColumn(
        docBufferId, docLocAttr.getLine(), docLocAttr.getColumn());
  }

  void validate(ASTDecl &decl) {
    if (decl.isErroneous())
      return;
    TypeSwitch<ASTDecl &>(decl)
        .Case<LIT::FuncOp, StructDeclOp, StructFieldOp,
              TraitDeclOp>([&](auto op) {
          ValidationKind validation = requiresDocString(op)
                                          ? ValidationKind::Strict
                                          : ValidationKind::Normal;
          if (!docStr) {
            if (validation == ValidationKind::Strict &&
                diagnoseMissingDocStrings && !isOpInPrivateModule(op))
              emitDiag(op.getLoc(), "public symbol ")
                  << op.getDeclName() << " is missing a doc string";
            return;
          }

          if (validation == ValidationKind::Strict) {
            // Check that the summary line is a complete sentence: it begins
            // with a capital letter (or a punctuator such as '`'), and ends
            // with a period.
            StringRef summary = docStr->getSummary();
            if (!summary.empty()) {
              if (!isValidFirstCharacter(summary.front()))
                emitDiag(docStr->getLoc(),
                         "doc string summary should begin with "
                         "a capital letter or non-alpha "
                         "character, but this begins with '")
                    << summary.front() << "'";

              if (!summary.ends_with("."))
                emitDiag(
                    docStr->getLoc(),
                    "doc string summary should end with a period '.', but this "
                    "ends with '")
                    << summary.back() << "'";
            }
          }

          validateDecl(decl, op, validation);
        });
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
  /// The \p processEntry callback is invoked for each entry. It is passed two
  /// string references: the entry name ("Element1" in the example above), and
  /// its description (the epllipsis "..." in the example above).
  void process2ColumnDocSection(
      ArrayRef<StringRef> &lines,
      function_ref<void(StringRef, StringRef)> processEntry) {
    size_t sectionIndent = getIndentationLevel(lines[0]);
    lines = lines.drop_front();
    while (!lines.empty()) {
      size_t lineIndent = getIndentationLevel(lines[0]);
      if (lineIndent <= sectionIndent)
        break;

      StringRef entryName = lines[0].split(':').first.trim();
      const char *body = lines[0].split(':').second.ltrim().data();

      // Skip additional description lines that have a larger indentation.
      StringRef lastDocLine;
      do {
        lastDocLine = lines[0];
        lines = lines.drop_front();
      } while (!lines.empty() && getIndentationLevel(lines[0]) > lineIndent);
      processEntry(entryName, StringRef(body, lastDocLine.end() - body));
    }
  }

  /// Given pointers to the \p first and \p last characters in a portion of a
  /// doc string, emits warnings (prefixed with the given \p name) if:
  /// 1. The first character is not either a capital letter or a backtick '`'.
  /// 2. The last character is not a period.
  void validateStyle(StringRef name, const char *first, const char *last) {
    if (!isValidFirstCharacter(*first))
      emitDiag(first, name) << " should begin with a capital letter or "
                               "non-alpha character, but this begins with '"
                            << *first << "'";

    if (*last != '.')
      emitDiag(last, name)
          << " should end with a period '.', but this ends with '" << *last
          << "'";
  }

  //===----------------------------------------------------------------------===//
  // Arguments and Parameters

  /// Process a parameter or argument section.
  void processParamOrArgs(const char *loc, StringRef tag,
                          llvm::MapVector<StringRef, const char *> &elements,
                          ArrayRef<StringRef> &lines,
                          ValidationKind validation) {
    StringRef sectionLine = lines[0];
    bool emittedUnexpectedOrderWarning = false;
    ptrdiff_t nextEltIndex = 0;
    SmallVector<const char *> elementDocEndLocs(elements.size());
    process2ColumnDocSection(
        lines, [&](StringRef paramName, StringRef paramBody) {
          const char *paramLoc = paramName.data();
          size_t currentEltIndex = nextEltIndex++;

          auto it = elements.find(paramName);
          if (it == elements.end()) {
            emitDiag(paramLoc)
                << "unknown " << tag << " '" << paramName << "' in doc string";
            return;
          }

          // If we have already seen this element, emit a warning.
          if (std::exchange(it->second, paramLoc)) {
            emitDiag(paramLoc) << "duplicate " << tag << " '" << paramName
                               << "' in doc string";
            return;
          }

          // Ensure the elements are in the same order as the decl.
          if (!emittedUnexpectedOrderWarning) {
            size_t expectedEltIndex = it - elements.begin();
            if (currentEltIndex != expectedEltIndex) {
              emitDiag(paramLoc) << "'" << paramName << "' is defined at index "
                                 << expectedEltIndex
                                 << ", but specified in doc string at index "
                                 << currentEltIndex;
              emittedUnexpectedOrderWarning = true;
            }
          }

          // Diagnose empty element descriptions.
          const char *docEndLoc = paramBody.end();
          if (paramBody.empty()) {
            emitDiag(paramLoc)
                << "'" << paramName << "' does not have a description";
            docEndLoc = paramName.end();
          }

          // Diagnose descriptions with poor style.
          if (validation == ValidationKind::Strict && !paramBody.empty())
            validateStyle((Twine("'") + paramName + "' description").str(),
                          paramBody.begin(), paramBody.end() - 1);

          // Record the location of the end of the doc string for this element.
          elementDocEndLocs[it - elements.begin()] = docEndLoc;
        });

    // Emit warnings for any elements that were not documented.
    StringRef indentStr = sectionLine.take_front(loc - sectionLine.data());
    const char *sectionEndLoc = sectionLine.end();
    for (auto [i, it] : llvm::enumerate(elements)) {
      auto &[element, seenLoc] = it;
      if (seenLoc)
        continue;
      InflightDiag diag = emitDiag(loc)
                          << tag << " '" << element << "' is not documented";

      // Attach a fixit to add the element to the doc string.
      const char *prevEndLoc =
          (i == 0) ? sectionEndLoc : elementDocEndLocs[i - 1];
      SMLoc prevEndSMLoc = translateLoc(prevEndLoc);
      if (prevEndSMLoc.isValid()) {
        diag.addFixIt(
            FixIt(SourceRange::getByteLevel(prevEndSMLoc, prevEndSMLoc),
                  "\n" + indentStr + std::string(4, ' ') + element + ":"));
      }
      elementDocEndLocs[i] = prevEndLoc;
    }
  }
  void processArguments(const char *loc,
                        llvm::MapVector<StringRef, const char *> &elements,
                        ArrayRef<StringRef> &lines, ValidationKind validation) {
    processParamOrArgs(loc, "argument", elements, lines, validation);
  }
  void processParameters(const char *loc,
                         llvm::MapVector<StringRef, const char *> &elements,
                         ArrayRef<StringRef> &lines,
                         ValidationKind validation) {
    processParamOrArgs(loc, "parameter", elements, lines, validation);
  }

  /// Process the sections within the given doc string description.
  void processDocSections(
      ArrayRef<StringRef> &lines, DenseMap<StringRef, const char *> &sections,
      function_ref<void(StringRef, const char *)> processSection) {
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
        emitDiag(section.data())
            << "section tag '" << section << "' is overindented";
      }
      const char *lineLoc = section.data();
      const char *&sectionLoc = sectionIt->second;

      // If we have already seen this section, emit a warning.
      if (sectionLoc) {
        auto diag = emitDiag(lineLoc, "duplicate '" + section +
                                          "' section found in doc string");
        diag.attachNote(translateLoc(sectionLoc))
            << "see previous definition here";
        continue;
      }
      sectionLoc = lineLoc;

      // Check that text follows the section header. For example, this should
      // diagnose a doc string that ends with `Returns:"""`.
      if (lines.size() == 1) {
        emitDiag(sectionLoc, "'" + section + "' section is empty");
        break;
      }

      // Process the section.
      processSection(section, lineLoc);
      if (lines.empty())
        break;
    }
  }

  //===--------------------------------------------------------------------===//
  // Functions

  /// Validate documentation for the given function.
  void validateDecl(ASTDecl &decl, LIT::FuncOp funcOp,
                    ValidationKind validation) {
    // Grab the types of the arguments to the function.
    llvm::MapVector<StringRef, const char *> seenArguments;
    for (StringAttr argName : getFunctionArgumentNames(funcOp))
      seenArguments.insert({argName, nullptr});

    // Grab the parameters to the function.
    llvm::MapVector<StringRef, const char *> seenParameters;
    for (StringAttr paramName : getFunctionParameterNames(funcOp))
      seenParameters.insert({paramName, nullptr});

    // Process the sections of the doc string.
    DenseMap<StringRef, const char *> sections = {
        {DocString::kSectionConstraints, nullptr},
        {DocString::kSectionArgs, nullptr},
        {DocString::kSectionParameters, nullptr},
        {DocString::kSectionReturns, nullptr},
        {DocString::kSectionRaises, nullptr},
    };
    ArrayRef<StringRef> description = docStr->getDescription();
    bool hasResults = doesFunctionHaveResults(funcOp);
    bool canThrow = funcOp.getSignature().isThrows();

    auto processFn = [&](StringRef section, const char *loc) mutable {
      if (section == DocString::kSectionArgs)
        return processArguments(loc, seenArguments, description, validation);

      if (section == DocString::kSectionParameters)
        return processParameters(loc, seenParameters, description, validation);

      if (section == DocString::kSectionReturns && !hasResults)
        emitDiag(loc, "unexpected 'Returns' in doc string for "
                      "function with no results");

      if (section == DocString::kSectionRaises && !canThrow)
        emitDiag(loc, "unexpected 'Raises' in doc string for "
                      "function that does not throw");

      // Validate paragraph sections such as "Constraints:" and "Returns:".
      if (validation == ValidationKind::Strict) {
        StringRef firstLine = description[1].ltrim();
        StringRef lastLine = description.back().rtrim();
        validateStyle("section body", firstLine.begin(), lastLine.end() - 1);
      }
    };
    processDocSections(description, sections, processFn);

    if (validation == ValidationKind::Strict && diagnoseMissingDocStrings &&
        !isOpInPrivateModule(funcOp)) {
      if (!sections[DocString::kSectionParameters] && !seenParameters.empty())
        emitDiag(
            funcOp.getLoc(),
            "function takes parameters, but no 'Parameters' in doc string");
      if (!sections[DocString::kSectionArgs] && !seenArguments.empty())
        emitDiag(funcOp.getLoc(),
                 "function takes arguments, but no 'Args' in doc string");
      if (!sections[DocString::kSectionReturns] && hasResults)
        emitDiag(funcOp.getLoc(),
                 "function has results, but no 'Returns' in doc string");
    }
  }

  //===--------------------------------------------------------------------===//
  // Structs

  void validateDecl(ASTDecl &decl, StructDeclOp structOp,
                    ValidationKind validation) {
    // Grab the parameters to the struct.
    llvm::MapVector<StringRef, const char *> seenParameters;
    for (ParamDeclAttr decl : structOp.getParams())
      seenParameters.insert({demangleIfNeeded(decl).getName(), nullptr});

    // Process the sections of the doc string.
    DenseMap<StringRef, const char *> sections = {
        {DocString::kSectionParameters, nullptr},
    };
    ArrayRef<StringRef> description = docStr->getDescription();
    auto processFn = [&](StringRef section, const char *loc) mutable {
      if (section == DocString::kSectionParameters)
        processParameters(loc, seenParameters, description, validation);
    };
    processDocSections(description, sections, processFn);

    if (validation == ValidationKind::Strict && diagnoseMissingDocStrings &&
        !isOpInPrivateModule(structOp) &&
        !sections[DocString::kSectionParameters] && !seenParameters.empty())
      emitDiag(structOp.getLoc(),
               "struct takes parameters, but no 'Parameters' in doc string");
  }

  //===--------------------------------------------------------------------===//
  // Fields

  void validateDecl(ASTDecl &decl, StructFieldOp fieldOp,
                    ValidationKind validation) {
    // Nothing to do.
  }

  //===--------------------------------------------------------------------===//
  // Traits

  void validateDecl(ASTDecl &decl, TraitDeclOp traitDeclOp,
                    ValidationKind validation) {
    // TODO: issue #21850 add validation for trait docstring.
  }

  //===--------------------------------------------------------------------===//
  // Diagnostics

  /// Emit a diagnostic at the given doc string location.
  InflightDiag emitDiag(const char *loc, const Twine &msg = {}) {
    SMLoc smLoc = translateLoc(loc);
    return smLoc.isValid() ? emitDiag(smLoc, msg)
                           : emitDiag(docStr->getLoc(), msg);
  }
  template <typename T>
  InflightDiag emitDiag(T loc, const Twine &msg = {}) {
    if (errorOnInvalidDocStrings)
      return sharedState.emitError(loc, msg);
    return sharedState.emitWarning(loc, msg);
  }

  /// Translate a doc string location to a source location.
  SMLoc translateLoc(const char *loc) {
    if (!docStartLoc.isValid())
      return SMLoc();

    // Compute the location
    return SMLoc::getFromPointer(docStartLoc.getPointer() +
                                 (loc - rawDocStr.data()));
  }

  /// Reference to the main shared state.
  SharedState &sharedState;

  /// Flag indicating if we should diagnose missing doc strings.
  bool diagnoseMissingDocStrings;

  /// Flag indicating if we should error on invalid doc strings.
  bool errorOnInvalidDocStrings;

  /// The doc string currently being processed.
  std::optional<DocString> docStr;
  StringRef rawDocStr;

  /// The starting source location of the doc string.
  SMLoc docStartLoc;
};
} // namespace

void M::KGEN::LIT::validateDocString(SharedState &sharedState, ASTDecl &decl) {
  DocStringValidator validator(sharedState, decl);
  validator.validate(decl);
}

//===----------------------------------------------------------------------===//
// Generation
//===----------------------------------------------------------------------===//

namespace {
class DocStringGenerator {
public:
  DocStringGenerator(size_t indent) : indent(indent), os(rawOS) {}

  std::optional<std::string> generate(ASTDecl &decl) {
    if (decl.isErroneous())
      return std::nullopt;

    TypeSwitch<ASTDecl &>(decl)
        .Case<LIT::FuncOp, FileModuleOp, StructDeclOp, StructFieldOp,
              TraitDeclOp>([&](auto op) {
          StringRef summaryCodeBlock = "[summary].";
          os << summaryCodeBlock;

          // Indent and generate the rest of the decl.
          for (size_t i = 0; i < indent; i += 2)
            os.indent();
          generateDecl(decl, op);

          // If we added anything other than the summary, add a newline.
          if (rawOS.str().size() > summaryCodeBlock.size()) {
            os << "\n";
            os.indent(indent);
          }
        });

    // If we actually generated something, return it, otherwise bail.
    if (rawOS.str().empty())
      return std::nullopt;
    return std::move(result);
  }

private:
  //===----------------------------------------------------------------------===//
  // Arguments and Parameters

  /// Process a parameter or argument section.
  void processParamOrArgs(StringRef sectionName, ArrayRef<StringAttr> names) {
    if (names.empty())
      return;
    os << "\n\n" << sectionName << ":";
    for (StringRef name : names)
      os << "\n    " << name << ": [description].";
  }
  void processArguments(ArrayRef<StringAttr> argNames) {
    processParamOrArgs("Args", argNames);
  }
  void processParameters(ArrayRef<StringAttr> params) {
    processParamOrArgs("Parameters", params);
  }

  //===--------------------------------------------------------------------===//
  // Functions

  void generateDecl(ASTDecl &decl, LIT::FuncOp funcOp) {
    processParameters(getFunctionParameterNames(funcOp));
    processArguments(getFunctionArgumentNames(funcOp));
    if (doesFunctionHaveResults(funcOp))
      os << "\n\nReturns:\n    [description].";
    if (funcOp.isThrows())
      os << "\n\nRaises:\n    [description].";
  }

  //===--------------------------------------------------------------------===//
  // Modules

  void generateDecl(ASTDecl &decl, FileModuleOp moduleOp) {
    // Nothing to do.
  }

  //===--------------------------------------------------------------------===//
  // Structs

  void generateDecl(ASTDecl &decl, StructDeclOp structOp) {
    // Grab the parameters to the struct.
    SmallVector<StringAttr> paramNames;
    for (ParamDeclAttr decl : structOp.getParams())
      paramNames.push_back(demangleIfNeeded(decl).getName());
    processParameters(paramNames);
  }

  //===--------------------------------------------------------------------===//
  // Fields

  void generateDecl(ASTDecl &decl, StructFieldOp fieldOp) {
    // Nothing to do.
  }

  //===--------------------------------------------------------------------===//
  // Traits

  void generateDecl(ASTDecl &decl, TraitDeclOp traitDeclOp) {
    // TODO(#21850): Add generation for trait docstrings.
  }

  /// The desired indentation level for the generated doc string.
  size_t indent;

  /// The resultant template string.
  std::string result;

  /// The stream used to populate the template.
  llvm::raw_string_ostream rawOS{result};
  mlir::raw_indented_ostream os;
};
} // namespace

std::optional<std::string>
M::KGEN::LIT::generateDocStringTemplate(ASTDecl &decl, size_t indent) {
  DocStringGenerator generator(indent);
  return generator.generate(decl);
}
