//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "DocString.h"
#include "ASTDecl.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Regex.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Within a doc string, the "Constraints" section describes invariants that
/// must be true for the struct or function.
static constexpr const char *kConstraints = "Constraints";

/// Within a doc string, the "Parameters" section lists descriptions of each
/// parameter.
static constexpr const char *kParameters = "Parameters";

/// Within a doc string, the "Args" section lists descriptions of each function
/// argument.
static constexpr const char *kArgs = "Args";

/// Within a doc string, the "Returns" section describes the results of a
/// function.
static constexpr const char *kReturns = "Returns";

/// Return the indentation level of the first line of the string.
static size_t getIndentationLevel(StringRef str) {
  return str.size() - str.ltrim().size();
}

/// Extract a DocString from a given decl, or None if there is no doc string.
static std::optional<DocString> getDocString(ASTDecl &decl) {
  // FIXME: This isn't right, this should be using Lexer::getStringLiteralValue.
  StringRef docStr = decl.getDocString();
  if (!docStr.empty()) {
    if (docStr.size() >= 6 &&
        (docStr.starts_with("\"\"\"") || docStr.starts_with("'''")))
      docStr = docStr.drop_front(3).drop_back(3);
    else
      docStr = docStr.drop_front(1).drop_back(1);
  }

  if (docStr.empty())
    return std::nullopt;
  return DocString(docStr);
}

/// A struct requires a doc string if it's defined at the top level of a module,
/// unless its name begins with an underscore.
static bool requiresDocString(StructDeclOp op) {
  return !op.getName().starts_with("_") && isa<FileModuleOp>(op->getParentOp());
}

/// Given a function name such as "__init__($module::Struct=&)", returns whether
/// it is a "special function," also known as a "dunder method"
/// (double-underscore method).
static bool isSpecialFunction(StringRef name) {
  return SpecialFunctionInfo::getKind(name.split("(").first) !=
         SpecialFunctionKind::kNormal;
}

/// If a function matches all of the following conditions, it requires a doc
/// string:
/// 1. It's a "public" function, meaning its name does not start with an
///    underscore, unless it's a special function such as `__init__`.
/// 2. It's defined at the top level of a module, or as a method on a struct
///    that requires a doc string.
static bool requiresDocString(LIT::FuncOp op) {
  if (op.getName().starts_with("_") && !isSpecialFunction(op.getName()))
    return false;

  Operation *parent = op->getParentOp();
  StructDeclOp parentStruct = dyn_cast<StructDeclOp>(parent);
  return isa<FileModuleOp>(parent) ||
         (parentStruct && requiresDocString(parentStruct));
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

  loc = SMLoc::getFromPointer(lines.front().begin());

  // Determine the minimum indentation (first line doesn't count).
  size_t indent = std::numeric_limits<size_t>::max();
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

//===----------------------------------------------------------------------===//
// JSON Generation
//===----------------------------------------------------------------------===//

namespace {
class JSONGenerator {
public:
  JSONGenerator(raw_ostream &os) : os(os, /*IndentSize=*/2) {}

  void generate(ASTDecl &decl) {
    TypeSwitch<ASTDecl &>(decl)
        .Case<FileModuleOp, LIT::FuncOp, ParamDeclareOp, StructDeclOp,
              StructFieldOp>([&](auto op) { generateJSONFor(decl, op); });
  }

private:
  //===--------------------------------------------------------------------===//
  // Utils

  /// Return an ordering priority number for the given decl name. Lower numbers
  /// are ordered first.
  static unsigned getDeclNamePriority(StringRef name) {
    // If the name is a special function, use that as the priority.
    SpecialFunctionKind specialFnKind = SpecialFunctionInfo::getKind(name);
    if (specialFnKind != SpecialFunctionKind::kNormal)
      return static_cast<unsigned>(specialFnKind);

    // Otherwise, we can't discern any priorty from the name.
    return std::numeric_limits<unsigned>::max();
  }

  /// Given the names of two decls, returns if `lhs` should be ordered before
  /// `rhs`.
  static bool compareDeclNames(StringRef lhs, StringRef rhs) {
    // If the names are the same, we don't need to do anything.
    if (lhs == rhs)
      return false;

    // First compare the priority of the names.
    unsigned lhsPriority = getDeclNamePriority(lhs);
    unsigned rhsPriority = getDeclNamePriority(rhs);
    if (lhsPriority != rhsPriority)
      return lhsPriority < rhsPriority;

    // Then compare the names themselves.
    return lhs < rhs;
  }

  void generateJSONForChildren(ASTDecl &decl) {
    using ChildrenVecT =
        SmallVector<std::pair<StringAttr, TinyPtrVector<ASTDecl *>>>;
    ChildrenVecT aliases, functions, structs, fields;

    // Bucket the different types of children.
    const auto &declsInScope = decl.getDeclsInScope();
    for (auto &[name, decls] : declsInScope) {
      if (shouldHideName(name) || decls.empty())
        continue;

      if (isa<ParamDeclareOp>(**decls.begin()))
        aliases.emplace_back(name, decls);
      else if (isa<LIT::FuncOp>(**decls.begin()))
        functions.emplace_back(name, decls);
      else if (isa<StructDeclOp>(**decls.begin()))
        structs.emplace_back(name, decls);
      else if (isa<StructFieldOp>(**decls.begin()))
        fields.emplace_back(name, decls);
    }

    // Functor used to generically process a bucket of children.
    auto processChildren = [&](StringRef tag, ChildrenVecT &children,
                               auto &&processChildFn) {
      llvm::sort(children, [](auto &lhs, auto &rhs) {
        return compareDeclNames(lhs.first, rhs.first);
      });

      // Skip declarations that were imported from other scopes.
      // TODO: We should note that these are imported/aliases.
      auto filterChildren = [&](TinyPtrVector<ASTDecl *> &children) {
        return llvm::make_filter_range(children, [&](ASTDecl *child) {
          return child->getParentDecl() == &decl;
        });
      };

      os.attributeArray(tag, [&] {
        for (auto &[name, decls] : children) {
          auto filteredChildren = filterChildren(decls);
          if (!filteredChildren.empty())
            processChildFn(name, filteredChildren);
        }
      });
    };

    // Functor used to process all of the given children.
    auto processAllChildrenFn = [&](StringRef name, auto &&children) {
      for (auto &childDecl : children)
        generate(*childDecl);
    };

    // Process aliases.
    if (!aliases.empty())
      processChildren("aliases", aliases, processAllChildrenFn);

    // Process fields.
    if (!fields.empty())
      processChildren("fields", fields, processAllChildrenFn);

    // Process functions.
    if (!functions.empty()) {
      auto processFn = [&](StringRef name, auto &&children) {
        generateJSONForFunctions(name, children);
      };
      processChildren("functions", functions, processFn);
    }

    // Process structs.
    if (!structs.empty())
      processChildren("structs", structs, processAllChildrenFn);
  }

  /// Return if the given name should be hidden from the output.
  bool shouldHideName(StringRef name) {
    // Non-underscore names are never hidden.
    if (!name.startswith("_"))
      return false;

    // Keep special language names, which have leading and trailing underscores,
    // even though they start with `_`.
    return !(name.startswith("__") && name.endswith("__"));
  }

  /// Generate an array section from the given form:
  ///
  /// Header:
  ///   Element1: ...
  ///   Element2: ...
  ///     ...
  ///   ElementN: ...
  ///
  template <typename DetailMapT>
  void generateArraySection(StringRef header, ArrayRef<StringRef> lines,
                            size_t &line, size_t lineE, DetailMapT &&entryMap) {
    std::string fullArgDesc;
    llvm::raw_string_ostream fullArgDescOS(fullArgDesc);
    os.attributeArray(header, [&] {
      for (++line; line < lineE && !lines[line].empty();) {
        // Extract the argument name and description.
        auto [argName, argDesc] = lines[line].split(':');
        argName = argName.trim();
        argDesc = argDesc.trim();

        fullArgDesc.clear();
        fullArgDescOS << argDesc;

        // Merge in additional description lines that have a larger indentation.
        size_t indent = getIndentationLevel(lines[line]);
        while (++line < lineE && getIndentationLevel(lines[line]) > indent)
          fullArgDescOS << " " << lines[line].trim();

        // If it's a known entry, process it, otherwise skip it.
        if (auto it = entryMap.find(argName); it != entryMap.end()) {
          os.object([&] {
            os.attribute("name", it->first());
            os.attribute("type", it->second);
            os.attribute("description", fullArgDesc);
          });
        }
      }
    });
  }

  /// Generate a string attribute from the given paragraph form:
  ///
  /// Header:
  ///   Element1...
  void generateParagraphSection(StringRef header, ArrayRef<StringRef> lines,
                                size_t &line, size_t lineE) {
    std::string paragraph;
    llvm::raw_string_ostream paragraphOS(paragraph);

    // A doc string may end with "Header:". This is diagnosed by the validator,
    // but invalid doc strings may still be emitted as JSON.
    if (line >= lines.size()) {
      os.attribute(header, paragraphOS.str());
      return;
    }

    paragraphOS << lines[++line].trim();

    // Merge in additional description lines that have equal or larger
    // indentation.
    size_t indent = getIndentationLevel(lines[line]);
    while (++line < lineE && getIndentationLevel(lines[line]) >= indent)
      paragraphOS << " " << lines[line].trim();
    os.attribute(header, paragraphOS.str());
  }

  //===--------------------------------------------------------------------===//
  // Types

  /// Generate a documentation string for the given type, with an optional
  /// value convention, parent struct "Self" type.
  std::string generateTypeString(
      Type type, std::optional<ASTType> selfType = std::nullopt,
      std::optional<ValueInputConvention> convention = std::nullopt) {
    std::string typeName;
    llvm::raw_string_ostream os(typeName);
    ASTType astType(type);

    // Handle variadic types.
    if (isa<VariadicType>(type)) {
      astType = astType.getVariadicElementType();
      os << "*";
    }

    // Process the convention if present.
    StringRef typeSuffix;
    if (convention) {
      switch (*convention) {
      case ValueInputConvention::ByRef:
      case ValueInputConvention::InitSelf:
        astType = astType.getPointerElementType();
        typeSuffix = "&";
        break;
      case ValueInputConvention::ByRefResult:
        astType = astType.getPointerElementType();
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
      os << astType.getAsString(/*forDiag=*/true);

    // Append the type suffix.
    os << typeSuffix;
    return os.str();
  }

  //===--------------------------------------------------------------------===//
  // Alias Generation

  void generateJSONFor(ASTDecl &decl, ParamDeclareOp paramOp) {
    os.object([&] {
      os.attribute("kind", "alias");
      os.attribute("name", paramOp.getName().getValue());

      // Pretty print the value.
      std::string valueStr;
      llvm::raw_string_ostream valueOS(valueStr);
      PValue(paramOp.getValue()).printForDiag(valueOS);
      os.attribute("value", valueOS.str());

      // Emit the doc string if present.
      if (std::optional<DocString> docStr = getDocString(decl)) {
        os.attribute("summary", docStr->getSummary());
        os.attribute("description", llvm::join(docStr->getDescription(), "\n"));
      }
    });
  }

  //===--------------------------------------------------------------------===//
  // Field Generation

  void generateJSONFor(ASTDecl &decl, StructFieldOp fieldOp) {
    os.object([&] {
      os.attribute("kind", "field");
      os.attribute("name", fieldOp.getName());

      // Pretty print the value.
      std::string valueStr;
      llvm::raw_string_ostream typeOS(valueStr);
      ASTType(fieldOp.getType()).print(typeOS, /*forDiag=*/true);
      os.attribute("value", typeOS.str());

      // Emit the doc string if present.
      if (std::optional<DocString> docStr = getDocString(decl)) {
        os.attribute("summary", docStr->getSummary());
        os.attribute("description", llvm::join(docStr->getDescription(), "\n"));
      }
    });
  }

  //===--------------------------------------------------------------------===//
  // Function Generation

  /// Generate documentation for the given function.
  void generateJSONFor(ASTDecl &decl, LIT::FuncOp funcOp) {
    // Strip off the mangled suffix from the base function name.
    StringRef name =
        funcOp.getName().take_until([](char c) { return c == '('; });
    return generateJSONForFunctions(name, ArrayRef<ASTDecl *>(&decl));
  }
  template <typename DeclRangeT>
  void generateJSONForFunctions(StringRef name, const DeclRangeT &decls) {
    os.object([&] {
      os.attribute("kind", "function");
      os.attribute("name", name);
      os.attributeArray("overloads", [&] {
        for (auto *decl : decls)
          generateJSONForOverload(*decl, cast<LIT::FuncOp>(*decl), name);
      });
    });
  }

  /// Generate a sub-section for the overload described by the given function.
  void generateJSONForOverload(ASTDecl &decl, LIT::FuncOp funcOp,
                               StringRef name) {
    auto argTypes = funcOp.getArgumentTypes();
    auto argNames = funcOp.getValueParamNames();
    auto argConventions = funcOp.getSignature().getValueInputConventions();
    ASTType resultType = funcOp.getUserResultType();

    // Check for a by-ref result type, which gets modeled as the first argument
    // (as it needs to be passed through memory), and we don't want to include
    // it in the normal argument list.
    if (!argConventions.empty() &&
        argConventions.front() == ValueInputConvention::ByRefResult) {
      argTypes = argTypes.drop_front();
      argNames = argNames.drop_front();
      argConventions = argConventions.drop_front();
    }

    // If this is a method, grab the expected "Self" type.
    std::optional<ASTType> selfType;
    if (isa<StructDeclOp>(funcOp->getParentOp()))
      selfType = decl.getParentDecl()->getSelfType();

    // Grab the types of the arguments to the function.
    SmallVector<std::string> argTypeDetails;
    for (auto [index, argType] : llvm::enumerate(argTypes)) {
      argTypeDetails.push_back(
          generateTypeString(argType, selfType, argConventions[index]));
    }
    llvm::StringMap<StringRef> argNameToDetail;
    for (auto [index, value] : llvm::enumerate(argNames)) {
      if (index < argTypeDetails.size())
        argNameToDetail[value] = argTypeDetails[index];
    }

    // Grab the types of the parameters to the function.
    SmallVector<std::string> paramTypeDetails;
    llvm::StringMap<StringRef> paramToDetail;
    ArrayRef<ParamDeclAttr> params = funcOp.getInputParams();
    for (ParamDeclAttr param : params)
      paramTypeDetails.push_back(generateTypeString(param.getType(), selfType));
    for (auto [index, value] : llvm::enumerate(params))
      paramToDetail[value.getName()] = paramTypeDetails[index];

    // Grab the result type, if it's non-none.
    std::optional<std::string> resultTypeName;
    if (!resultType.isNoneType())
      resultTypeName = generateTypeString(resultType, selfType);

    os.object([&] {
      os.attribute("signature", generateFunctionSignature(
                                    name, argNames, argTypeDetails, params,
                                    paramTypeDetails, resultTypeName));

      // Emit the doc string if present.
      if (std::optional<DocString> docStr = getDocString(decl)) {
        os.attribute("summary", docStr->getSummary());
        processFunctionDocDescription(docStr->getDescription(), paramToDetail,
                                      argNameToDetail, resultTypeName);
      }
      if (funcOp.isThrows())
        os.attribute("raises", true);
      if (funcOp.isAsync())
        os.attribute("async", true);
    });
  }

  /// Generate a string for the signature of a function, given its components.
  std::string
  generateFunctionSignature(StringRef name, ArrayRef<StringAttr> argNames,
                            ArrayRef<std::string> argTypes,
                            ArrayRef<ParamDeclAttr> params,
                            ArrayRef<std::string> paramTypes,
                            const std::optional<std::string> &resultTypeName) {
    std::string signature;
    llvm::raw_string_ostream signatureOS(signature);

    // Functor used to emit a parameter or type to the signature.
    auto emitParamOrArg = [&](StringRef name, StringRef type) {
      // If the argument is variadic, we put the star before the name when
      // printing a signature.
      if (type.consume_front("*"))
        signatureOS << "*";
      signatureOS << name << ": " << type;
    };

    // Strip off the mangled suffix from the base function name.
    signatureOS << name.split('(').first;

    // Emit the parameters of the function.
    if (!params.empty()) {
      signatureOS << "[";
      interleaveComma(
          llvm::seq<int>(0, paramTypes.size()), signatureOS, [&](int index) {
            emitParamOrArg(params[index].getName(), paramTypes[index]);
          });
      signatureOS << "]";
    }

    // Emit the arguments of the function.
    signatureOS << "(";
    interleaveComma(
        llvm::seq<int>(0, argTypes.size()), signatureOS,
        [&](int index) { emitParamOrArg(argNames[index], argTypes[index]); });
    signatureOS << ")";

    // Emit the result type.
    if (resultTypeName)
      signatureOS << " -> " << *resultTypeName;
    return signatureOS.str();
  }

  void processFunctionDocDescription(
      ArrayRef<StringRef> description,
      const llvm::StringMap<StringRef> &paramToDetail,
      const llvm::StringMap<StringRef> &argNameToDetail,
      const std::optional<std::string> &returnType) {
    // Process the lines of the description, looking for markers.
    SmallVector<StringRef> pureDescriptionLines;
    for (size_t line = 0, lineE = description.size(); line < lineE; ++line) {
      if (description[line] == (Twine(kArgs) + ":").str()) {
        generateArraySection("args", description, line, lineE, argNameToDetail);
      } else if (description[line] == (Twine(kParameters) + ":").str()) {
        generateArraySection("parameters", description, line, lineE,
                             paramToDetail);
      } else if (description[line] == (Twine(kReturns) + ":").str()) {
        if (returnType)
          generateParagraphSection("returns", description, line, lineE);
      } else if (description[line] == (Twine(kConstraints) + ":").str()) {
        generateParagraphSection("constraints", description, line, lineE);
      } else {
        pureDescriptionLines.push_back(description[line]);
      }
    }
    os.attribute("description", llvm::join(pureDescriptionLines, "\n"));
  }

  //===--------------------------------------------------------------------===//
  // Struct Generation

  void generateJSONFor(ASTDecl &decl, StructDeclOp structOp) {
    os.object([&] {
      os.attribute("kind", "struct");
      os.attribute("name", structOp.getName());

      // Emit the doc string if present.
      if (std::optional<DocString> docStr = getDocString(decl)) {
        os.attribute("summary", docStr->getSummary());

        // Grab the types of the parameters to the struct.
        llvm::StringMap<std::string> paramToDetail;
        for (auto [index, value] : llvm::enumerate(structOp.getInputParams()))
          paramToDetail[value.getName()] = generateTypeString(value.getType());
        processStructDocDescription(docStr->getDescription(), paramToDetail);
      }

      // Recursively generate documentation for the module's children.
      generateJSONForChildren(decl);
    });
  }

  void processStructDocDescription(
      ArrayRef<StringRef> description,
      const llvm::StringMap<std::string> &paramToDetail) {
    // Process the lines of the description, looking for markers.
    SmallVector<StringRef> pureDescriptionLines;
    for (size_t line = 0, lineE = description.size(); line < lineE; ++line) {
      if (description[line] == (Twine(kParameters) + ":").str()) {
        generateArraySection("parameters", description, line, lineE,
                             paramToDetail);
        continue;
      } else if (description[line] == (Twine(kConstraints) + ":").str()) {
        generateParagraphSection("constraints", description, line, lineE);
        continue;
      }
      pureDescriptionLines.push_back(description[line]);
    }
    os.attribute("description", llvm::join(pureDescriptionLines, "\n"));
  }

  //===--------------------------------------------------------------------===//
  // Module Generation

  void generateJSONFor(ASTDecl &decl, FileModuleOp moduleOp) {
    os.object([&] {
      os.attribute("kind", "module");

      StringRef name = moduleOp.getName();
      name.consume_front("$");
      os.attribute("name", name);

      // Emit the doc string if present.
      if (std::optional<DocString> docStr = getDocString(decl)) {
        os.attribute("summary", docStr->getSummary());
        os.attribute("description", llvm::join(docStr->getDescription(), "\n"));
      }

      // Recursively generate documentation for the module's children.
      generateJSONForChildren(decl);
    });
  }

  //===--------------------------------------------------------------------===//
  // Fields

  /// The output stream.
  llvm::json::OStream os;
};
} // namespace

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

void M::KGEN::LIT::generateMojoDocJSON(ASTDecl &decl, raw_ostream &os) {
  JSONGenerator generator(os);
  generator.generate(decl);
}

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

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
static bool isValidFirstCharacter(char c) { return c == '`' || isupper(c); }

class DocStringValidator {
public:
  DocStringValidator(SharedState &sharedState) : sharedState(sharedState) {}

  void validate(ASTDecl &decl) {
    std::optional<DocString> docStr = getDocString(decl);
    if (!decl.hasReferenceError) {
      TypeSwitch<ASTDecl &>(decl)
          .Case<LIT::FuncOp, StructDeclOp, StructFieldOp>([&](auto op) {
            ValidationKind validation = requiresDocString(op)
                                            ? ValidationKind::Strict
                                            : ValidationKind::Normal;
            if (!docStr) {
              if (validation == ValidationKind::Strict)
                sharedState.emitWarning(op.getLoc(), "public symbol '")
                    << op.getName() << "' is missing a doc string";
              return;
            }

            if (validation == ValidationKind::Strict) {
              // Check that the summary line is a complete sentence: it begins
              // with a capital letter (or a punctuator such as '`'), and ends
              // with a period.
              StringRef summary = docStr->getSummary();
              if (!isValidFirstCharacter(summary.front()))
                sharedState.emitWarning(docStr->getLoc(),
                                        "doc string summary should begin with "
                                        "a capital letter or '`', but this "
                                        "begins with '")
                    << summary.front() << "'";

              if (!summary.ends_with("."))
                sharedState.emitWarning(
                    docStr->getLoc(),
                    "doc string summary should end with a period '.', but this "
                    "ends with '")
                    << summary.back() << "'";
            }

            validateDecl(decl, op, *docStr, validation);
          });
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
      sharedState.emitWarning(SMLoc::getFromPointer(first), name)
          << " should begin with a capital letter or '`', but this begins"
             " with '"
          << *first << "'";

    if (*last != '.')
      sharedState.emitWarning(SMLoc::getFromPointer(last), name)
          << " should end with a period '.', but this ends with '" << *last
          << "'";
  }

  //===----------------------------------------------------------------------===//
  // Arguments and Parameters

  /// Process a parameter or argument section.
  void processParamOrArgs(SMLoc loc, StringRef tag,
                          llvm::MapVector<StringRef, SMLoc> &elements,
                          ArrayRef<StringRef> &lines,
                          ValidationKind validation) {
    StringRef sectionLine = lines[0];
    bool emittedUnexpectedOrderWarning = false;
    ptrdiff_t nextEltIndex = 0;
    SmallVector<SMLoc> elementDocEndLocs(elements.size());
    process2ColumnDocSection(lines, [&](StringRef paramName,
                                        StringRef paramBody) {
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
        sharedState.emitWarning(paramLoc)
            << "duplicate " << tag << " '" << paramName << "' in doc string";
        return;
      }

      // Ensure the elements are in the same order as the decl.
      if (!emittedUnexpectedOrderWarning) {
        size_t expectedEltIndex = it - elements.begin();
        if (currentEltIndex != expectedEltIndex) {
          sharedState.emitWarning(paramLoc)
              << "'" << paramName << "' is defined at index "
              << expectedEltIndex << ", but specified in doc string at index "
              << currentEltIndex;
          emittedUnexpectedOrderWarning = true;
        }
      }

      // Diagnose empty element descriptions.
      SMLoc docEndLoc;
      if (paramBody.empty()) {
        sharedState.emitWarning(paramLoc)
            << "'" << paramName << "' does not have a description";
        docEndLoc = SMLoc::getFromPointer(paramName.end());
      } else {
        docEndLoc = SMLoc::getFromPointer(paramBody.end());
      }

      // Diagnose descriptions with poor style.
      if (validation == ValidationKind::Strict && !paramBody.empty())
        validateStyle((Twine("'") + paramName + "' description").str(),
                      paramBody.begin(), paramBody.end() - 1);

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
      InflightDiag diag = sharedState.emitWarning(loc)
                          << tag << " '" << element << "' is not documented";

      // Attach a fixit to add the element to the doc string.
      SMLoc prevEndLoc = (i == 0) ? sectionEndLoc : elementDocEndLocs[i - 1];
      diag.addFixIt(
          FixIt(SourceRange::getByteLevel(prevEndLoc, prevEndLoc),
                "\n" + indentStr + std::string(4, ' ') + element + ":"));
      elementDocEndLocs[i] = prevEndLoc;
    }
  }
  void processArguments(SMLoc loc, llvm::MapVector<StringRef, SMLoc> &elements,
                        ArrayRef<StringRef> &lines, ValidationKind validation) {
    processParamOrArgs(loc, "argument", elements, lines, validation);
  }
  void processParameters(SMLoc loc, llvm::MapVector<StringRef, SMLoc> &elements,
                         ArrayRef<StringRef> &lines,
                         ValidationKind validation) {
    processParamOrArgs(loc, "parameter", elements, lines, validation);
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

      // Check that text follows the section header. For example, this should
      // diagnose a doc string that ends with `Returns:"""`.
      if (lines.size() == 1) {
        sharedState.emitWarning(sectionLoc,
                                "'" + section + "' section is empty");
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
  void validateDecl(ASTDecl &decl, LIT::FuncOp funcOp, DocString &docStr,
                    ValidationKind validation) {
    // In general, each function argument must be documented, but exceptions are
    // pruned from the list below.
    ArrayRef<StringAttr> argNames = funcOp.getValueParamNames();
    // The compiler can insert an implicit `__result__` argument, which stores
    // memory-only results, at the beginning of an argument list.  Because these
    // arguments are hidden artifacts of the compiler, they don't need to be
    // documented.
    if (funcOp.getSignature().hasMemoryOnlyResult())
      argNames = argNames.drop_front();
    // Methods take `self` as an explicit first argument, for which
    // documentation isn't required.
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
        {kConstraints, SMLoc()},
        {kArgs, SMLoc()},
        {kParameters, SMLoc()},
        {kReturns, SMLoc()},
    };
    ArrayRef<StringRef> description = docStr.getDescription();
    bool hasResults = !ASTType(funcOp.getUserResultType()).isNoneType();

    auto processFn = [&](StringRef section, SMLoc loc) mutable {
      if (section == kArgs)
        return processArguments(loc, seenArguments, description, validation);

      if (section == kParameters)
        return processParameters(loc, seenParameters, description, validation);

      if (section == kReturns && !hasResults)
        sharedState.emitWarning(loc, "unexpected 'Returns' in doc string for "
                                     "function with no results");

      // Validate paragraph sections such as "Constraints:" and "Returns:".
      if (validation == ValidationKind::Strict) {
        StringRef firstLine = description[1].ltrim();
        StringRef lastLine = description.back().rtrim();
        validateStyle("section body", firstLine.begin(), lastLine.end() - 1);
      }
    };
    processDocSections(description, sections, processFn);

    if (validation == ValidationKind::Strict) {
      if (!sections[kParameters].isValid() && !seenParameters.empty())
        sharedState.emitWarning(
            funcOp.getLoc(),
            "function takes parameters, but no 'Parameters' in doc string");
      if (!sections[kArgs].isValid() && !seenArguments.empty())
        sharedState.emitWarning(
            funcOp.getLoc(),
            "function takes arguments, but no 'Args' in doc string");
      if (!sections[kReturns].isValid() && hasResults)
        sharedState.emitWarning(
            funcOp.getLoc(),
            "function has results, but no 'Returns' in doc string");
    }
  }

  //===--------------------------------------------------------------------===//
  // Structs

  void validateDecl(ASTDecl &decl, StructDeclOp structOp, DocString &docStr,
                    ValidationKind validation) {
    // Grab the parameters to the struct.
    llvm::MapVector<StringRef, SMLoc> seenParameters;
    for (auto [index, value] : llvm::enumerate(structOp.getInputParams()))
      seenParameters.insert({value.getName(), SMLoc()});

    // Process the sections of the doc string.
    DenseMap<StringRef, SMLoc> sections = {
        {kParameters, SMLoc()},
    };
    ArrayRef<StringRef> description = docStr.getDescription();
    auto processFn = [&](StringRef section, SMLoc loc) mutable {
      if (section == kParameters)
        processParameters(loc, seenParameters, description, validation);
    };
    processDocSections(description, sections, processFn);

    if (validation == ValidationKind::Strict &&
        !sections[kParameters].isValid() && !seenParameters.empty())
      sharedState.emitWarning(
          structOp.getLoc(),
          "struct takes parameters, but no 'Parameters' in doc string");
  }

  //===--------------------------------------------------------------------===//
  // Fields

  void validateDecl(ASTDecl &decl, StructFieldOp fieldOp, DocString &docStr,
                    ValidationKind validation) {
    // Nothing to do.
  }

  /// Reference to the main shared state.
  SharedState &sharedState;
};
} // namespace

void M::KGEN::LIT::validateDocString(SharedState &sharedState, ASTDecl &decl) {
  DocStringValidator validator(sharedState);
  validator.validate(decl);
}
