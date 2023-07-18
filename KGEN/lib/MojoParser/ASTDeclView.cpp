//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/ASTDeclView.h"
#include "ASTDecl.h"
#include "DocString.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Two spaces that are forcefully added to markdown lines that can be used for
/// identation.
static constexpr const char *kMarkdownIndent = "&nbsp;&nbsp;";

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

/// Generate a user-readable representation of the given type, with an optional
/// value convention, parent struct "Self" type. It also prepends * to variadic
/// types.
static std::string generateTypeString(
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
  if (convention) {
    switch (*convention) {
    case ValueInputConvention::ByRef:
    case ValueInputConvention::InitSelf:
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

  return os.str();
}

// Helper function that dumps a parameter or argument along with an optional
// type. It also takes care of varargs that need to encode * in the name.
static void dumpParamOrArg(raw_ostream &os, StringRef name, StringRef type) {
  // If the argument is variadic, we put the star before the name when
  // printing a signature.
  if (type.consume_front("*"))
    os << "*";
  os << name << ": " << type;
};

/// Parse the given docstring lines and augment the provided decls with the
/// appropriate documentation using the description.
template <typename DeclViewT>
static void augmentDeclsWithDocumentation(ArrayRef<StringRef> lines,
                                          size_t &line, size_t lineE,
                                          SmallVector<DeclViewT> &decls) {
  std::string fullArgDesc;
  llvm::raw_string_ostream fullArgDescOS(fullArgDesc);
  DenseMap<StringRef, DeclViewT *> declMap;
  for (auto &decl : decls)
    declMap.try_emplace(decl.getName(), &decl);

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
    if (auto it = declMap.find(argName); it != declMap.end()) {
      it->getSecond()->setDescription(fullArgDesc);
    }
  }
}

// Generate a string attribute from the given paragraph form:
///
/// Header:
///   Element1...
static std::string parseDocStringSection(ArrayRef<StringRef> lines,
                                         size_t &line, size_t lineE) {
  // A doc string may end with "Header:". This is diagnosed by the validator,
  // but invalid doc strings may still be emitted as JSON.
  if (line >= lines.size())
    return {};

  std::string paragraph;
  llvm::raw_string_ostream paragraphOS(paragraph);

  paragraphOS << lines[++line].trim();

  // Merge in additional description lines that have equal or larger
  // indentation.
  size_t indent = getIndentationLevel(lines[line]);
  while (++line < lineE && getIndentationLevel(lines[line]) >= indent)
    paragraphOS << " " << lines[line].trim();
  return paragraphOS.str();
}

//===----------------------------------------------------------------------===//
// DeclView
//===----------------------------------------------------------------------===//

std::string ParameterDeclView::getDeclarationSnippet() const {
  std::string buff;
  llvm::raw_string_ostream os(buff);
  dumpParamOrArg(os, getName(), getType());
  return buff;
}

std::optional<StringRef> DeclView::getDescription() const {
  if (description)
    return *description;
  return std::nullopt;
}

llvm::json::Object DeclView::toJSON() const {
  return llvm::json::Object{
      {"name", name}, {"type", type}, {"description", description}};
}

//===----------------------------------------------------------------------===//
// ArgumentDeclView
//===----------------------------------------------------------------------===//

std::string ArgumentDeclView::getDeclarationSnippet() const {
  std::string buff;
  llvm::raw_string_ostream os(buff);
  if (inout)
    os << "inout ";
  dumpParamOrArg(os, getName(), getType());
  return buff;
}

llvm::json::Object ArgumentDeclView::toJSON() const {
  auto result = DeclView::toJSON();
  result.insert({"inout", inout});
  return result;
}

//===----------------------------------------------------------------------===//
// FunctionDeclView
//===----------------------------------------------------------------------===//

void FunctionDeclView::augmentWithDocumentation(
    ArrayRef<StringRef> description) {
  // Process the lines of the description, looking for markers.
  SmallVector<StringRef> pureDescriptionLines;
  for (size_t line = 0, lineE = description.size(); line < lineE; ++line) {
    if (description[line] == (Twine(kArgs) + ":").str()) {
      augmentDeclsWithDocumentation(description, line, lineE, args);
    } else if (description[line] == (Twine(kParameters) + ":").str()) {
      augmentDeclsWithDocumentation(description, line, lineE, parameters);
    } else if (description[line] == (Twine(kReturns) + ":").str()) {
      if (returnType)
        returns = parseDocStringSection(description, line, lineE);
    } else if (description[line] == (Twine(kConstraints) + ":").str()) {
      constraints = parseDocStringSection(description, line, lineE);
    } else {
      pureDescriptionLines.push_back(description[line]);
    }
  }

  setDescription(llvm::join(pureDescriptionLines, "\n"));
}

std::string FunctionDeclView::getDeclarationSnippet() const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  if (isAsync())
    os << "async ";

  std::string signature = getSignature();
  StringRef typeLessSignature = StringRef(signature).split(" ->").first;
  os << (isDef() ? "def" : "fn") << " " << typeLessSignature;

  if (raises())
    os << " raises";

  if (returnType)
    os << " -> " << *returnType;
  return snippet;
}

std::string FunctionDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);

  if (summary)
    os << *summary << "\n";

  auto hasAnyItemDescription = [](const auto &items) {
    return llvm::any_of(items, [](const auto &item) {
      return item.getDescription().has_value();
    });
  };

  if (hasAnyItemDescription(parameters)) {
    os << "\n#### Parameters:\n";
    for (const auto &param : parameters) {
      if (auto desc = param.getDescription())
        os << kMarkdownIndent << param.getName() << ": " << *desc << "  \n";
    }
  }

  if (hasAnyItemDescription(args)) {
    os << "\n#### Args:\n";
    for (const auto &arg : args)
      if (auto desc = arg.getDescription())
        os << kMarkdownIndent << arg.getName() << ": " << *desc << "  \n";
  }

  if (returns)
    os << "\n#### Returns:\n" << kMarkdownIndent << *returns << "\n";

  if (constraints)
    os << "\n#### Constraints:\n" << kMarkdownIndent << *constraints << "\n";

  return markdown;
}

std::string FunctionDeclView::getSignature() const {
  std::string signature;
  llvm::raw_string_ostream signatureOS(signature);

  // Strip off the mangled suffix from the base function name.
  signatureOS << getName().split('(').first;

  // Emit the parameters of the function.
  if (!parameters.empty()) {
    signatureOS << "[";
    interleaveComma(getParameters(), signatureOS, [&](const auto &param) {
      signatureOS << param.getDeclarationSnippet();
    });
    signatureOS << "]";
  }

  // Emit the arguments of the function.
  signatureOS << "(";
  interleaveComma(args, signatureOS, [&](const auto &arg) {
    signatureOS << arg.getDeclarationSnippet();
  });
  signatureOS << ")";

  // Emit the result type.
  if (returnType)
    signatureOS << " -> " << *returnType;
  return signatureOS.str();
}

llvm::json::Object FunctionDeclView::toJSON() const {
  auto result = DeclView::toJSON();

  llvm::json::Array jsonArgs;
  for (const auto &arg : args)
    jsonArgs.push_back(arg.toJSON());
  result.insert({"args", std::move(jsonArgs)});

  llvm::json::Array jsonParameters;
  for (const auto &param : parameters)
    jsonParameters.push_back(param.toJSON());
  result.insert({"parameters", std::move(jsonParameters)});

  result.insert({"signature", getSignature()});
  result.insert({"returnType", returnType});
  result.insert({"async", isAsync()});
  result.insert({"isDef", isDef()});
  result.insert({"raises", raises()});
  result.insert({"summary", summary});
  result.insert({"description", getDescription()});
  result.insert({"returns", returns});
  result.insert({"constraints", constraints});

  return result;
}

FunctionDeclView::FunctionDeclView(MojoASTDeclRef declRef)
    : DeclView(DK_FunctionDeclView, declRef.getName().value_or(StringRef{})) {
  auto funcOp = cast<LIT::FuncOp>(declRef.getIfOperation());

  ASTDecl &decl = *reinterpret_cast<ASTDecl *>(declRef.getAsVoidPointer());

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
  for (auto [type, name, convention] :
       llvm::zip(argTypes, argNames, argConventions))
    args.push_back(ArgumentDeclView(
        name.getValue(), generateTypeString(type, selfType, convention),
        /*inout=*/convention == ValueInputConvention::ByRef ||
            convention == ValueInputConvention::InitSelf));

  // Grab the types of the parameters to the function.
  for (ParamDeclAttr param : funcOp.getInputParams())
    parameters.push_back(
        ParameterDeclView(demangleIfNeeded(param).getName().getValue(),
                          generateTypeString(param.getType(), selfType)));

  // Grab the result type, if it's non-none.
  if (!resultType.isNoneType())
    returnType = generateTypeString(resultType, selfType);

  // Emit the doc string if present.
  if (auto rawDocStr = decl.getDocString()) {
    DocString docStr(rawDocStr);
    summary = docStr.getSummary();
    augmentWithDocumentation(docStr.getDescription());
  }

  raisesFlag = funcOp.isThrows();
  isAsyncFlag = funcOp.isAsync();
  isDefFlag = funcOp.getIsDef();
}
