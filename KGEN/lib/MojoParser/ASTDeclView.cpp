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

/// Return if a decl should be hidden given its name.
static bool shouldHideName(StringRef name) {
  // Non-underscore names are never hidden.
  if (!name.startswith("_"))
    return false;

  // Keep special language names, which have leading and trailing underscores,
  // even though they start with `_`.
  return !(name.startswith("__") && name.endswith("__"));
}

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

/// Extract a list of directly child alias decls from a given decl. It omits
/// aliases whose name start with _.
static SmallVector<AliasDeclView> extractChildAliases(ASTDecl &decl) {
  SmallVector<AliasDeclView> aliases;

  for (const auto &[name, decls] : decl.getDeclsInScope()) {
    if (shouldHideName(name) || decls.empty())
      continue;
    if (!isa<AliasDeclOp>(**decls.begin()))
      continue;

    for (auto &child : decls) {
      // Skip declarations that were imported from other scopes.
      if (child->getParentDecl() == &decl)
        aliases.push_back(
            cast<AliasDeclView>(*MojoASTDeclRef(child).getView()));
    }
  }

  return aliases;
}

//===----------------------------------------------------------------------===//
// DeclView
//===----------------------------------------------------------------------===//

std::string ParameterDeclView::getDeclarationSnippet() const {
  std::string buff;
  llvm::raw_string_ostream os(buff);
  dumpParamOrArg(os, getName(), type);
  return buff;
}

//===----------------------------------------------------------------------===//
// ParameterDeclView
//===----------------------------------------------------------------------===//

llvm::json::Object ParameterDeclView::toJSON() const {
  return llvm::json::Object{{"kind", "parameter"},
                            {"name", getName()},
                            {"type", type},
                            {"description", description}};
}

//===----------------------------------------------------------------------===//
// ArgumentDeclView
//===----------------------------------------------------------------------===//

std::string ArgumentDeclView::getDeclarationSnippet() const {
  std::string buff;
  llvm::raw_string_ostream os(buff);
  if (inout)
    os << "inout ";
  dumpParamOrArg(os, getName(), type);
  return buff;
}

llvm::json::Object ArgumentDeclView::toJSON() const {
  return llvm::json::Object{
      {"description", description}, {"inout", inout}, {"kind", "parameter"},
      {"name", getName()},          {"type", type},
  };
}

//===----------------------------------------------------------------------===//
// AliasDeclView
//===----------------------------------------------------------------------===//

std::string AliasDeclView::getDeclarationSnippet() const { return {}; }

llvm::json::Object AliasDeclView::toJSON() const {
  return llvm::json::Object{{"description", description},
                            {"kind", "alias"},
                            {"name", getName()},
                            {"summary", summary},
                            {"value", value}};
}

AliasDeclView::AliasDeclView(MojoASTDeclRef declRef)
    : DeclView(DK_AliasDeclView, declRef.getName().value_or(StringRef())) {
  ASTDecl &decl = *reinterpret_cast<ASTDecl *>(declRef.getAsVoidPointer());
  auto aliasOp = cast<LIT::AliasDeclOp>(declRef.getIfOperation());

  llvm::raw_string_ostream valueOS(value);
  PValue(aliasOp.getValue()).printForDiag(valueOS);

  if (auto docStr = decl.getParsedDocString()) {
    summary = docStr->getSummary();
    description = llvm::join(docStr->getDescription(), "\n");
  }
}

//===----------------------------------------------------------------------===//
// FunctionDeclView
//===----------------------------------------------------------------------===//

void FunctionDeclView::augmentWithDocumentation(ArrayRef<StringRef> desc) {
  // Process the lines of the description, looking for markers.
  SmallVector<StringRef> pureDescriptionLines;
  for (size_t line = 0, lineE = desc.size(); line < lineE; ++line) {
    if (desc[line] == (Twine(kArgs) + ":").str()) {
      augmentDeclsWithDocumentation(desc, line, lineE, args);
    } else if (desc[line] == (Twine(kParameters) + ":").str()) {
      augmentDeclsWithDocumentation(desc, line, lineE, parameters);
    } else if (desc[line] == (Twine(kReturns) + ":").str()) {
      if (returnType)
        returns = parseDocStringSection(desc, line, lineE);
    } else if (desc[line] == (Twine(kConstraints) + ":").str()) {
      constraints = parseDocStringSection(desc, line, lineE);
    } else {
      pureDescriptionLines.push_back(desc[line]);
    }
  }

  description = llvm::join(pureDescriptionLines, "\n");
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

  if (!summary.empty())
    os << summary << "\n";

  auto hasAnyItemDescription = [](const auto &items) {
    return llvm::any_of(
        items, [](const auto &item) { return !item.getDescription().empty(); });
  };

  if (hasAnyItemDescription(parameters)) {
    os << "\n#### Parameters:\n";
    for (const auto &param : parameters) {
      if (auto desc = param.getDescription(); !desc.empty())
        os << kMarkdownIndent << param.getName() << ": " << desc << "  \n";
    }
  }

  if (hasAnyItemDescription(args)) {
    os << "\n#### Args:\n";
    for (const auto &arg : args)
      if (auto desc = arg.getDescription(); !desc.empty())
        os << kMarkdownIndent << arg.getName() << ": " << desc << "  \n";
  }

  if (!returns.empty())
    os << "\n#### Returns:\n" << kMarkdownIndent << returns << "\n";

  if (!constraints.empty())
    os << "\n#### Constraints:\n" << kMarkdownIndent << constraints << "\n";

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
  llvm::json::Object result{
      {"async", isAsync()},
      {"constraints", constraints},
      {"description", getDescription()},
      {"isDef", isDef()},
      {"kind", "function"},
      {"name", getName()},
      {"raises", raises()},
      {"returns", returns},
      {"returnType", returnType},
      {"signature", getSignature()},
      {"summary", summary},
  };

  llvm::json::Array jsonArgs;
  for (const auto &arg : args)
    jsonArgs.push_back(arg.toJSON());
  result.insert({"args", std::move(jsonArgs)});

  llvm::json::Array jsonParameters;
  for (const auto &param : parameters)
    jsonParameters.push_back(param.toJSON());
  result.insert({"parameters", std::move(jsonParameters)});

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

  if (auto docStr = decl.getParsedDocString()) {
    summary = docStr->getSummary();
    augmentWithDocumentation(docStr->getDescription());
  }

  raisesFlag = funcOp.isThrows();
  isAsyncFlag = funcOp.isAsync();
  isDefFlag = funcOp.getIsDef();
}

//===----------------------------------------------------------------------===//
// ModuleDeclView
//===----------------------------------------------------------------------===//

std::string ModuleDeclView::getDeclarationSnippet() const { return {}; }

llvm::json::Object ModuleDeclView::toJSON() const {
  llvm::json::Object result{{"description", description},
                            {"kind", "module"},
                            {"name", getName()},
                            {"summary", summary}};

  llvm::json::Array jsonAliases;
  for (const auto &alias : aliases)
    jsonAliases.push_back(alias.toJSON());
  result.insert({"aliases", std::move(jsonAliases)});

  return result;
}

ModuleDeclView::ModuleDeclView(MojoASTDeclRef declRef)
    : DeclView(DK_ModuleDeclView, declRef.getName().value_or(StringRef())) {
  ASTDecl &decl = *reinterpret_cast<ASTDecl *>(declRef.getAsVoidPointer());

  aliases = extractChildAliases(decl);

  if (auto docStr = decl.getParsedDocString()) {
    summary = docStr->getSummary();
    description = llvm::join(docStr->getDescription(), "\n");
  }
}
