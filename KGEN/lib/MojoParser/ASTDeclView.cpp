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

  // If there is no name priority, then leave in the original source order.
  return false;
}

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

// Helper function that dumps an identifier along with an optional
// type. It also takes care of varargs that need to encode * in the name.
static void dumpIdentifierWithType(raw_ostream &os, StringRef identifier,
                                   StringRef type) {
  // If the argument is variadic, we put the star before the name when
  // printing a signature.
  if (type.consume_front("*"))
    os << "*";
  os << identifier << ": " << type;
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

/// Extract a list of direct children decls from a given decl. It omits
/// children whose name start with _, except for special functions that start
/// and end with __.
template <typename DeclViewType, typename OpType>
static SmallVector<DeclViewType, 2> extractChildDecls(ASTDecl &decl) {
  SmallVector<DeclViewType, 2> children;

  for (const auto &[name, decls] : decl.getDeclsInScope()) {
    if (shouldHideName(name) || decls.empty())
      continue;
    if (!isa<OpType>(**decls.begin()))
      continue;

    for (auto &child : decls) {
      // Skip declarations that were imported from other scopes.
      if (child->getParentDecl() == &decl)
        children.push_back(
            cast<DeclViewType>(*MojoASTDeclRef(child).getView()));
    }
  }

  llvm::stable_sort(children, [](auto &lhs, auto &rhs) {
    return compareDeclNames(lhs.getName(), rhs.getName());
  });

  return children;
}

template <typename JSONSerializableItems>
static llvm::json::Array toJSONArray(const JSONSerializableItems &items) {
  llvm::json::Array jsonItems;
  for (const auto &item : items)
    jsonItems.push_back(item.toJSON());
  return jsonItems;
}

// Lambda that returns true if at least one item of the given collection has
// non-empty description.
template <typename Items>
static bool hasAnyItemDescription(const Items &items) {
  return llvm::any_of(
      items, [](const auto &item) { return !item.getDescription().empty(); });
};

//===----------------------------------------------------------------------===//
// DeclView
//===----------------------------------------------------------------------===//

std::string ParameterDeclView::getDeclarationSnippet() const {
  std::string buff;
  llvm::raw_string_ostream os(buff);
  dumpIdentifierWithType(os, getName(), type);
  return buff;
}

StringRef DeclView::getKindAsString() const {
  switch (kind) {
  case DK_AliasDeclView:
    return "alias";
  case DK_ArgumentDeclView:
    return "argument";
  case DK_FunctionDeclView:
    return "function";
  case DK_ModuleDeclView:
    return "module";
  case DK_ParameterDeclView:
    return "parameter";
  case DK_StructDeclView:
    return "struct";
  case DK_StructFieldDeclView:
    return "field";
  case DK_VariableDeclView:
    return "variable";
  }
}

//===----------------------------------------------------------------------===//
// VariableDeclView
//===----------------------------------------------------------------------===//

std::string VariableDeclView::getDeclarationSnippet() const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << (isVar() ? "var" : "let") << " ";
  dumpIdentifierWithType(os, getName(), type);
  return snippet;
}

llvm::json::Object VariableDeclView::toJSON() const {
  return llvm::json::Object{{"isVar", isVar()},
                            {"kind", getKindAsString()},
                            {"name", getName()},
                            {"type", type}};
}

VariableDeclView::VariableDeclView(MojoASTDeclRef declRef)
    : DeclView(DK_VariableDeclView, declRef.getName().value_or(StringRef{})) {
  if (auto op = dyn_cast<LIT::VarLetDeclOp>(declRef.getIfOperation())) {
    flagIsVar = op.getIsVar();
    type = declRef.getType().getPointerElementType().getAsString();
  } else if (auto op = cast<LIT::LetRegDeclOp>(declRef.getIfOperation())) {
    flagIsVar = false;
    type = declRef.getType().getAsString();
  }
}

//===----------------------------------------------------------------------===//
// ParameterDeclView
//===----------------------------------------------------------------------===//

llvm::json::Object ParameterDeclView::toJSON() const {
  return llvm::json::Object{{"kind", getKindAsString()},
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
  dumpIdentifierWithType(os, getName(), type);
  return buff;
}

llvm::json::Object ArgumentDeclView::toJSON() const {
  return llvm::json::Object{
      {"description", description},
      {"inout", inout},
      {"kind", getKindAsString()},
      {"name", getName()},
      {"type", type},
  };
}

//===----------------------------------------------------------------------===//
// AliasDeclView
//===----------------------------------------------------------------------===//

std::string AliasDeclView::getDeclarationSnippet() const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << "alias " << getName();
  if (!value.empty())
    os << " = " << value;
  return snippet;
}

std::string AliasDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);

  if (!summary.empty())
    os << summary << "\n";

  if (!description.empty())
    os << "\n" << description << "\n";

  return markdown;
}

llvm::json::Object AliasDeclView::toJSON() const {
  return llvm::json::Object{{"description", description},
                            {"kind", getKindAsString()},
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
    if (desc[line] == (Twine(DocString::kSectionArgs) + ":").str()) {
      augmentDeclsWithDocumentation(desc, line, lineE, args);
    } else if (desc[line] ==
               (Twine(DocString::kSectionParameters) + ":").str()) {
      augmentDeclsWithDocumentation(desc, line, lineE, parameters);
    } else if (desc[line] == (Twine(DocString::kSectionReturns) + ":").str()) {
      if (returnType)
        returns = parseDocStringSection(desc, line, lineE);
    } else if (desc[line] ==
               (Twine(DocString::kSectionConstraints) + ":").str()) {
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

  if (!description.empty())
    os << "\n" << description << "\n";

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
  return llvm::json::Object{
      {"args", toJSONArray(args)},
      {"async", isAsync()},
      {"constraints", constraints},
      {"description", getDescription()},
      {"isDef", isDef()},
      {"isStatic", isStatic()},
      {"kind", getKindAsString()},
      {"name", getName()},
      {"parameters", toJSONArray(parameters)},
      {"raises", raises()},
      {"returns", returns},
      {"returnType", returnType},
      {"signature", getSignature()},
      {"summary", summary},
  };
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
// StructFieldDeclView
//===----------------------------------------------------------------------===//

std::string StructFieldDeclView::getDeclarationSnippet() const { return {}; }

llvm::json::Object StructFieldDeclView::toJSON() const {
  return llvm::json::Object{
      {"description", description},
      {"kind", getKindAsString()},
      {"name", getName()},
      {"summary", summary},
      {"type", type},
      {"value", value},
  };
}

StructFieldDeclView::StructFieldDeclView(MojoASTDeclRef declRef)
    : DeclView(DK_StructFieldDeclView,
               declRef.getName().value_or(StringRef{})) {
  ASTDecl &decl = *reinterpret_cast<ASTDecl *>(declRef.getAsVoidPointer());
  auto fieldOp = cast<StructFieldOp>(declRef.getIfOperation());

  llvm::raw_string_ostream typeOS(value);
  ASTType(fieldOp.getType()).print(typeOS, /*forDiag=*/true);

  if (std::optional<DocString> docStr = decl.getParsedDocString()) {
    summary = docStr->getSummary();
    description = llvm::join(docStr->getDescription(), "\n");
  }
}

//===----------------------------------------------------------------------===//
// FunctionDeclViewOverloadSet
//===----------------------------------------------------------------------===//

SmallVector<FunctionDeclViewOverloadSet, 2>
FunctionDeclViewOverloadSet::fromSortedFunctions(
    SmallVector<FunctionDeclView, 2> &&functions) {
  SmallVector<FunctionDeclViewOverloadSet, 2> overloads;
  for (auto &function : functions) {
    if (overloads.empty() ||
        overloads.back().getBaseName() != function.getName())
      overloads.emplace_back(FunctionDeclViewOverloadSet(function.getName()));

    overloads.back().append(std::move(function));
  }
  return overloads;
}

llvm::json::Object FunctionDeclViewOverloadSet::toJSON() const {
  return llvm::json::Object{{"kind", "function"},
                            {"name", baseName},
                            {"overloads", toJSONArray(functions)}};
}

//===----------------------------------------------------------------------===//
// StructDeclView
//===----------------------------------------------------------------------===//

void StructDeclView::augmentWithDocumentation(ArrayRef<StringRef> desc) {
  // Process the lines of the description, looking for markers.
  SmallVector<StringRef> pureDescriptionLines;
  for (size_t line = 0, lineE = desc.size(); line < lineE; ++line) {
    if (desc[line] == (Twine(DocString::kSectionParameters) + ":").str())
      augmentDeclsWithDocumentation(desc, line, lineE, parameters);
    else if (desc[line] == (Twine(DocString::kSectionConstraints) + ":").str())
      constraints = parseDocStringSection(desc, line, lineE);
    else
      pureDescriptionLines.push_back(desc[line]);
  }

  description = llvm::join(pureDescriptionLines, "\n");
}

std::string StructDeclView::getDeclarationSnippet() const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << "struct " << getName();

  if (!parameters.empty()) {
    os << "[";
    interleaveComma(getParameters(), os, [&](const auto &param) {
      os << param.getDeclarationSnippet();
    });
    os << "]";
  }

  return snippet;
}

std::string StructDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);

  if (!summary.empty())
    os << summary << "\n";

  if (!description.empty())
    os << "\n" << description << "\n";

  if (hasAnyItemDescription(parameters)) {
    os << "\n#### Parameters:\n";
    for (const auto &param : parameters) {
      if (auto desc = param.getDescription(); !desc.empty())
        os << kMarkdownIndent << param.getName() << ": " << desc << "  \n";
    }
  }

  if (!constraints.empty())
    os << "\n#### Constraints:\n" << kMarkdownIndent << constraints << "\n";

  return markdown;
}

llvm::json::Object StructDeclView::toJSON() const {
  return llvm::json::Object{
      {"aliases", toJSONArray(aliases)},
      {"constraints", constraints},
      {"description", description},
      {"fields", toJSONArray(fields)},
      {"functions", toJSONArray(functionOverloads)},
      {"kind", getKindAsString()},
      {"name", getName()},
      {"parameters", toJSONArray(parameters)},
      {"summary", summary},
  };
}

StructDeclView::StructDeclView(MojoASTDeclRef declRef)
    : DeclView(DK_StructDeclView, declRef.getName().value_or(StringRef())) {
  ASTDecl &decl = *reinterpret_cast<ASTDecl *>(declRef.getAsVoidPointer());
  auto structOp = cast<StructDeclOp>(declRef.getIfOperation());

  aliases = extractChildDecls<AliasDeclView, AliasDeclOp>(decl);
  fields = extractChildDecls<StructFieldDeclView, StructFieldOp>(decl);
  functionOverloads = FunctionDeclViewOverloadSet::fromSortedFunctions(
      extractChildDecls<FunctionDeclView, FuncOp>(decl));

  // Grab the types of the parameters to the struct.
  for (ParamDeclAttr param : structOp.getInputParams())
    parameters.push_back(
        ParameterDeclView(demangleIfNeeded(param).getName().getValue(),
                          generateTypeString(param.getType())));

  if (auto docStr = decl.getParsedDocString()) {
    summary = docStr->getSummary();
    augmentWithDocumentation(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// ModuleDeclView
//===----------------------------------------------------------------------===//

std::string ModuleDeclView::getDeclarationSnippet() const { return {}; }

llvm::json::Object ModuleDeclView::toJSON() const {
  return llvm::json::Object{{"aliases", toJSONArray(aliases)},
                            {"description", description},
                            {"functions", toJSONArray(functionOverloads)},
                            {"kind", getKindAsString()},
                            {"name", getName()},
                            {"structs", toJSONArray(structs)},
                            {"summary", summary}};
}

ModuleDeclView::ModuleDeclView(MojoASTDeclRef declRef)
    : DeclView(DK_ModuleDeclView, declRef.getName().value_or(StringRef())) {
  ASTDecl &decl = *reinterpret_cast<ASTDecl *>(declRef.getAsVoidPointer());

  aliases = extractChildDecls<AliasDeclView, AliasDeclOp>(decl);
  structs = extractChildDecls<StructDeclView, StructDeclOp>(decl);
  functionOverloads = FunctionDeclViewOverloadSet::fromSortedFunctions(
      extractChildDecls<FunctionDeclView, FuncOp>(decl));

  if (auto docStr = decl.getParsedDocString()) {
    summary = docStr->getSummary();
    description = llvm::join(docStr->getDescription(), "\n");
  }
}
