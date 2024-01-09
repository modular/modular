//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTooling/ASTDeclView.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DocString.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
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
  if (convention && SignatureType::hasAddress(*convention))
    astType = astType.getReferenceElementType();

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
  os << identifier;
  if (!type.empty())
    os << ": " << type;
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
static SmallVector<DeclViewType, 2> extractChildDecls(const ASTDecl &decl) {
  DenseSet<Operation *> seenOps;
  SmallVector<DeclViewType, 2> children;

  for (const auto &[name, decls] : decl.getDeclsInScope()) {
    if (shouldHideName(name) || decls.empty() || !isa<OpType>(**decls.begin()))
      continue;

    for (auto &child : decls) {
      if (!isa<OpType>(*child))
        continue;
      // Skip declarations that were imported from other scopes.
      if (child->getParentDecl() != &decl ||
          !seenOps.insert(child->getIfOperation()).second)
        continue;
      // Skip synthetic declarations that don't have accompanying documentation
      // generated with them.
      // FIXME(#26535): Use a proper API to check if a decl is synthetic.
      if (child->getIfOperation()->getLoc() ==
              decl.getIfOperation()->getLoc() &&
          !child->getDocString())
        continue;

      children.push_back(cast<DeclViewType>(*MojoASTDeclRef(child).getView()));
    }
  }

  llvm::stable_sort(children, [](auto &lhs, auto &rhs) {
    return compareDeclNames(lhs.getName(), rhs.getName());
  });

  return children;
}

template <typename JSONSerializableItems>
static llvm::json::Array toJSONArray(MojoParserContext &ctx,
                                     const JSONSerializableItems &items) {
  llvm::json::Array jsonItems;
  for (const auto &item : items)
    jsonItems.push_back(item.toJSON(ctx));
  return jsonItems;
}

/// Dump the markdown header common to all decls that support docstring
/// documentation.
static void dumpMarkdownDocumentationHeader(llvm::raw_ostream &os,
                                            StringRef summary,
                                            StringRef description = {}) {
  if (!summary.empty())
    os << summary << "\n";

  if (!description.empty())
    os << "\n" << description << "\n";
}

static void dumpMarkdownSectionTitle(llvm::raw_ostream &os, StringRef title) {
  os << "\n#### " << title << ":\n";
}

/// Dump a markdown section with a list of decls. Each decl is printed with the
/// format `name: description`. Decls without description are ommitted, and the
/// section title is only dumped if there is at least one decl to show.
template <typename DeclViewList>
static void dumpMarkdownDeclListSection(llvm::raw_ostream &os,
                                        StringRef sectionTitle,
                                        const DeclViewList &decls) {
  bool isFirst = true;
  for (const auto &decl : decls) {
    if (decl.getDescription().empty())
      continue;

    if (isFirst) {
      isFirst = false;
      /// We only show the section title if there's at least one item to show.
      dumpMarkdownSectionTitle(os, sectionTitle);
    } else {
      /// This is a special separator for unbulleted lists.
      os << "\\\n";
    }
    os << kMarkdownIndent << decl.getName() << ": " << decl.getDescription()
       << "\n";
  }
}

/// Dump a markdown section with plain text as content and a section title. The
/// section is only dumped if the text is not empty.
static void dumpMarkdownTextSection(llvm::raw_ostream &os,
                                    StringRef sectionTitle, StringRef text) {
  if (!text.empty()) {
    dumpMarkdownSectionTitle(os, sectionTitle);
    os << kMarkdownIndent << text << "\n";
  }
}

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
  case DeclViewKind::DK_AliasDeclView:
    return "alias";
  case DeclViewKind::DK_ArgumentDeclView:
    return "argument";
  case DeclViewKind::DK_FunctionDeclView:
    return "function";
  case DeclViewKind::DK_ModuleDeclView:
    return "module";
  case DeclViewKind::DK_PackageDeclView:
    return "package";
  case DeclViewKind::DK_ParameterDeclView:
    return "parameter";
  case DeclViewKind::DK_StructDeclView:
    return "struct";
  case DeclViewKind::DK_StructFieldDeclView:
    return "field";
  case DeclViewKind::DK_TraitDeclView:
    return "trait";
  case DeclViewKind::DK_VariableDeclView:
    return "variable";
  }
  llvm_unreachable("invalid kind");
}

std::string DeclView::getFullMarkdownString() const {
  std::string buff;
  llvm::raw_string_ostream os(buff);

  // A code snippet used when rendering the documentation string.
  const char *docStringSnippet = R"(
---

###
{0}
)";

  // A code snippet used when rendering the declaration snippet.
  const char *declarationSnippet = R"(
---

###
```mojo
{0}
```)";

  os << formatv("### {0} `{1}`\n", getKindAsString(), getName());
  if (auto docString = getMarkdownDocString(); !docString.empty())
    os << llvm::formatv(docStringSnippet, docString);

  os << llvm::formatv(declarationSnippet, getDeclarationSnippet());
  return buff;
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

llvm::json::Object VariableDeclView::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{{"isVar", isVar()},
                            {"kind", getKindAsString()},
                            {"name", getName().str()},
                            {"type", type}};
}

VariableDeclView::VariableDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_VariableDeclView,
               declRef.getName().value_or(StringRef{})),
      isGlobalVariable(false) {
  TypeSwitch<mlir::Operation *>(declRef.getIfOperation())
      .Case([&](VarLetDeclOp op) {
        flagIsVar = op.getKind() != VarLetDeclKind::Let;
        type = declRef.getType().getReferenceElementType().getAsString();
      })
      .Case([&](LetRegDeclOp op) {
        flagIsVar = false;
        type = declRef.getType().getAsString();
      })
      .Case([&](GlobalVarDeclOp op) {
        flagIsVar = op.getIsVar();
        type = declRef.getType().getAsString();
        isGlobalVariable = true;
      });
}

//===----------------------------------------------------------------------===//
// ParameterDeclView
//===----------------------------------------------------------------------===//

std::string ParameterDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, description);
  return markdown;
}

llvm::json::Object ParameterDeclView::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{{"kind", getKindAsString()},
                            {"name", getName().str()},
                            {"type", type},
                            {"description", description}};
}

//===----------------------------------------------------------------------===//
// ArgumentDeclView
//===----------------------------------------------------------------------===//

std::string ArgumentDeclView::getDeclarationSnippet() const {
  std::string buff;
  llvm::raw_string_ostream os(buff);

  // We don't print the `borrowed` convention because that's the default for all
  // args.
  if (inout)
    os << "inout ";
  if (owned)
    os << "owned ";

  dumpIdentifierWithType(os, getName(), type);
  return buff;
}

std::string ArgumentDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, description);
  return markdown;
}

llvm::json::Object ArgumentDeclView::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{
      {"description", description},
      {"inout", inout},
      {"kind", getKindAsString()},
      {"name", getName().str()},
      {"owned", owned},
      {"type", type},
      {"passingKind", stringifyPassingKind(passingKind)},
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
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object AliasDeclView::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{{"description", description},
                            {"kind", getKindAsString()},
                            {"name", getName().str()},
                            {"summary", summary},
                            {"value", value}};
}

/// Return if the given alias decl is global, i.e. nested within a module,
/// package, or struct.
static bool isGlobalAliasDecl(MojoASTDeclRef declRef) {
  return isa<FileModuleOp, PackageOp, StructDeclOp>(*declRef->getParentDecl());
}

AliasDeclView::AliasDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_AliasDeclView,
               declRef.getName().value_or(StringRef())),
      isGlobalAlias(isGlobalAliasDecl(declRef)) {
  auto aliasOp = cast<LIT::AliasDeclOp>(declRef->getIfOperation());

  llvm::raw_string_ostream valueOS(value);
  PValue(aliasOp.getValue()).printForDiag(valueOS);

  if (auto docStr = declRef->getParsedDocString()) {
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
  return getDeclarationSnippet(/*parameterOffsets=*/nullptr,
                               /*argumentOffsets=*/nullptr);
}

std::string FunctionDeclView::getDeclarationSnippet(
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets) const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  if (isAsync())
    os << "async ";

  std::string signature = getSignature(parameterOffsets, argumentOffsets);
  StringRef typeLessSignature = StringRef(signature).split(" ->").first;
  os << (isDef() ? "def" : "fn") << " ";

  // Adjust the signature offsets.
  size_t signatureStart = os.str().size();
  auto adjustOffsets = [&](auto *v) {
    for (auto &offset : *v) {
      offset.first += signatureStart;
      offset.second += signatureStart;
    }
  };
  if (parameterOffsets)
    adjustOffsets(parameterOffsets);
  if (argumentOffsets)
    adjustOffsets(argumentOffsets);

  // Emit the signature.
  os << typeLessSignature;

  if (raises())
    os << " raises";

  if (returnType)
    os << " -> " << *returnType;
  return snippet;
}

std::string FunctionDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);

  dumpMarkdownDocumentationHeader(os, summary, description);
  dumpMarkdownDeclListSection(os, DocString::kSectionParameters, parameters);
  dumpMarkdownDeclListSection(os, DocString::kSectionArgs, args);
  dumpMarkdownTextSection(os, DocString::kSectionReturns, returns);
  dumpMarkdownTextSection(os, DocString::kSectionConstraints, constraints);

  return markdown;
}

std::string FunctionDeclView::getSignature(
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets) const {
  std::string signature;
  llvm::raw_string_ostream signatureOS(signature);

  // Strip off the mangled suffix from the base function name.
  signatureOS << getName().split('(').first;

  // Emit the parameters of the function.
  if (!parameters.empty()) {
    signatureOS << "[";
    interleaveComma(getParameters(), signatureOS, [&](const auto &param) {
      unsigned paramStart = signature.size();
      signatureOS << param.getDeclarationSnippet();
      if (parameterOffsets)
        parameterOffsets->push_back({paramStart, signature.size()});
    });
    signatureOS << "]";
  }

  // Emit the arguments of the function.
  PassingKindPrinter ssPrinter(signatureOS, args.size(),
                               /*suppressSlashAfterSelf=*/isMethod());
  size_t idx = 0;
  auto printArg = [&](const M::ArgumentDeclView &arg) {
    ssPrinter.printOptionalStarSlash(arg.getPassingKind(), idx);

    unsigned argStart = signature.size();
    signatureOS << arg.getDeclarationSnippet();
    if (argumentOffsets)
      argumentOffsets->push_back({argStart, signature.size()});

    // Check if we are at the end; if so, we might still have to print a '/'.
    ssPrinter.printOptionalTrailingSlash(idx++);
  };

  signatureOS << "(";
  interleaveComma(args, signatureOS, printArg);
  signatureOS << ")";

  // Emit the result type.
  if (returnType)
    signatureOS << " -> " << *returnType;
  return signatureOS.str();
}

llvm::json::Object FunctionDeclView::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{
      {"args", toJSONArray(ctx, args)},
      {"async", isAsync()},
      {"constraints", constraints},
      {"description", description},
      {"isDef", isDef()},
      {"isStatic", isStatic()},
      {"kind", getKindAsString()},
      {"name", getName().str()},
      {"parameters", toJSONArray(ctx, parameters)},
      {"raises", raises()},
      {"returns", returns},
      {"returnType", returnType},
      {"signature", getSignature()},
      {"summary", summary},
  };
}

FunctionDeclView::FunctionDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_FunctionDeclView,
               declRef.getName().value_or(StringRef{})) {
  auto funcOp = cast<LIT::FuncOp>(declRef.getIfOperation());

  ArrayRef<Type> argTypes = funcOp.getArgumentTypes();
  ArrayRef<StringAttr> argNames = funcOp.getSignature().getArgNames();
  ArrayRef<ValueInputConvention> argConventions =
      funcOp.getSignature().getInputConventions();
  ASTType resultType = funcOp.getUserResultType();
  ArrayRef<PassingKind> argPassingKinds =
      funcOp.getSignature().getArgPassingKinds();

  // Check for a by-ref result type, which gets modeled as the first argument
  // (as it needs to be passed through memory), and we don't want to include
  // it in the normal argument list.
  if (!argConventions.empty() &&
      argConventions.front() == ValueInputConvention::ByRefResult) {
    argTypes = argTypes.drop_front();
    argNames = argNames.drop_front();
    argConventions = argConventions.drop_front();
    argPassingKinds = argPassingKinds.drop_front();
  }

  // If this is a method, grab the expected "Self" type.
  std::optional<ASTType> selfType;
  if (isa<StructDeclOp>(funcOp->getParentOp()))
    selfType = declRef->getParentDecl()->getSelfType();

  // Grab the types of the arguments to the function.
  for (auto [type, name, convention, passingKind] :
       llvm::zip(argTypes, argNames, argConventions, argPassingKinds))
    args.push_back(ArgumentDeclView(
        name.getValue(), generateTypeString(type, selfType, convention),
        passingKind,
        /*inout=*/convention == ValueInputConvention::ByRef ||
            convention == ValueInputConvention::InitSelf,
        /*owned=*/convention == ValueInputConvention::OwnedInMem ||
            convention == ValueInputConvention::OwnedInReg));

  // Grab the types of the parameters to the function.
  size_t numImplicitLifetimes =
      funcOp.getSignature().getNumImplicitLifetimeDecls();
  for (ParamDeclAttr param :
       funcOp.getInputParams().drop_front(numImplicitLifetimes))
    parameters.push_back(
        ParameterDeclView(demangleIfNeeded(param).getName().getValue(),
                          generateTypeString(param.getType(), selfType)));

  // Grab the result type, if it's non-none.
  if (!resultType.isNoneType())
    returnType = generateTypeString(resultType, selfType);

  if (auto docStr = declRef->getParsedDocString()) {
    summary = docStr->getSummary();
    augmentWithDocumentation(docStr->getDescription());
  }

  raisesFlag = funcOp.isThrows();
  isAsyncFlag = funcOp.isAsync();
  isStaticFlag = funcOp.getIsStatic();
  isMethodFlag = !isStaticFlag && isa<StructDeclOp>(funcOp->getParentOp());
  isDefFlag = funcOp.getIsDef();
}

//===----------------------------------------------------------------------===//
// StructFieldDeclView
//===----------------------------------------------------------------------===//

std::string StructFieldDeclView::getDeclarationSnippet() const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << "var ";
  dumpIdentifierWithType(os, getName(), type);
  return snippet;
}

std::string StructFieldDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object StructFieldDeclView::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{
      {"description", description},
      {"kind", getKindAsString()},
      {"name", getName().str()},
      {"summary", summary},
      {"type", type},
  };
}

StructFieldDeclView::StructFieldDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_StructFieldDeclView,
               declRef.getName().value_or(StringRef{})) {
  auto fieldOp = cast<StructFieldOp>(declRef.getIfOperation());

  llvm::raw_string_ostream typeOS(type);
  ASTType(fieldOp.getType()).print(typeOS, /*forDiag=*/true);

  if (std::optional<DocString> docStr = declRef->getParsedDocString()) {
    summary = docStr->getSummary();
    description = llvm::join(docStr->getDescription(), "\n");
  }
}

//===----------------------------------------------------------------------===//
// FunctionDeclOverloadSetView
//===----------------------------------------------------------------------===//

SmallVector<FunctionDeclOverloadSetView, 2>
FunctionDeclOverloadSetView::fromSortedFunctions(
    SmallVector<FunctionDeclView, 2> &&functions) {
  SmallVector<FunctionDeclOverloadSetView, 2> overloads;
  for (auto &function : functions) {
    if (overloads.empty() ||
        overloads.back().getBaseName() != function.getName())
      overloads.emplace_back(FunctionDeclOverloadSetView(function.getName()));

    overloads.back().append(std::move(function));
  }
  return overloads;
}

llvm::json::Object
FunctionDeclOverloadSetView::toJSON(MojoParserContext &ctx) const {
  return llvm::json::Object{{"kind", "function"},
                            {"name", baseName},
                            {"overloads", toJSONArray(ctx, functions)}};
}

//===----------------------------------------------------------------------===//
// TraitDeclView
//===----------------------------------------------------------------------===//

/// Collect the names of the various parent types of the given set of type
/// lineages.
/// TODO: Whenever we support inherited classes/structs, collect those as well.
static void collectParentTypes(MojoParserContext &ctx,
                               SmallVectorImpl<StringRef> &parentTraits,
                               ArrayRef<TypeLineageAttr> parentTypes) {
  DenseSet<Type> seenTypes;
  auto addParentType = [&](Type parentType) {
    if (!seenTypes.insert(parentType).second)
      return;
    MojoASTDeclRef decl = ctx.getDecl(parentType);
    if (!decl)
      return;
    std::optional<StringRef> name = decl.getName();
    if (!name)
      return;
    if (isa<TraitDeclOp>(*decl))
      parentTraits.push_back(*name);
  };

  for (TypeLineageAttr parentType : parentTypes) {
    addParentType(parentType.getType());
    for (Type type : parentType.getInheritedFrom())
      addParentType(type);
  }

  llvm::sort(parentTraits);
}

std::string TraitDeclView::getDeclarationSnippet() const {
  return "trait " + getName().str();
}

std::string TraitDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object TraitDeclView::toJSON(MojoParserContext &ctx) const {
  auto functionOverloads = FunctionDeclOverloadSetView::fromSortedFunctions(
      extractChildDecls<FunctionDeclView, FuncOp>(*decl));
  SmallVector<StringRef> parentTraits;
  collectParentTypes(ctx, parentTraits,
                     cast<TraitDeclOp>(*decl).getParentTypes());
  return llvm::json::Object{
      {"description", description},
      {"fields", llvm::json::Array()},
      {"functions", toJSONArray(ctx, functionOverloads)},
      {"kind", getKindAsString()},
      {"name", getName().str()},
      {"parentTraits", llvm::json::Array(parentTraits)},
      {"summary", summary},
  };
}

TraitDeclView::TraitDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_TraitDeclView,
               declRef.getName().value_or(StringRef())),
      decl(declRef) {
  if (auto docStr = decl->getParsedDocString()) {
    summary = docStr->getSummary();
    description = llvm::join(docStr->getDescription(), "\n");
  }
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
  return getDeclarationSnippet(/*parameterOffsets=*/nullptr);
}

std::string StructDeclView::getDeclarationSnippet(
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets) const {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  os << "struct " << getName();

  if (!parameters.empty()) {
    os << "[";
    interleaveComma(getParameters(), os, [&](const auto &param) {
      unsigned paramStart = snippet.size();
      os << param.getDeclarationSnippet();
      if (parameterOffsets)
        parameterOffsets->push_back({paramStart, snippet.size()});
    });
    os << "]";
  }

  return snippet;
}

std::string StructDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);

  dumpMarkdownDocumentationHeader(os, summary, description);
  dumpMarkdownDeclListSection(os, DocString::kSectionParameters, parameters);
  dumpMarkdownTextSection(os, DocString::kSectionConstraints, constraints);

  return markdown;
}

llvm::json::Object StructDeclView::toJSON(MojoParserContext &ctx) const {
  auto aliases = extractChildDecls<AliasDeclView, AliasDeclOp>(*decl);
  auto fields = extractChildDecls<StructFieldDeclView, StructFieldOp>(*decl);
  auto functionOverloads = FunctionDeclOverloadSetView::fromSortedFunctions(
      extractChildDecls<FunctionDeclView, FuncOp>(*decl));
  SmallVector<StringRef> parentTraits;
  collectParentTypes(ctx, parentTraits,
                     cast<StructDeclOp>(*decl).getParentTypes());
  return llvm::json::Object{
      {"aliases", toJSONArray(ctx, aliases)},
      {"constraints", constraints},
      {"description", description},
      {"fields", toJSONArray(ctx, fields)},
      {"functions", toJSONArray(ctx, functionOverloads)},
      {"kind", getKindAsString()},
      {"name", getName().str()},
      {"parameters", toJSONArray(ctx, parameters)},
      {"parentTraits", llvm::json::Array(parentTraits)},
      {"summary", summary},
  };
}

StructDeclView::StructDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_StructDeclView,
               declRef.getName().value_or(StringRef())),
      decl(declRef) {
  auto structOp = cast<StructDeclOp>(declRef.getIfOperation());

  // Grab the types of the parameters to the struct.
  for (ParamDeclAttr param : structOp.getInputParams())
    parameters.push_back(
        ParameterDeclView(demangleIfNeeded(param).getName().getValue(),
                          generateTypeString(param.getType())));

  if (auto docStr = decl->getParsedDocString()) {
    summary = docStr->getSummary();
    augmentWithDocumentation(docStr->getDescription());
  }
}

//===----------------------------------------------------------------------===//
// ModuleDeclView
//===----------------------------------------------------------------------===//

std::string ModuleDeclView::getDeclarationSnippet() const { return {}; }

std::string ModuleDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object ModuleDeclView::toJSON(MojoParserContext &ctx) const {
  auto aliases = extractChildDecls<AliasDeclView, AliasDeclOp>(*decl);
  auto structs = extractChildDecls<StructDeclView, StructDeclOp>(*decl);
  auto traits = extractChildDecls<TraitDeclView, TraitDeclOp>(*decl);
  auto functionOverloads = FunctionDeclOverloadSetView::fromSortedFunctions(
      extractChildDecls<FunctionDeclView, FuncOp>(*decl));

  return llvm::json::Object{{"aliases", toJSONArray(ctx, aliases)},
                            {"description", description},
                            {"functions", toJSONArray(ctx, functionOverloads)},
                            {"kind", getKindAsString()},
                            {"name", getName().str()},
                            {"structs", toJSONArray(ctx, structs)},
                            {"traits", toJSONArray(ctx, traits)},
                            {"summary", summary}};
}

ModuleDeclView::ModuleDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_ModuleDeclView,
               declRef.getName().value_or(StringRef())),
      decl(declRef) {
  if (auto docStr = decl->getParsedDocString()) {
    summary = docStr->getSummary();
    description = llvm::join(docStr->getDescription(), "\n");
  }
}

//===----------------------------------------------------------------------===//
// PackageDeclView
//===----------------------------------------------------------------------===//

std::string PackageDeclView::getDeclarationSnippet() const { return {}; }

std::string PackageDeclView::getMarkdownDocString() const {
  std::string markdown;
  llvm::raw_string_ostream os(markdown);
  dumpMarkdownDocumentationHeader(os, summary, description);
  return markdown;
}

llvm::json::Object PackageDeclView::toJSON(MojoParserContext &ctx) const {
  auto packages = extractChildDecls<PackageDeclView, PackageOp>(*decl);
  auto modules = extractChildDecls<ModuleDeclView, FileModuleOp>(*decl);
  return llvm::json::Object{
      {"description", description},
      {"kind", getKindAsString()},
      {"name", getName().str()},
      {"summary", summary},
      {"modules", toJSONArray(ctx, modules)},
      {"packages", toJSONArray(ctx, packages)},
  };
}

PackageDeclView::PackageDeclView(MojoASTDeclRef declRef)
    : DeclView(DeclViewKind::DK_PackageDeclView,
               declRef.getName().value_or(StringRef())),
      decl(declRef) {
  if (auto docStr = declRef->getParsedDocString()) {
    summary = docStr->getSummary();
    description = llvm::join(docStr->getDescription(), "\n");
  }
}
