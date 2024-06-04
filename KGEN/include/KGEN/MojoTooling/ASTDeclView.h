//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOTOOLING_ASTDECLVIEW_H
#define KGEN_MOJOTOOLING_ASTDECLVIEW_H

#include "KGEN/MojoTooling/ASTDeclRef.h"
#include <string>

namespace llvm {
namespace json {
class Object;
} // namespace json
} // namespace llvm

namespace M {

namespace KGEN::LIT {
class LITSignatureType;
enum class PassingKind : uint32_t;
} // namespace KGEN::LIT

class MojoASTDeclRef;
class MojoParserContext;

/// Helper enum to make stringifying of variadic types easier.
enum class VariadicKind : uint8_t { kNone, kPack, kPosVar, kKwVar };

//===----------------------------------------------------------------------===//
// MojoASTDecl Views
//
// The following classes provide simple and JSON-serializable representations of
// the most common Mojo decls. They also include structured documentation from
// the corresponding DocStrings and the ability to summarize themselves as code
// snippets.
//===----------------------------------------------------------------------===//

/// The different kinds of decl views.
enum class DeclViewKind {
  DK_AliasDeclView,
  DK_ArgumentDeclView,
  DK_FunctionDeclView,
  DK_ModuleDeclView,
  DK_PackageDeclView,
  DK_ParameterDeclView,
  DK_StructDeclView,
  DK_StructFieldDeclView,
  DK_TraitDeclView,
  DK_VariableDeclView,
};

/// Base class of all decl views.
class DeclView {
public:
  virtual ~DeclView() = default;

  /// Generate a correct piece of code that summarizes this decl.
  virtual std::string getDeclarationSnippet() const = 0;

  /// Get the name of the decl. It might be empty.
  StringRef getName() const { return name; }

  /// Get a string representation of the kind of decl, e.g., 'variable',
  /// 'function', etc.
  StringRef getKindAsString() const;

  static StringRef getKindAsString(DeclViewKind kind);

  /// Serialize the fields in this view to JSON.
  virtual llvm::json::Object toJSON(MojoParserContext &ctx) const = 0;

  /// Return a nicely formatted markdown docstring of this declaration. It might
  /// be empty if no docstring is available.
  virtual std::string getMarkdownDocString() const { return {}; }

  /// Return a nicely formatted markdown blob containing the declaration snippet
  /// and doc string of the decl.
  std::string getFullMarkdownString() const;

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  DeclViewKind getKind() const { return kind; }

protected:
  DeclView(DeclViewKind kind, StringRef name) : kind(kind), name(name) {}

private:
  DeclViewKind kind;
  StringRef name;
};

/// View for `var` declarations.
class VariableDeclView : public DeclView {
public:
  /// Return if this variable is global.
  bool isGlobal() const { return isGlobalVariable; }

  /// Return the type of this variable.
  StringRef getType() const { return type; }

  std::string getDeclarationSnippet() const override;

  /// The output of the generation is defined in the following schema:
  ///
  ///  {
  ///    "kind": "variable",
  ///    "name": string,
  ///    "type": string
  ///  }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static DeclViewKind getKindStatic() {
    return DeclViewKind::DK_VariableDeclView;
  }

  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  VariableDeclView(MojoASTDeclRef declRef);

  std::string type;
  bool isGlobalVariable;
  StringRef deprecated;
};

/// View for parameters of structs or functions.
class ParameterDeclView : public DeclView {
public:
  ParameterDeclView(StringRef name, StringRef type,
                    KGEN::LIT::PassingKind passingKind,
                    VariadicKind variadicKind,
                    std::optional<std::string> defaultValue)
      : DeclView(DeclViewKind::DK_ParameterDeclView, name), type(type),
        passingKind(passingKind), variadicKind(variadicKind),
        defaultValue(std::move(defaultValue)) {}

  KGEN::LIT::PassingKind getPassingKind() const { return passingKind; };

  std::string getDeclarationSnippet() const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  std::string getMarkdownDocString() const override;

  /// Set the description of this decl.
  void setDescription(StringRef desc) { description = desc; }

  /// The output of the generation is defined in the following schema:
  ///
  ///  {
  ///    "kind": "parameter",
  ///    "description": string,
  ///    "name": string,
  ///    "type": string,
  ///    "passingKind": string,
  ///    "defaultValue": string?
  ///  }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static DeclViewKind getKindStatic() {
    return DeclViewKind::DK_ParameterDeclView;
  }

  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  std::string type;
  KGEN::LIT::PassingKind passingKind;
  VariadicKind variadicKind;
  std::optional<std::string> defaultValue;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
};

/// View for function arguments, including varargs arguments.
class ArgumentDeclView : public DeclView {
public:
  /// The convention this argument is passed with.
  enum class Convention {
    kBorrowed,
    kInOut,
    kOwned,
  };
  ArgumentDeclView(StringRef name, StringRef type,
                   KGEN::LIT::PassingKind passingKind,
                   VariadicKind variadicKind,
                   std::optional<std::string> defaultValue,
                   Convention convention)
      : DeclView(DeclViewKind::DK_ArgumentDeclView, name), type(type),
        passingKind(passingKind), variadicKind(variadicKind),
        defaultValue(std::move(defaultValue)), convention(convention) {}

  std::string getDeclarationSnippet() const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  std::string getMarkdownDocString() const override;

  bool isBorrowed() const { return convention == Convention::kBorrowed; }
  bool isInout() const { return convention == Convention::kInOut; }
  bool isOwned() const { return convention == Convention::kOwned; }

  KGEN::LIT::PassingKind getPassingKind() const { return passingKind; };

  /// Set the description of this decl.
  void setDescription(StringRef desc) { description = desc; }

  /// The output of the generation is defined in the following schema:
  ///
  ///  {
  ///    "kind": "argument",
  ///    "name": string,
  ///    "description": string,
  ///    "convention": string, // "borrowed", "inout", "owned"
  ///    "type": string
  ///    "passingKind": string,
  ///    "defaultValue": string?
  ///  }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static DeclViewKind getKindStatic() {
    return DeclViewKind::DK_ArgumentDeclView;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  std::string type;
  KGEN::LIT::PassingKind passingKind;
  VariadicKind variadicKind;
  std::optional<std::string> defaultValue;
  Convention convention;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
};

/// View for alias decls.
class AliasDeclView : public DeclView {
public:
  std::string getDeclarationSnippet() const override;

  std::string getMarkdownDocString() const override;

  StringRef getValue() const { return value; }

  /// Return if this alias is global.
  bool isGlobal() const { return isGlobalAlias; }

  /// The output of the generation is defined in the following schema:
  ///
  ///  {
  ///    "kind": "alias",
  ///    "name": string,
  ///    "description": string,
  ///    "summary": string,
  ///    "value": string
  ///  }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static DeclViewKind getKindStatic() { return DeclViewKind::DK_AliasDeclView; }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  AliasDeclView(MojoASTDeclRef declRef);

  std::string value;
  bool isGlobalAlias;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  StringRef deprecated;
  std::string description;
  std::string summary;
};

/// View for functions of any kind, including closures, instance methods, fn
/// functions, def functions, etc.
class FunctionDeclView : public DeclView {
public:
  /// Return true if this is an async function.
  bool isAsync() const { return isAsyncFlag; }

  /// Return true if this is a function declared with `def`, and false if it is
  /// declared with `fn`.
  bool isDef() const { return isDefFlag; }

  /// Return true if this is a static struct method, i.e., marked with @static.
  bool isStatic() const { return isStaticFlag; }

  /// Return true if this is a non-static struct method.
  bool isMethod() const { return isMethodFlag; }

  /// Return the list of arguments of this function.
  ArrayRef<ArgumentDeclView> getArguments() const { return args; }

  std::string getDeclarationSnippet() const override;

  /// Get the declaration snippet for the function. The positions of parameters
  /// and arguments within the printed signature may be extracted via the
  /// optional `parameterOffsets` and `argumentOffsets`. nullptr should be
  /// provided if collecting the offsets isn't desired.
  std::string getDeclarationSnippet(
      SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets =
          nullptr) const;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  std::string getMarkdownDocString() const override;

  /// Return the parameters of this function.
  ArrayRef<ParameterDeclView> getParameters() const { return parameters; }

  // TODO: always return the type, including None. The doc and snippet
  // generation will need to be updated accordingly.
  /// Return the type of this function if it is not None.
  std::optional<StringRef> getReturnType() const { return returnType; }

  /// Generate a string for the signature of this function, given its
  /// components. The positions of parameters, arguments, and the return type
  /// within the printed signature may be extracted via the optionally null
  /// `parameterOffsets`, `argumentOffsets`, and `returnOffset` parameters.
  std::string getSignature(
      SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets =
          nullptr,
      SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets = nullptr,
      unsigned *returnOffset = nullptr) const;

  /// Return true if this function raises.
  bool raises() const { return raisesFlag; }

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "function",
  ///   "name": string,
  ///   "args": ArgumentDeclView[],
  ///   "constraints": string,
  ///   "description": string,
  ///   "isDef": boolean,
  ///   "isStatic": boolean,
  ///   "parameters": ParameterDeclView[],
  ///   "raises": boolean,
  ///   "raisesDoc": string,
  ///   "returnsDoc": string,
  ///   "returnType": string,
  ///   "signature": string, // E.g., "baz() -> Int"
  ///   "summary": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static DeclViewKind getKindStatic() {
    return DeclViewKind::DK_FunctionDeclView;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  FunctionDeclView(MojoASTDeclRef declRef);
  FunctionDeclView(MojoASTDeclRef declRef,
                   KGEN::LIT::LITSignatureType signature);

  /// Initialize the function view with the given signature.
  void initFromSignature(MojoASTDeclRef declRef,
                         KGEN::LIT::LITSignatureType signature,
                         ArrayRef<Type> argTypes);

  /// Augment this function view with docstring documentation, as well as its
  /// parameters and args.
  void augmentWithDocumentation(ArrayRef<StringRef> description);

  SmallVector<ArgumentDeclView> args;
  SmallVector<ParameterDeclView> parameters;
  // TODO: Convert this to MojoASTTypeRef.
  std::optional<std::string> returnType;

  //===----------------------------------------------------------------------===//
  // Effects and modifiers
  //===----------------------------------------------------------------------===//

  bool isAsyncFlag = false;
  bool isDefFlag = false;
  bool isMethodFlag = false;
  bool isStaticFlag = false;
  bool raisesFlag = false;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  StringRef deprecated;
  std::string constraints;
  std::string description;
  std::string raisesDoc;
  std::string returnsDoc;
  std::string summary;
};

/// View for struct field.
class StructFieldDeclView : public DeclView {
public:
  StructFieldDeclView(StringRef name, StringRef type)
      : DeclView(DeclViewKind::DK_StructFieldDeclView, name), type(type) {}

  std::string getDeclarationSnippet() const override;

  std::string getMarkdownDocString() const override;

  /// Return the type of this field.
  StringRef getType() const { return type; }

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "field",
  ///   "name": string,
  ///   "description": string,
  ///   "summary": string,
  ///   "type": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static DeclViewKind getKindStatic() {
    return DeclViewKind::DK_StructFieldDeclView;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  StructFieldDeclView(MojoASTDeclRef declRef);

  std::string type;
  std::string value;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
  std::string summary;
};

/// A collection of overloaded functions in the same scope.
class FunctionDeclOverloadSetView {
public:
  /// Create a list of function overload sets by grouping the given functions by
  /// their name. It's assumed that the input list is sorted by name.
  static SmallVector<FunctionDeclOverloadSetView, 2>
  fromSortedFunctions(SmallVector<FunctionDeclView, 2> &&functions);

  /// Get the common name of the functions in this overload set.
  StringRef getBaseName() const { return baseName; }

  /// Get the functions in this overload set.
  ArrayRef<FunctionDeclView> getFunctions() const { return functions; }

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "function",
  ///   "name": string,
  ///   "overloads": FunctionDeclView[]
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const;

  static DeclViewKind getKindStatic() {
    return DeclViewKind::DK_StructFieldDeclView;
  }

private:
  FunctionDeclOverloadSetView(StringRef baseName) : baseName(baseName) {}

  void append(FunctionDeclView function) {
    functions.push_back(std::move(function));
  }

  std::string baseName;
  SmallVector<FunctionDeclView, 2> functions;
};

// View for trait decls.
class TraitDeclView : public DeclView {
public:
  std::string getDeclarationSnippet() const override;

  std::string getMarkdownDocString() const override;

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "trait",
  ///   "name": string,
  ///   "description": string,
  ///   "functions": FunctionDeclOverloadSetView[],
  ///   "parentTraits": string[],
  ///   "summary": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static DeclViewKind getKindStatic() { return DeclViewKind::DK_TraitDeclView; }

  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  TraitDeclView(MojoASTDeclRef declRef);

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  StringRef deprecated;
  std::string description;
  std::string summary;
  MojoASTDeclRef decl;
};

/// View for struct decls.
class StructDeclView : public DeclView {
public:
  std::string getDeclarationSnippet() const override;

  /// Get the declaration snippet for the struct. The positions of parameters
  /// within the printed snippet may be extracted via `parameterOffsets`, which
  /// may be null if parameter offsets are not desired.
  std::string getDeclarationSnippet(
      SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets) const;

  std::string getMarkdownDocString() const override;

  /// Return the parameters of this struct.
  ArrayRef<ParameterDeclView> getParameters() const { return parameters; }

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "struct",
  ///   "name": string,
  ///   "aliases": AliasDeclView[],
  ///   "description": string,
  ///   "functions": FunctionDeclOverloadSetView[],
  ///   "parameters": ParameterDeclView[],
  ///   "parentTraits": string[],
  ///   "fields": StructFieldDeclView[],
  ///   "summary": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static DeclViewKind getKindStatic() {
    return DeclViewKind::DK_StructDeclView;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  StructDeclView(MojoASTDeclRef declRef);

  /// Augment this struct view with docstring documentation, as well as its
  /// parameters.
  void augmentWithDocumentation(ArrayRef<StringRef> description);

  SmallVector<ParameterDeclView> parameters;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  StringRef deprecated;
  std::string constraints;
  std::string description;
  std::string summary;
  MojoASTDeclRef decl;
};

/// View for module decls.
class ModuleDeclView : public DeclView {
public:
  std::string getDeclarationSnippet() const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  std::string getMarkdownDocString() const override;

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "module",
  ///   "name": string,
  ///   "aliases": AliasDeclView[],
  ///   "description": string,
  ///   "functions": FunctionDeclOverloadSetView[],
  ///   "structs": StructDeclView[],
  ///   "traits": TraitDeclView[],
  ///   "summary": string
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static DeclViewKind getKindStatic() {
    return DeclViewKind::DK_ModuleDeclView;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  ModuleDeclView(MojoASTDeclRef declRef);

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
  std::string summary;
  MojoASTDeclRef decl;
};

class PackageDeclView : public DeclView {
public:
  std::string getDeclarationSnippet() const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  std::string getMarkdownDocString() const override;

  /// The output of the generation is defined in the following schema:
  ///
  /// {
  ///   "kind": "package",
  ///   "name": string,
  ///   "description": string,
  ///   "summary": string,
  ///   "modules": ModuleDeclView[],
  ///   "packages": PackageDeclView[],
  /// }
  llvm::json::Object toJSON(MojoParserContext &ctx) const override;

  static DeclViewKind getKindStatic() {
    return DeclViewKind::DK_PackageDeclView;
  }

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == getKindStatic();
  }

private:
  friend class MojoASTDeclRef;

  PackageDeclView(MojoASTDeclRef declRef);

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
  std::string summary;
  MojoASTDeclRef decl;
};

} // namespace M

#endif // KGEN_MOJOTOOLING_ASTDECLVIEW_H
