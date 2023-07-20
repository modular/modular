//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_ASTDECLVIEW_H
#define KGEN_MOJOPARSER_ASTDECLVIEW_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace llvm {
namespace json {
class Object;
} // namespace json
} // namespace llvm

namespace M {

class MojoASTDeclRef;

//===----------------------------------------------------------------------===//
// MojoASTDecl Views
//
// The following classes provide simple and JSON-serializable representations of
// the most common Mojo decls. They also include structured documentation from
// the corresponding DocStrings and the ability to summarize themselves as code
// snippets.
//===----------------------------------------------------------------------===//

/// Base class of all decl views.
class DeclView {
public:
  virtual ~DeclView() = default;

  /// Generate a correct piece of code that summarizes this decl.
  virtual std::string getDeclarationSnippet() const = 0;

  /// Get the name of the decl. It might be empty.
  StringRef getName() const { return name; }

  /// Serialize the fields in this view to JSON.
  virtual llvm::json::Object toJSON() const = 0;

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  enum DeclViewKind {
    DK_AliasDeclView,
    DK_ArgumentDeclView,
    DK_FunctionDeclView,
    DK_ModuleDeclView,
    DK_ParameterDeclView,
    DK_StructDeclView,
    DK_StructFieldDeclView,
  };

  DeclViewKind getKind() const { return kind; }

protected:
  DeclView(DeclViewKind kind, StringRef name) : kind(kind), name(name) {}

private:
  DeclViewKind kind;
  StringRef name;
};

/// View for parameters of structs or functions.
class ParameterDeclView : public DeclView {
public:
  ParameterDeclView(StringRef name, StringRef type)
      : DeclView(DK_ParameterDeclView, name), type(type){};

  std::string getDeclarationSnippet() const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  /// Set the description of this decl.
  void setDescription(StringRef desc) { description = desc; }

  llvm::json::Object toJSON() const override;

  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == DK_ParameterDeclView;
  }

private:
  std::string type;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
};

/// View for function arguments, including varargs arguments.
class ArgumentDeclView : public DeclView {
public:
  ArgumentDeclView(StringRef name, StringRef type, bool inout)
      : DeclView(DK_ArgumentDeclView, name), type(type), inout(inout) {}

  std::string getDeclarationSnippet() const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  bool isInout() const { return inout; }

  /// Set the description of this decl.
  void setDescription(StringRef desc) { description = desc; }

  llvm::json::Object toJSON() const override;

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == DK_ArgumentDeclView;
  }

private:
  std::string type;
  bool inout;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
};

/// View for alias decls.
class AliasDeclView : public DeclView {
public:
  std::string getDeclarationSnippet() const override;

  StringRef getValue() const { return value; }

  llvm::json::Object toJSON() const override;

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == DK_AliasDeclView;
  }

private:
  friend class MojoASTDeclRef;

  AliasDeclView(MojoASTDeclRef declRef);

  std::string value;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

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

  /// Return the list of arguments of this function.
  ArrayRef<ArgumentDeclView> getArgs() const { return args; }

  std::string getDeclarationSnippet() const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  /// Return a nicely formatted markdown docstring of this declaration. It might
  /// be empty if no docstring is available.
  std::string getMarkdownDocString() const;

  /// Return the parameters of this function.
  ArrayRef<ParameterDeclView> getParameters() const { return parameters; }

  // TODO: always return the type, including None. The doc and snippet
  // generation will need to be updated accordingly.
  /// Return the type of this function if it is not None.
  std::optional<StringRef> getReturnType() const { return returnType; }

  /// Generate a string for the signature of this function, given its
  /// components.
  std::string getSignature() const;

  /// Return true if this function raises.
  bool raises() const { return raisesFlag; }

  llvm::json::Object toJSON() const override;

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == DK_FunctionDeclView;
  }

private:
  friend class MojoASTDeclRef;

  FunctionDeclView(MojoASTDeclRef declRef);

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

  bool isAsyncFlag;
  bool isDefFlag;
  bool raisesFlag;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string constraints;
  std::string description;
  std::string returns;
  std::string summary;
};

/// View for struct field.
class StructFieldDeclView : public DeclView {
public:
  StructFieldDeclView(StringRef name, StringRef type)
      : DeclView(DK_StructFieldDeclView, name), type(type) {}

  std::string getDeclarationSnippet() const override;

  llvm::json::Object toJSON() const override;

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == DK_StructFieldDeclView;
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
class FunctionDeclViewOverloadSet {
public:
  /// Create a list of function overload sets by grouping the given functions by
  /// their name. It's assumed that the input list is sorted by name.
  static SmallVector<FunctionDeclViewOverloadSet, 2>
  fromSortedFunctions(SmallVector<FunctionDeclView, 2> &&functions);

  /// Get the common name of the functions in this overload set.
  StringRef getBaseName() const { return baseName; }

  /// Get the functions in this overload set.
  ArrayRef<FunctionDeclView> getFunctions() const { return functions; }

  /// Serialize the fields in this view to JSON.
  llvm::json::Object toJSON() const;

private:
  FunctionDeclViewOverloadSet(StringRef baseName) : baseName(baseName) {}

  void append(FunctionDeclView function) {
    functions.push_back(std::move(function));
  }

  std::string baseName;
  SmallVector<FunctionDeclView, 2> functions;
};

/// View for struct decls.
class StructDeclView : public DeclView {
public:
  /// Return the aliases defined at the top-level of this module.
  llvm::ArrayRef<AliasDeclView> getAliases() const { return aliases; }

  /// Return the fields of this struct.
  ArrayRef<StructFieldDeclView> getFields() const { return fields; }

  std::string getDeclarationSnippet() const override;

  /// Return the parameters of this struct.
  ArrayRef<ParameterDeclView> getParameters() const { return parameters; }

  llvm::json::Object toJSON() const override;

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == DK_StructDeclView;
  }

private:
  friend class MojoASTDeclRef;

  StructDeclView(MojoASTDeclRef declRef);

  /// Augment this struct view with docstring documentation, as well as its
  /// parameters.
  void augmentWithDocumentation(ArrayRef<StringRef> description);

  SmallVector<AliasDeclView> aliases;
  SmallVector<StructFieldDeclView> fields;
  SmallVector<ParameterDeclView> parameters;
  SmallVector<FunctionDeclViewOverloadSet, 2> functionOverloads;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string constraints;
  std::string description;
  std::string summary;
};

/// View for module decls.
class ModuleDeclView : public DeclView {
public:
  /// Return the aliases defined at the top-level of this module.
  llvm::ArrayRef<AliasDeclView> getAliases() const { return aliases; }

  std::string getDeclarationSnippet() const override;

  /// Get the description of this decl extracted from its docstring. It might be
  /// empty.
  StringRef getDescription() const { return description; }

  /// Return the structs defined at the top-level of this module.
  llvm::ArrayRef<StructDeclView> getStructs() const { return structs; }

  llvm::json::Object toJSON() const override;

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == DK_ModuleDeclView;
  }

private:
  friend class MojoASTDeclRef;

  ModuleDeclView(MojoASTDeclRef declRef);

  SmallVector<AliasDeclView> aliases;
  SmallVector<StructDeclView, 2> structs;
  SmallVector<FunctionDeclViewOverloadSet, 2> functionOverloads;

  //===----------------------------------------------------------------------===//
  // Parsed DocString
  //===----------------------------------------------------------------------===//

  std::string description;
  std::string summary;
};

} // namespace M

#endif // KGEN_MOJOPARSER_ASTDECLVIEW_H
