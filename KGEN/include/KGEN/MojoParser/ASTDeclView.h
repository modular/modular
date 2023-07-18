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

  /// Get the description of this decl extracted from its docstring.
  std::optional<StringRef> getDescription() const;

  /// Get the name of the decl. It might be empty.
  StringRef getName() const { return name; }

  /// Get the type of the decl. It might be empty.
  StringRef getType() const { return type; }

  /// Set the description of this decl.
  void setDescription(StringRef desc) { description = desc; }

  /// Serialize the fields in this view to JSON.
  virtual llvm::json::Object toJSON() const;

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  enum DeclViewKind {
    DK_ParameterDeclView,
    DK_ArgumentDeclView,
    DK_FunctionDeclView
  };

  DeclViewKind getKind() const { return kind; }

protected:
  DeclView(DeclViewKind kind, StringRef name, StringRef type = {})
      : kind(kind), name(name), type(type) {}

private:
  const DeclViewKind kind;
  StringRef name;
  // TODO: convert to MojoASTTypeRef.
  std::string type;
  // TODO: convert into StringRef.
  std::optional<std::string> description;
};

/// View for parameters of structs or functions.
class ParameterDeclView : public DeclView {
public:
  ParameterDeclView(StringRef name, StringRef type)
      : DeclView(DK_ParameterDeclView, name, type){};

  std::string getDeclarationSnippet() const override;

  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == DK_ParameterDeclView;
  }
};

/// View for function arguments, including varargs arguments.
class ArgumentDeclView : public DeclView {
public:
  ArgumentDeclView(StringRef name, StringRef type, bool inout)
      : DeclView(DK_ArgumentDeclView, name, type), inout(inout) {}

  std::string getDeclarationSnippet() const override;

  bool isInout() const { return inout; }

  /// Serialize the fields in this view to JSON.
  llvm::json::Object toJSON() const override;

public:
  //===----------------------------------------------------------------------===//
  // LLVM RTTI Support
  //===----------------------------------------------------------------------===//

  static bool classof(const DeclView *decl) {
    return decl->getKind() == DK_ArgumentDeclView;
  }

private:
  bool inout;
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

  /// Return a nicely formatted markdown docstring of this declaration. It might
  /// be empty if no docstring is available.
  std::string getMarkdownDocString() const;

  /// Return the parameters of arguments of this function.
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

  std::optional<std::string> summary;
  std::optional<std::string> returns;
  std::optional<std::string> constraints;
};

} // namespace M

#endif // KGEN_MOJOPARSER_ASTDECLVIEW_H
