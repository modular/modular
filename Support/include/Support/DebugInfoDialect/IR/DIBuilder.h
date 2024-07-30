//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_IR_DIBUILDER_H
#define SUPPORT_DEBUGINFODIALECT_IR_DIBUILDER_H

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"

//===----------------------------------------------------------------------===//
// DIBuilder
//===----------------------------------------------------------------------===//

namespace M::DebugInfo {
/// This class provides a high level builder abstraction that greatly simplifies
/// the work needed to build debug info constructs for frontend users.
class DIBuilder {
public:
  DIBuilder(MLIRContext *context) : context(context) {}

  /// Return the context held by this builder.
  MLIRContext *getContext() const { return context; }

  /// Initialize the compilation unit for the builder. The compilation unit
  /// provides an anchor for all of the debug information generated during
  /// compilation.
  DICompileUnitAttr
  initializeCompileUnit(unsigned sourceLanguage, DIFileAttr file,
                        StringRef producer, bool isOptimized,
                        EmissionKind emissionKind,
                        NameTableKind nameTableKind = NameTableKind::None);

  //===--------------------------------------------------------------------===//
  // Scopes
  //===--------------------------------------------------------------------===//

  /// This class provides an RAII guard for pushing and popping debug info
  /// scopes.
  class [[nodiscard]] ScopeGuard {
  public:
    ScopeGuard() = default;
    ScopeGuard(const ScopeGuard &) = delete;
    ScopeGuard &operator=(const ScopeGuard &) = delete;

    ScopeGuard(DIBuilder &builder, DIScopeAttr scope)
        : builder(&builder), scope(scope) {
      builder.pushScope(scope);
    }
    ScopeGuard(ScopeGuard &&rhs) : builder(rhs.builder), scope(rhs.scope) {
      rhs.builder = nullptr;
    }
    ScopeGuard &operator=(ScopeGuard &&rhs) {
      builder = rhs.builder;
      scope = rhs.scope;
      rhs.builder = nullptr;
      return *this;
    }

    ~ScopeGuard() {
      if (builder)
        builder->popScope();
    }

  private:
    DIBuilder *builder = nullptr;
    DIScopeAttr scope;
  };

  /// Push a new debuginfo scope onto the stack. This scope will be used when
  /// creating debug info constructs that require a parent scope.
  void pushScope(DIScopeAttr scope);

  /// Push a guarded scope to the stack.
  ScopeGuard pushScopeGuard(DIScopeAttr scope) {
    return ScopeGuard(*this, scope);
  }

  /// Pop the current debuginfo scope off the stack.
  void popScope();

  /// Augment the given location to include the current scope information.
  Location createScopedLoc(Location loc);

  //===--------------------------------------------------------------------===//
  // Creation
  //===--------------------------------------------------------------------===//

  /// Create a new lexical block scope whose parent scope is the current scope
  /// on the stack within the builder. If the parent scope is null, returns
  /// null.
  DILexicalBlockAttr createNestedLexicalBlock(DIFileAttr file, unsigned line,
                                              unsigned column);
  /// Create a nested lexical block scope using `createNestedLexicalBlock` and
  /// push it onto the stack with a scope guard.
  ScopeGuard pushNestedLexicalBlock(DIFileAttr file, unsigned line,
                                    unsigned column) {
    return pushScopeGuard(createNestedLexicalBlock(file, line, column));
  }

  /// Create a new subprogram. The parent scope is the current scope on the
  /// stack within the builder.
  DISubprogramAttr createSubprogram(SourceNameAttr name, StringAttr linkageName,
                                    DIFileAttr file, unsigned line,
                                    unsigned scopeLine,
                                    SubprogramFlags subprogramFlags,
                                    DISubroutineType type);
  /// Create and push a new subprogram scope.
  ScopeGuard pushSubprogram(SourceNameAttr name, StringAttr linkageName,
                            DIFileAttr file, unsigned line, unsigned scopeLine,
                            SubprogramFlags subprogramFlags,
                            DISubroutineType type) {
    return pushScopeGuard(createSubprogram(name, linkageName, file, line,
                                           scopeLine, subprogramFlags, type));
  }

  /// Create a new file.
  DIFileAttr createFile(StringRef name, StringRef directory);
  DIFileAttr createFile(FileLineColLoc loc);
  /// Create and push a new file scope onto the stack.
  ScopeGuard pushFile(StringRef name, StringRef directory) {
    return pushScopeGuard(createFile(name, directory));
  }

  /// Create a new local variable. The parent scope is the current scope on the
  /// stack within the builder.
  DILocalVariableAttr createLocalVariable(StringRef name, DIFileAttr file,
                                          unsigned line, unsigned arg,
                                          unsigned alignInBits, DIType type,
                                          DIFlags flags = DIFlags::Zero);

private:
  MLIRContext *context;
  DICompileUnitAttr compileUnit;
  /// A null scope in the stack signifies a local scope that does not want any
  /// debug info for ops inside. As long as a null scope is at the top, any
  /// additional lexical blocks pushed are also null.
  SmallVector<DIScopeAttr> scopes;
};
} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_IR_DIBUILDER_H
