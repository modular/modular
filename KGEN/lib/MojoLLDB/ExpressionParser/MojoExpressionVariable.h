//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONVARIABLE_H
#define KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONVARIABLE_H

#include "JITExecutionUnit.h"
#include "MojoUserExpression.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Symbol/TaggedASTType.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-public.h"
#include "llvm/Support/Casting.h"

namespace M::KGEN::Mojo {
//===----------------------------------------------------------------------===//
// MojoExpressionVariable
//===----------------------------------------------------------------------===//

/// This class represents a single Mojo expression variable.
class MojoExpressionVariable : public lldb_private::ExpressionVariable {
public:
  MojoExpressionVariable(lldb_private::ExecutionContextScope *exeScope,
                         lldb::ByteOrder byteOrder, uint32_t addrByteSize);
  MojoExpressionVariable(const lldb::ValueObjectSP &valobj);
  MojoExpressionVariable(lldb_private::ExecutionContextScope *exeScope,
                         lldb_private::ConstString name,
                         const lldb_private::TypeFromUser &userType,
                         lldb::ByteOrder byteOrder, uint32_t addrByteSize);

  // Prevent copying.
  MojoExpressionVariable(const MojoExpressionVariable &) = delete;
  const MojoExpressionVariable &
  operator=(const MojoExpressionVariable &) = delete;

  //--------------------------------------------------------------------------//
  // llvm casting support
  //--------------------------------------------------------------------------//

  static LLVMCastKind classofKind() {
    // TODO: MojoExpressionVariable should use a more open casting mechanism,
    // but for now just pretend we are Go.
    return LLVMCastKind::eKindGo;
  }

  static bool classof(const ExpressionVariable *ev) {
    return ev->getKind() == classofKind();
  }
};

//===----------------------------------------------------------------------===//
// MojoPersistentExpressionState
//===----------------------------------------------------------------------===//

/// This class manages persistent values that need to be preserved between
/// Mojo expression invocations.
class MojoPersistentExpressionState
    : public lldb_private::PersistentExpressionState {
public:
  MojoPersistentExpressionState() : PersistentExpressionState(classofKind()) {}
  ~MojoPersistentExpressionState() override = default;

  //===--------------------------------------------------------------------===//
  // Expression Instance
  //===--------------------------------------------------------------------===//

  /// This struct represents all of the state related to a single successful
  /// expression evaluation.
  struct ExpressionInstanceState {
    ExpressionInstanceState(std::shared_ptr<JITExecutionUnit> executionUnit,
                            std::vector<lldb::ExpressionVariableSP> &&variables,
                            std::optional<MojoExpressionSourceCode> sourceCode,
                            std::optional<std::string> pythonModuleName)
        : executionUnit(std::move(executionUnit)),
          persistentVariables(std::move(variables)),
          sourceCode(std::move(sourceCode)),
          pythonModuleName(std::move(pythonModuleName)) {}

    /// An optional execution unit associated with the expression, present only
    /// when JIT symbols must be persisted.
    std::shared_ptr<JITExecutionUnit> executionUnit;

    /// The persistent variables added during the execution of the expression.
    std::vector<lldb::ExpressionVariableSP> persistentVariables;

    /// The source code after fix-its.
    std::optional<MojoExpressionSourceCode> sourceCode;

    /// The name of the python module represented by the expression, if it was
    /// a python expression, nullopt if it was a mojo expression.
    std::optional<std::string> pythonModuleName;
  };

  /// Returns the number of expression instances.
  size_t getNumExpressionInstances() const {
    return expressionInstances.size();
  }

  /// Returns the expression instances persisted within the state.
  auto getExpressionInstances() const {
    return llvm::make_pointee_range(expressionInstances);
  }

  /// Register a new expression instance.
  void registerExpressionInstance(
      std::shared_ptr<JITExecutionUnit> executionUnit,
      std::vector<lldb::ExpressionVariableSP> &&variables,
      std::optional<MojoExpressionSourceCode> sourceCode,
      std::optional<std::string> pythonModuleName);

  /// Reset the expression state to before the  instance at the provided index.
  void resetStateToBeforeExpressionInstance(size_t index);

  //===--------------------------------------------------------------------===//
  // Python Expression State
  //===--------------------------------------------------------------------===//

  /// Returns true if python is known to have already been initialized by the
  /// persistent expression state.
  bool hasInitializedPython() const;

  /// Return the prefix of all python expression modules.
  static StringRef getPythonExpressionModuleNamePrefix() {
    return "__lldb_python_module_";
  }

  /// Return the next name to use for a Python expression module.
  std::string getNextPythonExpressionModuleName();

  //===--------------------------------------------------------------------===//
  // PersistentExpressionState
  //===--------------------------------------------------------------------===//

  lldb::ExpressionVariableSP
  CreatePersistentVariable(const lldb::ValueObjectSP &valobj) override;

  lldb::ExpressionVariableSP
  CreatePersistentVariable(lldb_private::ExecutionContextScope *exeScope,
                           lldb_private::ConstString name,
                           const lldb_private::CompilerType &compilerType,
                           lldb::ByteOrder byteOrder,
                           uint32_t addrByteSize) override;

  llvm::StringRef GetPersistentVariablePrefix(bool isError) const override {
    // TODO: This is a placeholder, and should be replaced when we actually
    // support persistent variables.
    return isError ? "$E" : "$R";
  }

  void RemovePersistentVariable(lldb::ExpressionVariableSP variable) override {
    RemoveVariable(variable);
  }

  lldb_private::ConstString
  GetNextPersistentVariableName(bool isError = false) override {
    return lldb_private::ConstString("");
  }

  std::optional<lldb_private::CompilerType> GetCompilerTypeFromPersistentDecl(
      lldb_private::ConstString typeName) override {
    return std::nullopt;
  }

  /// Lookup a symbol with the provided name.
  lldb::addr_t LookupSymbol(lldb_private::ConstString name) override;

  //===--------------------------------------------------------------------===//
  // RTTI support
  //===--------------------------------------------------------------------===//

  static LLVMCastKind classofKind() {
    // TODO: PersistentExpressionState should use a more open casting mechanism,
    // but for now just pretend we are Go.
    return LLVMCastKind::eKindGo;
  }

  static bool classof(const PersistentExpressionState *pv) {
    return pv->getKind() == classofKind();
  }

private:
  /// Instance state associated with successful expression evaluations.
  std::vector<std::unique_ptr<ExpressionInstanceState>> expressionInstances;

  /// The addresses of the symbols in executionUnits.
  llvm::StringMap<lldb::addr_t> symbolMap;

  /// The next identifier to use when building a python expression module.
  size_t nextPythonModuleID = 0;
};
} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_EXPRESSIONPARSER_MOJOEXPRESSIONVARIABLE_H
