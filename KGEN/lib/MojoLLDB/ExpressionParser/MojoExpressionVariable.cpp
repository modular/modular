//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoExpressionVariable.h"
#include "JITExecutionUnit.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Utility/LLDBLog.h"

using namespace M;
using namespace M::KGEN::Mojo;
using namespace lldb_private;

using JittedEntity = JITExecutionUnit::JittedEntity;

//===----------------------------------------------------------------------===//
// MojoExpressionVariable
//===----------------------------------------------------------------------===//

MojoExpressionVariable::MojoExpressionVariable(ExecutionContextScope *exeScope,
                                               lldb::ByteOrder byteOrder,
                                               uint32_t addrByteSize)
    : ExpressionVariable(classofKind()) {
  m_frozen_sp =
      ValueObjectConstResult::Create(exeScope, byteOrder, addrByteSize);
}

MojoExpressionVariable::MojoExpressionVariable(
    const lldb::ValueObjectSP &valobj)
    : ExpressionVariable(classofKind()) {
  m_frozen_sp = valobj;
}

MojoExpressionVariable::MojoExpressionVariable(ExecutionContextScope *exeScope,
                                               ConstString name,
                                               const TypeFromUser &type,
                                               lldb::ByteOrder byteOrder,
                                               uint32_t addrByteSize)
    : ExpressionVariable(classofKind()) {
  m_frozen_sp =
      ValueObjectConstResult::Create(exeScope, byteOrder, addrByteSize);
  SetName(name);
  SetCompilerType(type);
}

//===----------------------------------------------------------------------===//
// MojoPersistentExpressionState
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Expression Instance

/// Walk the external JIT symbols within the given execution unit, invoking the
/// provided callback for each.
static void walkExternalJITSymbols(
    JITExecutionUnit &executionUnit, Log *log,
    function_ref<void(const JITExecutionUnit::JittedEntity &)> callback) {
  LLDB_LOGF(log, "Processing JITted Functions:\n");
  for (const auto &jitSym : executionUnit.getJittedFunctions()) {
    if (jitSym.external && jitSym.name != executionUnit.getFunctionName() &&
        jitSym.remoteAddr != LLDB_INVALID_ADDRESS) {
      LLDB_LOGF(log, "  Function: %s at 0x%" PRIx64 ".",
                jitSym.name.GetCString(), jitSym.remoteAddr);
      callback(jitSym);
    }
  }

  LLDB_LOGF(log, "Processing JIIted Symbols:\n");
  for (const auto &jitSym : executionUnit.getJittedGlobalVariables()) {
    if (jitSym.remoteAddr != LLDB_INVALID_ADDRESS) {
      LLDB_LOGF(log, "  Symbol: %s at 0x%" PRIx64 ".", jitSym.name.GetCString(),
                jitSym.remoteAddr);
      callback(jitSym);
    }
  }
}

void MojoPersistentExpressionState::registerExpressionInstance(
    std::shared_ptr<JITExecutionUnit> executionUnit,
    std::vector<lldb::ExpressionVariableSP> &&variables,
    std::optional<MojoExpressionSourceCode> sourceCode,
    std::optional<std::string> pythonModuleName) {
  Log *log = GetLog(LLDBLog::Expressions);

  // Register the JIT symbols within the execution unit.
  if (executionUnit) {
    auto walkFn = [&](const JittedEntity &jitSym) {
      symbolMap[jitSym.name.GetStringRef()] = jitSym.remoteAddr;
    };
    walkExternalJITSymbols(*executionUnit, log, walkFn);
  }

  // Push a new expression state.
  expressionInstances.emplace_back(std::make_unique<ExpressionInstanceState>(
      std::move(executionUnit), std::move(variables), std::move(sourceCode),
      std::move(pythonModuleName)));
}

void MojoPersistentExpressionState::resetStateToBeforeExpressionInstance(
    size_t index) {
  assert(index < getNumExpressionInstances() && "invalid expression instance");
  Log *log = GetLog(LLDBLog::Expressions);

  // Drop each of the instance states in reverse up to the given index.
  for (auto &exprInst :
       llvm::reverse(llvm::drop_begin(expressionInstances, index))) {
    // Drop all of the JIT symbols that we previously registered.
    if (exprInst->executionUnit) {
      auto walkFn = [&](const JittedEntity &jitSym) {
        symbolMap.erase(jitSym.name.GetStringRef());
      };
      walkExternalJITSymbols(*exprInst->executionUnit, log, walkFn);
    }

    // Drop the persistent variables.
    for (const lldb::ExpressionVariableSP &var : exprInst->persistentVariables)
      RemoveVariable(var);
  }
  expressionInstances.resize(index);
}

//===----------------------------------------------------------------------===//
// Python Expression State

bool MojoPersistentExpressionState::hasInitializedPython() const {
  return llvm::any_of(expressionInstances, [](const auto &exprInst) {
    return exprInst->pythonModuleName.has_value();
  });
}

std::string MojoPersistentExpressionState::getNextPythonExpressionModuleName() {
  return "lldb_python_module_" + std::to_string(nextPythonModuleID++);
}

//===----------------------------------------------------------------------===//
// PersistentExpressionState

lldb::ExpressionVariableSP
MojoPersistentExpressionState::CreatePersistentVariable(
    const lldb::ValueObjectSP &valobj) {
  return AddNewlyConstructedVariable(new MojoExpressionVariable(valobj));
}

lldb::ExpressionVariableSP
MojoPersistentExpressionState::CreatePersistentVariable(
    ExecutionContextScope *exeScope, ConstString name,
    const CompilerType &compilerType, lldb::ByteOrder byteOrder,
    uint32_t addrByteSize) {
  return AddNewlyConstructedVariable(new MojoExpressionVariable(
      exeScope, name, compilerType, byteOrder, addrByteSize));
}

lldb::addr_t MojoPersistentExpressionState::LookupSymbol(ConstString name) {
  auto si = symbolMap.find(name.GetStringRef());
  if (si != symbolMap.end())
    return si->second;
  return PersistentExpressionState::LookupSymbol(name);
}
