//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoExpressionVariable.h"
#include "JITExecutionUnit.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Utility/LLDBLog.h"

using namespace M::KGEN::Mojo;
using namespace lldb_private;

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

void MojoPersistentExpressionState::registerExecutionUnit(
    std::shared_ptr<JITExecutionUnit> &executionUnit) {
  Log *log = GetLog(LLDBLog::Expressions);
  executionUnits.insert(executionUnit);

  LLDB_LOGF(log, "Registering JITted Functions:\n");
  for (const auto &jittedFunction : executionUnit->getJittedFunctions()) {
    if (jittedFunction.external &&
        jittedFunction.name != executionUnit->getFunctionName() &&
        jittedFunction.remoteAddr != LLDB_INVALID_ADDRESS) {
      symbolMap[jittedFunction.name.GetStringRef()] = jittedFunction.remoteAddr;
      LLDB_LOGF(log, "  Function: %s at 0x%" PRIx64 ".",
                jittedFunction.name.GetCString(), jittedFunction.remoteAddr);
    }
  }

  LLDB_LOGF(log, "Registering JIIted Symbols:\n");
  for (const auto &globalVar : executionUnit->getJittedGlobalVariables()) {
    if (globalVar.remoteAddr != LLDB_INVALID_ADDRESS) {
      symbolMap[globalVar.name.GetStringRef()] = globalVar.remoteAddr;
      LLDB_LOGF(log, "  Symbol: %s at 0x%" PRIx64 ".",
                globalVar.name.GetCString(), globalVar.remoteAddr);
    }
  }
}
