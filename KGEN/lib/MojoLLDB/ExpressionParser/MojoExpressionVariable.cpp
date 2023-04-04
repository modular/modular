//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoExpressionVariable.h"
#include "lldb/Core/ValueObjectConstResult.h"

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
